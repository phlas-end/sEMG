from esp32_client import send_sample
from connection_config import resolve_endpoint
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from emg_pipeline import apply_filters, is_rest_window, scale_signal, segment_continuous_signal
from model import EMG2DCNN
from myo_runtime import MyoEMGCollector




def load_model(cfg, checkpoint_path, device):
    model = EMG2DCNN(
        input_shape=(1, cfg["data"]["window"], cfg["data"]["channel"]),
        model_cfg=cfg["model"],
        num_classes=cfg["experiment"]["num_classes"],
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_window(model, device, window):
    sample = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(sample)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    return pred, confidence, probs


def send_window(server_ip, server_port, window):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((server_ip, server_port))
        sock.sendall(window.astype(np.float32, copy=False).tobytes())
        recv_bytes = sock.recv(4)
    return int.from_bytes(recv_bytes, byteorder="little", signed=True)


def main():
    parser = argparse.ArgumentParser(description="Run dual realtime inference on the same Myo stream.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--server-ip", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=100)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(args.config).resolve())

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint or cfg["deploy"]["checkpoint"])
    try:
        server_ip, server_port = resolve_endpoint(args.server_ip, args.server_port, cfg.get("deploy"))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    model = load_model(cfg, checkpoint_path, device)

    collector = MyoEMGCollector(cfg)
    print("waiting for Myo connection...")
    if not collector.wait_for_connection(timeout_seconds=cfg["myo"].get("connect_timeout", 15.0)):
        raise RuntimeError("Myo connection timeout")

    window_index = 0

    while window_index < args.max_windows:
        chunk = collector.collect_duration(args.chunk_seconds)
        if chunk.shape[0] < cfg["data"]["window"]:
            continue

        windows = segment_continuous_signal(
            chunk,
            window=cfg["data"]["window"],
            step=cfg["data"]["step"],
        )
        if windows.shape[0] == 0:
            continue

        for window in windows:
            sample = apply_filters(window, cfg) if cfg["collect"].get("apply_filters", False) else window
            is_rest, metrics = is_rest_window(sample, cfg)
            if is_rest:
                python_pred = 0
                esp_pred = 0
            else:
                scaled = scale_signal(sample, cfg)
                python_pred, _, _ = predict_window(model, device, scaled)
                esp_pred = send_window(server_ip, server_port, scaled)

            window_index += 1

            print(f"window={window_index}, py={python_pred}, esp={esp_pred}")

            if window_index >= args.max_windows:
                break


if __name__ == "__main__":
    main()
