import argparse
import csv
import os
import socket
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from emg_pipeline import apply_filters, is_rest_window, scale_signal, segment_continuous_signal
from model import EMG2DCNN
from myo_runtime import MyoEMGCollector


def find_latest_best_pt(log_dir):
    candidates = []
    for p in Path(log_dir).iterdir():
        if not p.is_dir():
            continue
        best_pt = p / "checkpoints" / "best.pt"
        if best_pt.exists():
            candidates.append(best_pt)

    if not candidates:
        raise FileNotFoundError(f"no best.pt found under {log_dir}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


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


def save_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_index",
                "python_pred",
                "python_stable",
                "python_conf",
                "esp_pred",
                "esp_stable",
                "abs_mean",
                "rms",
                "active_channels",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run dual realtime inference on the same Myo stream.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--server-ip", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=100)
    parser.add_argument("--vote-size", type=int, default=5)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(args.config).resolve())

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_best_pt(cfg["experiment"]["log_dir"])
    server_ip = args.server_ip or cfg["deploy"]["server_ip"]
    server_port = args.server_port or cfg["deploy"]["server_port"]

    print(f"using checkpoint: {checkpoint_path}")
    print(f"using device: {device}")
    print(f"using ESP32 endpoint: {server_ip}:{server_port}")

    model = load_model(cfg, checkpoint_path, device)

    collector = MyoEMGCollector(cfg)
    print("waiting for Myo connection...")
    if not collector.wait_for_connection(timeout_seconds=cfg["myo"].get("connect_timeout", 15.0)):
        raise RuntimeError("Myo connection timeout")

    python_votes = deque(maxlen=args.vote_size)
    esp_votes = deque(maxlen=args.vote_size)
    rows = []
    window_index = 0
    same_count = 0
    non_rest_count = 0

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
                python_pred = "rest"
                python_conf = 1.0
                esp_pred = "rest"
            else:
                scaled = scale_signal(sample, cfg)
                python_pred, python_conf, _ = predict_window(model, device, scaled)
                esp_pred = send_window(server_ip, server_port, scaled)
                non_rest_count += 1
                if python_pred == esp_pred:
                    same_count += 1

            python_votes.append(python_pred)
            esp_votes.append(esp_pred)
            python_stable = Counter(python_votes).most_common(1)[0][0]
            esp_stable = Counter(esp_votes).most_common(1)[0][0]
            window_index += 1

            rows.append(
                {
                    "window_index": window_index,
                    "python_pred": python_pred,
                    "python_stable": python_stable,
                    "python_conf": python_conf,
                    "esp_pred": esp_pred,
                    "esp_stable": esp_stable,
                    "abs_mean": metrics["abs_mean"],
                    "rms": metrics["rms"],
                    "active_channels": metrics["active_channels"],
                }
            )

            print(
                f"window={window_index}, py={python_pred}, py_stable={python_stable}, "
                f"esp={esp_pred}, esp_stable={esp_stable}, conf={python_conf:.3f}, "
                f"abs_mean={metrics['abs_mean']:.3f}, rms={metrics['rms']:.3f}, "
                f"active_channels={metrics['active_channels']}"
            )

            if window_index >= args.max_windows:
                break

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join("runs", "experiments", f"dual_realtime_{timestamp}.csv")
    save_rows(out_path, rows)
    print(f"saved dual realtime results to {out_path}")
    if non_rest_count:
        print(f"non-rest agreement: {same_count}/{non_rest_count} ({same_count / non_rest_count:.4f})")
    else:
        print("all windows were rest")


if __name__ == "__main__":
    main()
