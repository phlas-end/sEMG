import argparse
from collections import Counter, deque
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


def main():
    parser = argparse.ArgumentParser(description="Run realtime Myo EMG inference with the original PyTorch model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
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
    print(f"using checkpoint: {checkpoint_path}")
    print(f"using device: {device}")

    model = load_model(cfg, checkpoint_path, device)

    collector = MyoEMGCollector(cfg)
    print("waiting for Myo connection...")
    if not collector.wait_for_connection(timeout_seconds=cfg["myo"].get("connect_timeout", 15.0)):
        raise RuntimeError("Myo connection timeout")

    vote_queue = deque(maxlen=args.vote_size)
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
                pred = "rest"
                confidence = 1.0
                probs = np.zeros(cfg["experiment"]["num_classes"], dtype=np.float32)
            else:
                pred, confidence, probs = predict_window(model, device, scale_signal(sample, cfg))
            vote_queue.append(pred)
            stable_pred = Counter(vote_queue).most_common(1)[0][0]
            window_index += 1

            probs_str = " ".join(f"{p:.3f}" for p in probs)
            print(
                f"window={window_index}, pred={pred}, stable_pred={stable_pred}, "
                f"conf={confidence:.3f}, probs=[{probs_str}], "
                f"abs_mean={metrics['abs_mean']:.3f}, rms={metrics['rms']:.3f}, "
                f"active_channels={metrics['active_channels']}"
            )

            if window_index >= args.max_windows:
                break


if __name__ == "__main__":
    main()
