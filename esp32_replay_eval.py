import argparse
import os
from datetime import datetime

import numpy as np
import yaml

from emg_pipeline import ensure_window_channel_layout, save_experiment_csv
from esp32_client import send_sample


def find_latest_run_test_split(log_dir, experiment_name):
    prefix = f"{experiment_name}_"
    candidates = []
    if not os.path.isdir(log_dir):
        return None, None

    for name in os.listdir(log_dir):
        run_dir = os.path.join(log_dir, name)
        if not os.path.isdir(run_dir) or not name.startswith(prefix):
            continue
        x_path = os.path.join(run_dir, "test_split", "X.npy")
        y_path = os.path.join(run_dir, "test_split", "y.npy")
        if os.path.exists(x_path) and os.path.exists(y_path):
            candidates.append((os.path.getmtime(x_path), x_path, y_path))

    if not candidates:
        return None, None

    _, x_path, y_path = max(candidates, key=lambda item: item[0])
    return x_path, y_path


def main():
    parser = argparse.ArgumentParser(description="Replay test samples to ESP32 and export experiment results.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--server-ip", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--input-x", default=None)
    parser.add_argument("--input-y", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    if bool(args.input_x) != bool(args.input_y):
        parser.error("--input-x and --input-y must be supplied together")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    channels = cfg["data"]["channel"]
    window = cfg["data"]["window"]
    experiment = cfg["experiment"]["name"]

    if args.input_x and args.input_y:
        x_path, y_path = args.input_x, args.input_y
    elif args.run_dir:
        x_path = os.path.join(args.run_dir, "test_split", "X.npy")
        y_path = os.path.join(args.run_dir, "test_split", "y.npy")
    else:
        x_path, y_path = find_latest_run_test_split(cfg["experiment"]["log_dir"], experiment)
        if not x_path or not y_path:
            x_path = os.path.join("test_data", f"{experiment}_X.npy")
            y_path = os.path.join("test_data", f"{experiment}_y.npy")

    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"using replay split: {x_path}")
        X = np.load(x_path)
        y = np.load(y_path).astype(np.int64)
        X = ensure_window_channel_layout(X, channels=channels, window=window).astype(np.float32, copy=False)
    else:
        raise FileNotFoundError("Saved validation split not found; use --run-dir or --input-x/--input-y")

    classes = int(cfg['experiment'].get('num_classes', 5))
    if X.ndim != 3 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError("Replay requires a nonempty, paired X/y split")
    if not np.isfinite(X).all() or np.any((y < 0) | (y >= classes)):
        raise ValueError("Replay data contains invalid inputs or labels")

    server_ip = args.server_ip or cfg["deploy"]["server_ip"]
    server_port = args.server_port or cfg["deploy"]["server_port"]

    rows = []
    correct = 0

    for idx, sample in enumerate(X):
        pred = send_sample(server_ip, server_port, sample)
        if not 0 <= pred < classes:
            raise ValueError(f"Board returned class {pred}, expected 0..{classes - 1}")
        truth = int(y[idx])
        is_correct = int(pred == truth)
        correct += is_correct

        rows.append(
            {
                "sample_index": idx,
                "true_label": truth,
                "pred_label": pred,
                "correct": is_correct,
            }
        )

        if (idx + 1) % 20 == 0 or idx == len(X) - 1:
            print(f"processed {idx + 1}/{len(X)}, accuracy={correct / (idx + 1):.4f}")

    accuracy = correct / max(len(X), 1)
    print(f"final accuracy: {accuracy:.4f} ({correct}/{len(X)})")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join("runs", "experiments", f"{experiment}_replay_{timestamp}.csv")
    save_experiment_csv(csv_path, rows)
    print(f"saved experiment results to {csv_path}")


if __name__ == "__main__":
    main()
