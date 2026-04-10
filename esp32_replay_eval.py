import argparse
import os
import socket
from datetime import datetime

import numpy as np
import yaml

from emg_pipeline import ensure_window_channel_layout, load_gesture_dataset, save_experiment_csv, scale_signal


def send_sample(server_ip, server_port, sample):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((server_ip, server_port))
        sock.sendall(sample.astype(np.float32, copy=False).tobytes())
        recv_bytes = sock.recv(4)
    return int.from_bytes(recv_bytes, byteorder="little", signed=True)


def main():
    parser = argparse.ArgumentParser(description="Replay test samples to ESP32 and export experiment results.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--server-ip", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--input-x", default=None)
    parser.add_argument("--input-y", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    channels = cfg["data"]["channel"]
    window = cfg["data"]["window"]
    experiment = cfg["experiment"]["name"]

    x_path = args.input_x or os.path.join("test_data", f"{experiment}_X.npy")
    y_path = args.input_y or os.path.join("test_data", f"{experiment}_y.npy")

    if os.path.exists(x_path) and os.path.exists(y_path):
        X = np.load(x_path)
        y = np.load(y_path).astype(np.int64)
        X = ensure_window_channel_layout(X, channels=channels, window=window).astype(np.float32, copy=False)
    else:
        X, y = load_gesture_dataset(cfg["data"]["root_dir"], channels=channels, window=window, dtype=np.float32)
        X = scale_signal(X, cfg)

    server_ip = args.server_ip or cfg["deploy"]["server_ip"]
    server_port = args.server_port or cfg["deploy"]["server_port"]

    rows = []
    correct = 0

    for idx, sample in enumerate(X):
        pred = send_sample(server_ip, server_port, sample)
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
