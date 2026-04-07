import argparse
import csv
import os
from collections import Counter, deque
from datetime import datetime

import numpy as np
import yaml

from emg_pipeline import apply_filters, segment_continuous_signal
from myo_runtime import MyoEMGCollector


def send_window(server_ip, server_port, window):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((server_ip, server_port))
        sock.sendall(window.astype(np.float32, copy=False).tobytes())
        recv_bytes = sock.recv(4)
    return int.from_bytes(recv_bytes, byteorder="little", signed=True)


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["window_index", "pred_label", "stable_label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run realtime Myo EMG inference and export logs.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--server-ip", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=100)
    parser.add_argument("--vote-size", type=int, default=5)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(args.config)

    collector = MyoEMGCollector(cfg)
    print("waiting for Myo connection...")
    if not collector.wait_for_connection(timeout_seconds=cfg["myo"].get("connect_timeout", 15.0)):
        raise RuntimeError("Myo connection timeout")

    server_ip = args.server_ip or cfg["deploy"]["server_ip"]
    server_port = args.server_port or cfg["deploy"]["server_port"]

    vote_queue = deque(maxlen=args.vote_size)
    rows = []
    window_index = 0

    while window_index < args.max_windows:
        chunk = collector.collect_duration(args.chunk_seconds)
        if chunk.shape[0] < cfg["data"]["window"]:
            continue

        windows = segment_continuous_signal(chunk, window=cfg["data"]["window"], step=cfg["data"]["step"])
        if windows.shape[0] == 0:
            continue

        for window in windows:
            sample = apply_filters(window, cfg) if cfg["collect"].get("apply_filters", False) else window
            pred = send_window(server_ip, server_port, sample)
            vote_queue.append(pred)
            stable_pred = Counter(vote_queue).most_common(1)[0][0]
            window_index += 1

            print(f"window={window_index}, pred={pred}, stable_pred={stable_pred}")
            rows.append(
                {
                    "window_index": window_index,
                    "pred_label": pred,
                    "stable_label": stable_pred,
                }
            )

            if window_index >= args.max_windows:
                break

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join("runs", "experiments", f"realtime_{timestamp}.csv")
    write_rows(output_path, rows)
    print(f"saved realtime results to {output_path}")


if __name__ == "__main__":
    main()
