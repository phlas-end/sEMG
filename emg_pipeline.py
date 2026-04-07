import csv
import json
import os
from collections import Counter, deque

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def notch_filter(signal, fs=200, freq=50, q=30):
    b, a = iirnotch(freq / (fs / 2), q)
    return filtfilt(b, a, signal, axis=0)


def bandpass_filter(signal, fs=200, low=20, high=100, order=4):
    nyq = fs / 2
    safe_high = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, safe_high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def apply_filters(signal, cfg):
    if signal.shape[0] < 16:
        return signal

    filtered = signal
    notch_cfg = cfg["filters"]["notch"]
    band_cfg = cfg["filters"]["bandpass"]
    filtered = notch_filter(filtered, fs=cfg["data"]["fs"], freq=notch_cfg["freq"], q=notch_cfg["Q"])
    filtered = bandpass_filter(
        filtered,
        fs=cfg["data"]["fs"],
        low=band_cfg["low"],
        high=band_cfg["high"],
        order=band_cfg["order"],
    )
    return filtered


def ensure_window_channel_layout(data, channels, window):
    if data.ndim == 2:
        if data.shape == (window, channels):
            return data
        if data.shape == (channels, window):
            return data.T
        raise ValueError(f"Expected 2D shape ({window}, {channels}) or ({channels}, {window}), got {data.shape}")

    if data.ndim == 3:
        if data.shape[1:] == (window, channels):
            return data
        if data.shape[1:] == (channels, window):
            return np.transpose(data, (0, 2, 1))
        raise ValueError(
            f"Expected 3D shape (N, {window}, {channels}) or (N, {channels}, {window}), got {data.shape}"
        )

    raise ValueError(f"Unsupported ndim: {data.ndim}")


def load_gesture_dataset(folder, channels, window, dtype=np.float32):
    X_list = []
    y_list = []

    files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if not files:
        raise ValueError(f"No npy files found in {folder}")

    for file_name in files:
        if not file_name.startswith("gesture_"):
            continue

        try:
            label = int(file_name.split("gesture_")[1].split(".")[0]) - 1
        except ValueError:
            print(f"skip invalid label file: {file_name}")
            continue

        path = os.path.join(folder, file_name)
        data = np.load(path)
        data = ensure_window_channel_layout(data, channels=channels, window=window).astype(dtype, copy=False)

        X_list.append(data)
        y_list.append(np.full((data.shape[0],), label, dtype=np.int64))
        print(f"loaded {file_name}: samples={data.shape[0]}, label={label}")

    if not X_list:
        raise ValueError(f"No valid gesture_*.npy files found in {folder}")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    print(f"dataset shape: X={X_all.shape}, y={y_all.shape}")
    print(f"class counts: {dict(sorted(Counter(y_all).items()))}")
    return X_all, y_all


class SlidingWindowCollector:
    def __init__(self, channels, window, step):
        self.channels = channels
        self.window = window
        self.step = step
        self.buffer = deque(maxlen=window)
        self.samples_since_emit = 0

    def append_frame(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (self.channels,):
            raise ValueError(f"Expected frame shape ({self.channels},), got {frame.shape}")

        self.buffer.append(frame)
        if len(self.buffer) < self.window:
            return None

        self.samples_since_emit += 1
        if self.samples_since_emit < self.step:
            return None

        self.samples_since_emit = 0
        return np.stack(self.buffer, axis=0)

    def reset(self):
        self.buffer.clear()
        self.samples_since_emit = 0


def save_gesture_samples(output_dir, gesture_id, samples):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"gesture_{gesture_id}.npy")

    data = np.asarray(samples, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected samples to have shape (N, window, channels), got {data.shape}")

    if os.path.exists(path):
        existing = np.load(path)
        existing = ensure_window_channel_layout(existing, channels=data.shape[2], window=data.shape[1])
        data = np.concatenate([existing, data], axis=0)

    np.save(path, data)
    return path, data.shape[0]


def save_experiment_csv(output_path, rows):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "true_label",
                "pred_label",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def segment_continuous_signal(signal, window, step):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Expected continuous signal shape (frames, channels), got {signal.shape}")

    if signal.shape[0] < window:
        return np.empty((0, window, signal.shape[1]), dtype=np.float32)

    windows = []
    for start in range(0, signal.shape[0] - window + 1, step):
        windows.append(signal[start : start + window])
    return np.asarray(windows, dtype=np.float32)


def save_metadata_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
