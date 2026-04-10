import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from emg_pipeline import load_gesture_dataset, scale_signal
from esp_ppq.api import espdl_quantize_torch
from model import EMG2DCNN

DEVICE = "cpu"


class FeatureOnlyDataset(Dataset):
    def __init__(self, np_array):
        # Quantization calibration should match the training input shape: (N, 1, 200, 8)
        self.features = torch.tensor(np_array, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


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


def build_calibration_dataset(cfg):
    X_all, y_all = load_gesture_dataset(
        cfg["data"]["root_dir"],
        channels=cfg["data"]["channel"],
        window=cfg["data"]["window"],
        dtype=np.float32,
    )

    quant_cfg = cfg.get("deploy", {}).get("quantization", {})
    samples_per_class = quant_cfg.get("samples_per_class")
    if not samples_per_class:
        return scale_signal(X_all, cfg)

    picked = []
    for cls in sorted(np.unique(y_all)):
        cls_samples = X_all[y_all == cls]
        picked.append(cls_samples[: min(len(cls_samples), samples_per_class)])
    return scale_signal(np.concatenate(picked, axis=0), cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a PyTorch checkpoint to ESPDL.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="./C++/main/models/s3/sEMG.espdl")
    args = parser.parse_args()

    BATCH_SIZE = 32
    INPUT_SHAPE = [1, 200, 8]
    TARGET = "esp32s3"
    NUM_OF_BITS = 8
    ESPDL_MODEL_PATH = args.output

    os.makedirs(os.path.dirname(ESPDL_MODEL_PATH), exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_best_pt(cfg["experiment"]["log_dir"])
    calibration_data = build_calibration_dataset(cfg)
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Export output: {ESPDL_MODEL_PATH}")
    print(f"Calibration source: {cfg['data']['root_dir']}")

    # 1. Load test data for calibration
    test_dataset = calibration_data.astype(np.float32, copy=False)
    print(f"Calibration samples used: {len(test_dataset)}")
    feature_only_test_data = FeatureOnlyDataset(test_dataset)
    testDataLoader = DataLoader(
        dataset=feature_only_test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 2. Load PyTorch model
    model = EMG2DCNN(
        input_shape=INPUT_SHAPE,
        model_cfg=cfg["model"],
        num_classes=cfg["experiment"].get("num_classes", 5),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. Quantize and export ESPDL
    espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=testDataLoader,
        calib_steps=len(testDataLoader),
        input_shape=[1] + INPUT_SHAPE,
        inputs=[torch.tensor(test_dataset[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)],
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
        dispatching_override=None,
    )
