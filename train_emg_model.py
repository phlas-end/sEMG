import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from emg_pipeline import load_gesture_dataset
from model import EMG2DCNN


class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def build_output_dirs(cfg):
    experiment = cfg["experiment"]["name"]
    note = cfg["experiment"].get("note") or "no_note"
    safe_note = note.replace(" ", "_").replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_dir = os.path.join(cfg["experiment"]["log_dir"], f"{experiment}_{safe_note}_{timestamp}")
    tb_dir = os.path.join(output_dir, "tensorboard")
    model_dir = os.path.join(output_dir, "checkpoints")

    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return output_dir, tb_dir, model_dir


def export_test_split(X_test, y_test, cfg):
    export_dir = "test_data"
    os.makedirs(export_dir, exist_ok=True)

    exp_name = cfg["experiment"]["name"]
    np.save(os.path.join(export_dir, f"{exp_name}_X.npy"), X_test.astype(np.float32))
    np.save(os.path.join(export_dir, f"{exp_name}_y.npy"), y_test.astype(np.int64))


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = cfg["data"]["channel"]
    window = cfg["data"]["window"]
    num_classes = cfg["experiment"].get("num_classes", 5)

    output_dir, tb_dir, model_dir = build_output_dirs(cfg)
    writer = SummaryWriter(log_dir=tb_dir)

    print("Loading gesture dataset")
    X_all, y_all = load_gesture_dataset(
        folder=cfg["data"]["root_dir"],
        channels=channels,
        window=window,
        dtype=np.float32,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=cfg["train"].get("test_size", 0.2),
        stratify=y_all,
        random_state=cfg["train"].get("random_seed", 42),
    )

    export_test_split(X_test, y_test, cfg)

    train_ds = EMGDataset(X_train, y_train)
    test_ds = EMGDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
    )

    model = EMG2DCNN(
        input_shape=(1, window, channels),
        model_cfg=cfg["model"],
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    config_str = yaml.dump(cfg, allow_unicode=True, sort_keys=False)
    writer.add_text("Experiment/Config", f"```yaml\n{config_str}\n```")
    if cfg["experiment"].get("note"):
        writer.add_text("Experiment/Note", cfg["experiment"]["note"])

    with open(os.path.join(model_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(config_str)

    best_acc = -1.0
    epochs = cfg["train"]["epochs"]

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, test_loader, criterion, device)

        print(
            f"[{cfg['experiment']['name']}] "
            f"Epoch {epoch:03d} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc * 100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.1f}%"
        )

        writer.add_scalar("Loss/Train", tr_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", tr_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "best.pt"))

        if epoch % 5 == 0 or epoch == epochs:
            torch.save(model.state_dict(), os.path.join(model_dir, f"epoch_{epoch:03d}.pt"))

    writer.close()
    print(f"Training finished. Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        main(yaml.safe_load(f))
