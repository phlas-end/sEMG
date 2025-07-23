import os
from collections import defaultdict
import numpy as np
import scipy.io
from scipy.signal import butter, iirnotch, filtfilt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from model import EMG2DCNN
from datetime import datetime

# -----------------------------
# 滤波器
# -----------------------------
def notch_filter(signal, fs=200, freq=50, Q=30):
    b, a = iirnotch(freq/(fs/2), Q)
    return filtfilt(b, a, signal, axis=0)

def bandpass_filter(signal, fs=200, low=20, high=100, order=4):
    nyq = fs / 2
    if high >= nyq:
        high = nyq * 0.99
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)

# -----------------------------
# 数据加载
# -----------------------------
def load_subject_data(subject_dir, experiment, fs, filters):
    emg_list = []
    label_list = []
    repetition_list = []

    if experiment == "ALL":
        mat_files = sorted([f for f in os.listdir(subject_dir) if f.endswith('.mat')])
    else:
        mat_files = sorted([f for f in os.listdir(subject_dir) if f.endswith('.mat') and experiment in f])

    print(f"[{subject_dir}] Found {len(mat_files)} files for {experiment}")

    for f in mat_files:
        path = os.path.join(subject_dir, f)
        data = scipy.io.loadmat(path)
        emg = data['emg']
        stimulus = data['restimulus'].flatten()
        repetition = data['rerepetition'].flatten()

        # 新增过滤，去掉标签为0的数据
        mask = stimulus > 0
        emg = emg[mask]
        stimulus = stimulus[mask]
        repetition = repetition[mask]

        # 🚀 给标签做偏移，防止重叠
        if experiment != "ALL":
            offset = 0
        elif "E1" in f:
            offset = 0
        elif "E2" in f:
            offset = 12
        elif "E3" in f:
            offset = 12+17
        else:
            offset = 0
        stimulus = stimulus + offset

        # 滤波
        notch_freq = filters["notch"]["freq"]
        notch_Q = filters["notch"]["Q"]
        bandpass_low = filters["bandpass"]["low"]
        bandpass_high = filters["bandpass"]["high"]
        bandpass_order = filters["bandpass"]["order"]

        emg = notch_filter(emg, fs=fs, freq=notch_freq, Q=notch_Q)
        emg = bandpass_filter(emg, fs=fs, low=bandpass_low, high=bandpass_high, order=bandpass_order)
        # emg = (emg - np.mean(emg, axis=0)) / (np.std(emg, axis=0) + 1e-6)

        emg_list.append(emg)
        label_list.append(stimulus)
        repetition_list.append(repetition)

    all_emg = np.vstack(emg_list)
    all_labels = np.concatenate(label_list)
    all_repetitions = np.concatenate(repetition_list)
    return all_emg, all_labels, all_repetitions


# -----------------------------
# 样本提取
# -----------------------------
def extract_samples(emg, labels, repetitions, window=200, step=20):
    samples = defaultdict(list)
    current_label = labels[0]
    current_repetition = repetitions[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label or repetitions[i] != current_repetition:
            segment_emg = emg[start_idx:i]
            segment_rep = repetitions[start_idx:i]
            if len(segment_emg) >= window:
                for j in range(0, len(segment_emg)-window+1, step):
                    clip = segment_emg[j:j+window]
                    clip_rep = segment_rep[j + window//2]
                    samples[current_label].append((clip, clip_rep))
            current_label = labels[i]
            current_repetition = repetitions[i]
            start_idx = i
#数据最后一段
    segment_emg = emg[start_idx:]
    segment_rep = repetitions[start_idx:]
    if len(segment_emg) >= window:
        for j in range(0, len(segment_emg)-window+1, step):
            clip = segment_emg[j:j+window]
            clip_rep = segment_rep[j + window//2]
            samples[current_label].append((clip, clip_rep))

    return samples

# -----------------------------
# Dataset
# -----------------------------
class EMGDataset(Dataset):
    def __init__(self, X, y):
        tensor_X = torch.tensor(X, dtype=torch.float32)
        self.X = tensor_X.unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# 训练和验证
# -----------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total

# -----------------------------
# 主函数
# -----------------------------
def main(cfg):
    root_dir = cfg["data"]["root_dir"]
    fs = cfg["data"]["fs"]
    window = cfg["data"]["window"]
    step = cfg["data"]["step"]
    experiment = cfg["experiment"]["name"]
    note = cfg["experiment"]["note"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_note = note.replace(" ", "_").replace("/", "_") if note else "no_note"

    # 创建输出目录
    output_dir = os.path.join(cfg["experiment"]["log_dir"], f"{experiment}_{safe_note}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # TensorBoard目录
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    # 模型保存目录
    model_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)

    all_emg = []
    all_labels = []
    all_repetitions = []
    subjects = sorted([d for d in os.listdir(root_dir) if d.startswith('s') and os.path.isdir(os.path.join(root_dir, d))])

    print(f"Selected experiment: {experiment}")
    print(f"Found subjects: {subjects}")

    for subj in subjects:
        emg, labels, repetitions = load_subject_data(
            os.path.join(root_dir, subj),
            experiment=experiment,
            fs=fs,
            filters=cfg["filters"],
        )
        all_emg.append(emg)
        all_labels.append(labels)
        all_repetitions.append(repetitions)

    all_emg = np.vstack(all_emg)
    all_labels = np.concatenate(all_labels)
    all_repetitions = np.concatenate(all_repetitions)


    unique_labels = np.unique(all_labels)
    print(f"Detected gesture classes (excluding 0): {unique_labels}")

    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in all_labels])

    from collections import Counter

    label_counts = Counter(mapped_labels)
    print("mapped_labels 中各类别样本数量：")
    for label, count in label_counts.items():
        print(f"类别 {label}: {count} 个样本")

    samples = extract_samples(
        all_emg,
        mapped_labels,
        all_repetitions,
        window=window,
        step=step
    )

    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}")

    X_train, y_train, X_test, y_test = [], [], [], []
    for c in range(num_classes):
        c_samples = samples[c]
        train_samples = [clip for clip, rep in c_samples if rep in [1,2,3,4]]
        test_samples = [clip for clip, rep in c_samples if rep in [5,6]]

        if len(train_samples) == 0 or len(test_samples) == 0:
            print(f"⚠️ Class {c} skipped due to insufficient samples")
            continue

        X_train += train_samples
        y_train += [c] * len(train_samples)
        X_test += test_samples
        y_test += [c] * len(test_samples)

    if len(X_train) == 0 or len(X_test) == 0:
        print("数据不足，无法训练")
        return

    train_ds = EMGDataset(X_train, y_train)
    test_ds = EMGDataset(X_test, y_test)
    # 保存测试数据供C++端验证
    X_test_tensor = test_ds.X.cpu().numpy()
    y_test_tensor = test_ds.y.cpu().numpy()

    export_dir = r"D:\Project\sEMG\test_data"
    exp_name = cfg["experiment"]["name"]
    os.makedirs(export_dir, exist_ok=True)

    np.save(os.path.join(export_dir, f"{exp_name}_X.npy"), X_test_tensor)
    np.save(os.path.join(export_dir, f"{exp_name}_y.npy"), y_test_tensor)

    print(f"✅ 测试数据已分别导出: {export_dir}")

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], num_workers=cfg["train"]["num_workers"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], num_workers=cfg["train"]["num_workers"])

    model = EMG2DCNN(
        input_shape=(1, window, cfg["data"]["channel"]),
        model_cfg=cfg["model"],
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])

    # 写入配置
    config_str = yaml.dump(cfg, allow_unicode=True, sort_keys=False)
    writer.add_text("Experiment/Config", f"```yaml\n{config_str}\n```")
    if note:
        writer.add_text("Experiment/Note", note)

    # 把配置保存到模型目录
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_str)

    epochs = cfg["train"]["epochs"]
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, test_loader, criterion, device)
        print(f"[{experiment}] Epoch {epoch:03d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")
        writer.add_scalar("Loss/Train", tr_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", tr_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        # 保存每个epoch模型
        if epoch % 5 == 0 or epoch == epochs:
            model_path = os.path.join(model_dir, f"epoch_{epoch:03d}.pt")
            torch.save(model.state_dict(), model_path)

    writer.close()

def generate_qt_ncnn():
    import numpy as np

    # 加载原始数据
    data = np.load('test_data/E1_X.npy')  # 形状 [N, C, H, W]

    # 创建存储目录
    import os

    os.makedirs('filelist_in0', exist_ok=True)

    # 逐样本保存为单独的.npy文件
    for i in range(data.shape[0]):
        sample = data[i]  # 获取第i个样本 [C, H, W]
        np.save(f'filelist_in0/{i}_in0.npy', sample)  # 保存为 0.npy, 1.npy, ...

    print(f"已拆分 {data.shape[0]} 个样本到 calib_data/ 目录")
    with open('filelist_in0.txt', 'w') as f:
        for i in range(data.shape[0]):
            f.write(f'filelist_in0/{i}_in0.npy\n')

    print("已生成 calibration.txt 文件")

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    main(cfg)

