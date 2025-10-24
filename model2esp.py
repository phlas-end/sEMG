import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import EMG2DCNN
from esp_ppq.api import espdl_quantize_torch
import os
import yaml

DEVICE = "cpu"

# -------- Dataset 定义 --------
class FeatureOnlyDataset(Dataset):
    def __init__(self, np_array):
        self.features = torch.tensor(np_array, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def collate_fn2(batch):
    return torch.stack(batch)  # 不要在这里 .to(DEVICE)，DataLoader 内部会卡

# -------- 主流程 --------
if __name__ == '__main__':
    BATCH_SIZE = 32
    INPUT_SHAPE = [1, 200, 16]   # ⚠️ 和你的模型输入保持一致
    TARGET = "esp32s3"
    NUM_OF_BITS = 8
    ESPDL_MODEL_PATH = "./s3/class_3/touch_recognition_from_pt.espdl"

    os.makedirs(os.path.dirname(ESPDL_MODEL_PATH), exist_ok=True)

    # 1. 加载测试数据（取前8个做校准）
    test_dataset = np.load("./test_data/E2_X.npy")[:8]
    feature_only_test_data = FeatureOnlyDataset(test_dataset)
    testDataLoader = DataLoader(
        dataset=feature_only_test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn2
    )

    # 2. 加载 PyTorch 模型 (假设是 state_dict 格式保存的)
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = EMG2DCNN(
        input_shape = INPUT_SHAPE,
        model_cfg=cfg["model"],
        num_classes=5
    )
    model.load_state_dict(torch.load(r"D:\Project\sEMG\runs\E2_no_note_20251009-001206\checkpoints\epoch_050.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. 量化并导出 ESPDL
    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=testDataLoader,
        calib_steps=8,
        input_shape=[1] + INPUT_SHAPE,   # [1, 1, 200, 16]
        inputs=[torch.tensor(test_dataset[0], dtype=torch.float32).unsqueeze(0)],  # 加 batch 维度
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
        dispatching_override=None
    )
