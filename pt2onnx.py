import os.path

import torch
from model import EMG2DCNN
import yaml

model_dir = r"D:\Project\sEMG\runs\E2_no_note_20250708-225008\checkpoints"
model_epoch = "epoch_120.pt"

cfg_path = f"{model_dir}/config.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

output_dir = "output_model/model_{}".format(cfg["experiment"]["name"])
output_name = cfg["experiment"]["name"]+'.onnx'
os.makedirs(output_dir, exist_ok=True)


model_path = f"{model_dir}/{model_epoch}"

dict_class = {"E1": 12, "E2": 17, "E3":23, "ALL": 52}

device = torch.device("cpu")

model = EMG2DCNN(
    input_shape=(1, cfg["data"]["window"], 16),
    model_cfg=cfg["model"],
    num_classes=dict_class[cfg["experiment"]["name"]]  # 注意：这里和权重对应
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

dummy_input = torch.randn(1, 1, cfg["data"]["window"], 16)

torch.onnx.export(
    model,
    dummy_input,
    os.path.join(output_dir,output_name),
    input_names=["in0"],
    output_names=["out0"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("✅ 成功导出{}类模型为{}".format(dict_class[cfg["experiment"]["name"]], output_name))
