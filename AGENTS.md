# AGENTS.md

## 项目定位

这是一个 Myo 手环 8 通道 sEMG 手势识别项目，目标是训练 5 类手势模型，并部署到 ESP32-S3 上进行 ESPDL int8 实时推理。

当前不要把它当成通用 EMG 框架大改；优先保持现有链路稳定：Myo 采集 -> PyTorch 训练 -> ESPDL int8 导出 -> ESP32 回放验证 -> Myo 实时识别。

## 当前最佳基线

- 轻量模型 checkpoint：`runs/E2_light_final_pool_48x1_20260409-222302/checkpoints/best.pt`
- 当前板端模型：`C++/main/models/s3/sEMG.espdl`
- 最佳 int8 归档：`runs/candidates/best_int8_20260410_011824/`
- PyTorch 验证集：`91.58% (272/297)`
- ESP32 int8 回放：`79.46% (236/297)`
- 回放结果 CSV：`runs/experiments/E2_light_replay_20260410-011824.csv`

`runs/`、`data/`、`test_data/` 是本机实验产物，默认不进 git。需要复现实验时先确认这些目录在本机存在。

## 环境约定

常规 Python 环境：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe
```

ESPDL 导出环境：

```powershell
C:\Users\phlas\miniconda3\envs\espdl-int8\python.exe
```

ESP-IDF：

```text
C:\Users\phlas\esp\v5.5\esp-idf
```

ESP32 当前常用端口与网络配置在 `config.yaml`：

- `deploy.server_ip: 192.168.3.9`
- `deploy.server_port: 3333`

## 主要命令

训练：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe train_emg_model.py
```

导出 ESPDL：

```powershell
C:\Users\phlas\miniconda3\envs\espdl-int8\python.exe export_model_to_esp.py --checkpoint runs/E2_light_final_pool_48x1_20260409-222302/checkpoints/best.pt
```

ESP32 回放验证：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe esp32_replay_eval.py
```

Myo 实时板端识别：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe myo_realtime_infer.py --max-windows 1000000
```

本地 PyTorch 实时对照：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe python_realtime_infer.py --device cuda --max-windows 1000000
```

## 代码改动注意事项

- 尽量不要改 `myo_support/` 里的 Myo 依赖文件，除非明确是在修 Myo 运行时问题。
- 训练、导出、回放、实时推理必须保持同一输入规格：`(N, 200, 8)`，模型输入为 `(N, 1, 200, 8)`。
- 当前使用 `data.input_scale: 128.0`，不要只在某一条链路单独改缩放。
- `deploy.quantization.samples_per_class` 为空表示使用可用全量采集样本做校准；这比严格 train-only 校准效果更好。
- 静息态 `rest` 是实时推理前的能量门控，不是第 6 类分类输出。
- `pt2onnx.py`、`onnx2ncnn.txt`、`launch_tensorboard.bat` 是历史遗留，不是当前推荐主流程。
- ESP-IDF 生成的 `C++/build/`、`C++/managed_components/`、`sdkconfig` 和 ONNX/JSON/INFO 中间文件不要提交。

## 汇报材料优先级

报告展示建议优先引用：

- `README.md` 的“当前阶段报告”和“报告展示材料索引”
- `config.yaml` 的关键参数
- `model.py` 的 `final_pool` 轻量化结构
- `runs/experiments/E2_light_replay_20260410-011824.csv`
- `C++/main/models/s3/sEMG.espdl`

如果继续迭代模型，先用固定测试集和 ESP32 回放验证，不要急着做手环实时验证。
