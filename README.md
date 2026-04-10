# emg-esp32

## 项目简介

本项目用于完成基于 Myo 手环 8 通道 sEMG 信号的 5 类手势识别，并把训练好的模型部署到 ESP32-S3 上进行 int8 推理。

当前主线流程是：Myo 采集数据 -> 离线训练 PyTorch 模型 -> ESPDL int8 量化导出 -> ESP32-S3 回放验证 -> Myo 实时识别。

## 当前阶段报告

当前已形成一版可工作的轻量化基线，后续实时实验建议先基于这一版继续做。

| 项目 | 结果 |
| --- | --- |
| 任务 | 5 类手势识别 |
| 采集设备 | Myo 手环 |
| 输入通道 | 8 通道 EMG |
| 窗口尺寸 | `200 x 8` |
| 采样率 | `200 Hz` |
| 训练样本数 | `1482` 个窗口 |
| PyTorch 验证集准确率 | `91.58%`，`272/297` |
| ESP32 int8 回放准确率 | `79.46%`，`236/297` |
| 板端模型大小 | 约 `3.19 MB` |
| 当前板端模型 | `C++/main/models/s3/sEMG.espdl` |

当前最佳 PyTorch checkpoint 在本机实验目录中：

```text
runs/E2_light_final_pool_48x1_20260409-222302/checkpoints/best.pt
```

当前最佳 int8 归档在本机实验目录中：

```text
runs/candidates/best_int8_20260410_011824
```

说明：`runs/`、`data/`、`test_data/` 默认不进 git，避免把训练数据和实验产物全部上传。真正会跟随仓库的板端模型是 `C++/main/models/s3/sEMG.espdl`。

## 当前方案说明

这版不是最早的大模型，而是后续迭代得到的轻量化版本。

关键点如下：

- `model.final_pool: [48, 1]` 用于压缩卷积后的特征图，避免全连接层过大。
- `model.fc_hidden: 1024` 保持和旧版接近的容量。
- `data.input_scale: 128.0` 统一训练、导出、回放、实时推理的输入尺度。
- `deploy.quantization.samples_per_class` 为空，表示导出时使用 `data/gesture_*.npy` 中可用的全量样本做量化校准。
- 实时识别阶段增加了静息态门控，用于在放松状态下输出 `rest`，避免把低能量静息信号强行判成某个手势。

这轮尝试过更小的 `final_pool` 和更少 `fc_hidden`，但 ESP32 int8 回放没有超过当前版本。因此当前先固定为基线。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `myo_guided_collect.py` | Myo 引导式采集入口 |
| `myo_runtime.py` | Myo 运行时封装，负责连接手环并接收 EMG |
| `emg_pipeline.py` | 数据布局、切窗、输入缩放、静息态判断、实验 CSV 保存 |
| `train_emg_model.py` | 离线训练入口 |
| `model.py` | PyTorch 模型结构 |
| `export_model_to_esp.py` | PyTorch checkpoint 到 ESPDL int8 的导出入口 |
| `esp32_replay_eval.py` | 使用固定测试集回放到 ESP32 并统计准确率 |
| `myo_realtime_infer.py` | Myo 实时输入，发送到 ESP32 做 int8 实时识别 |
| `python_realtime_infer.py` | Myo 实时输入，本地 PyTorch 原始模型识别，用于和 ESP32 对照 |
| `config.yaml` | 项目主配置 |
| `C++/main/app_main.cpp` | ESP32-S3 端推理服务 |
| `C++/main/models/s3/sEMG.espdl` | 当前板端 int8 模型 |

## 环境说明

常规采集、训练、回放、实时识别使用：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe
```

ESPDL int8 导出使用：

```powershell
C:\Users\phlas\miniconda3\envs\espdl-int8\python.exe
```

ESP-IDF 环境在：

```text
C:\Users\phlas\esp\v5.5\esp-idf
```

## 数据采集流程

默认采集协议在 `config.yaml` 中：

| 配置项 | 当前值 | 含义 |
| --- | --- | --- |
| `collect.guided.gestures` | `[1, 2, 3, 4, 5]` | 5 个手势 |
| `collect.guided.repetitions` | `50` | 每个手势 50 次 |
| `collect.guided.action_seconds` | `1.0` | 每次动作保持 1 秒 |
| `collect.guided.rest_seconds` | `1.0` | 每次动作间休息 1 秒 |

运行采集：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe myo_guided_collect.py
```

采集脚本会先记录动作段，再按 `window=200`、`step=50` 做事后切分，最终保存为：

```text
data/gesture_1.npy
data/gesture_2.npy
data/gesture_3.npy
data/gesture_4.npy
data/gesture_5.npy
```

## 训练流程

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe train_emg_model.py
```

训练脚本会读取 `data/gesture_*.npy`，统一输入尺度后训练模型，并导出对应测试集到 `test_data/`。

典型输出：

```text
runs/<experiment>/checkpoints/best.pt
test_data/<experiment>_X.npy
test_data/<experiment>_y.npy
```

## ESPDL int8 导出流程

```powershell
C:\Users\phlas\miniconda3\envs\espdl-int8\python.exe export_model_to_esp.py --checkpoint runs/E2_light_final_pool_48x1_20260409-222302/checkpoints/best.pt
```

导出结果会覆盖：

```text
C++/main/models/s3/sEMG.espdl
```

如果不传 `--checkpoint`，脚本会从 `runs/` 下自动查找最新的 `best.pt`。为了复现实验，建议关键版本显式指定 checkpoint。

## ESP32 编译与烧录流程

进入板端工程：

```powershell
cd C++
```

加载 ESP-IDF 环境后执行：

```powershell
idf.py set-target esp32s3
idf.py build
idf.py -p COM6 flash
```

板端程序会启动 TCP 服务，PC 侧通过 `config.yaml` 中的 `deploy.server_ip` 和 `deploy.server_port` 连接。

## 固定测试集回放验证

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe esp32_replay_eval.py
```

这个脚本用于把保存好的测试集窗口发给 ESP32，统计 int8 板端准确率。当前最佳回放结果为：

```text
79.46% (236/297)
```

## 实时识别流程

ESP32 int8 实时识别：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe myo_realtime_infer.py --max-windows 1000000
```

本地 PyTorch 原始模型实时识别，用于对照：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe python_realtime_infer.py --device cuda --max-windows 1000000
```

实时脚本会输出当前窗口预测、稳定投票结果，以及静息态 `rest` 判断。

## 后续建议

短期建议先不要继续扩大模型。当前主要差距在 int8 量化后精度损失，下一步如果要继续提升，优先考虑：

- 采集更多覆盖不同佩戴状态和发力强度的数据。
- 针对混淆类别补采样本，而不是盲目增大模型。
- 保持当前输入缩放和量化校准流程一致，避免训练、导出、板端回放三者输入分布漂移。
- 实时实验时先看 `python_realtime_infer.py` 和 `myo_realtime_infer.py` 是否在同一动作上同时混淆，再判断是模型问题还是量化/板端问题。

## 报告展示材料索引

以下文件适合在论文、答辩或阶段报告里引用。注意：`runs/`、`data/`、`test_data/` 默认被 `.gitignore` 忽略，属于本机实验产物；仓库里会保留源码、配置、README 和当前板端 `sEMG.espdl`。

| 用途 | 路径 | 说明 |
| --- | --- | --- |
| 项目流程说明 | `README.md` | 项目介绍、操作流程、当前结果汇总 |
| AI 接手说明 | `AGENTS.md` | 给后续 AI/协作者看的上下文和注意事项 |
| 当前配置 | `config.yaml` | 窗口、通道、轻量模型、静息态、量化校准配置 |
| 模型结构源码 | `model.py` | 当前轻量化 CNN 结构，包含 `final_pool` |
| 训练入口 | `train_emg_model.py` | 离线训练与测试集导出 |
| 导出入口 | `export_model_to_esp.py` | PyTorch -> ESPDL int8 导出 |
| 板端回放评估 | `esp32_replay_eval.py` | 固定测试集发送到 ESP32 统计准确率 |
| 板端推理代码 | `C++/main/app_main.cpp` | ESP32-S3 TCP 接收和推理逻辑 |
| 当前板端模型 | `C++/main/models/s3/sEMG.espdl` | 已刷入并验证过的 int8 模型，约 `3.19 MB` |
| 最佳 PyTorch 模型 | `runs/E2_light_final_pool_48x1_20260409-222302/checkpoints/best.pt` | 本机最佳轻量模型，PyTorch 验证集 `91.58%` |
| 最佳 int8 归档 | `runs/candidates/best_int8_20260410_011824/` | 本机归档，包含 `best.pt`、`sEMG.espdl`、配置和回放结果 |
| 最佳回放结果 | `runs/experiments/E2_light_replay_20260410-011824.csv` | ESP32 int8 固定测试集回放，`79.46% (236/297)` |
| 训练数据 | `data/gesture_*.npy` | 5 类手势采集后的窗口数据，本机保存 |
| 测试数据 | `test_data/E2_light_X.npy`、`test_data/E2_light_y.npy` | 当前实验对应测试集，本机保存 |

报告里建议优先展示这几组数字：

| 指标 | 数值 |
| --- | --- |
| 手势类别数 | `5` |
| 输入窗口 | `200 x 8` |
| 采样率 | `200 Hz` |
| 总窗口样本数 | `1482` |
| PyTorch 验证集准确率 | `91.58% (272/297)` |
| ESP32 int8 回放准确率 | `79.46% (236/297)` |
| 板端模型大小 | 约 `3.19 MB` |

历史探索文件如 `pt2onnx.py`、`onnx2ncnn.txt`、`launch_tensorboard.bat` 不属于当前推荐主流程，报告里不建议引用。

