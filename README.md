# emg-esp32

## 项目简介

这个项目用于完成 5 类手势的表面肌电识别，覆盖了从 Myo 手环采集、离线训练、模型导出，到 ESP32 实时推理和实验评估的完整流程。

当前方案的核心特点：

- 使用 Myo 手环采集 8 通道 EMG
- 训练输入统一为 `200 x 8` 的时间窗口
- 默认采样率为 `200 Hz`
- 支持离线训练与在线实时识别
- 支持把模型部署到 ESP32 做实时推理

## 当前工作流

完整流程如下：

1. 使用 Myo 手环采集手势数据
2. 将采集得到的动作段切分成训练窗口
3. 使用训练数据离线训练模型
4. 导出模型到 ESP32
5. 使用保存好的测试集回放评估 ESP32
6. 使用 Myo 实时驱动 ESP32 做在线识别

## 主要入口文件

- `myo_guided_collect.py`
  作用：按实验协议进行 Myo 引导采集
- `train_emg_model.py`
  作用：读取 `data/gesture_*.npy` 并训练模型
- `export_model_to_esp.py`
  作用：把训练好的模型导出为 ESP32 部署格式
- `esp32_replay_eval.py`
  作用：把保存好的测试窗口发送给 ESP32，统计精度
- `myo_realtime_infer.py`
  作用：直接从 Myo 采实时 EMG，实时发送给 ESP32 推理

## 关键支持文件

- `myo_runtime.py`
  作用：Myo 设备输入层，负责连接 Myo、接收 EMG 帧
- `emg_pipeline.py`
  作用：窗口切分、滤波、数据布局转换、实验结果保存
- `model.py`
  作用：PyTorch 模型定义
- `config.yaml`
  作用：全局配置
- `C++/main/app_main.cpp`
  作用：ESP32 侧接收输入并执行推理

## Myo 依赖

项目已经把 Myo 运行依赖内置到仓库里，不再依赖外部 `D:/Project/MYO/myo-tools` 目录。

内置位置：

- `myo_support/myo`
- `myo_support/bin/myo64.dll`
- `myo_support/bin/myo32.dll`

因此当前项目默认会优先使用项目内的 Myo 运行时。

## Python 环境

当前默认使用这个 conda 环境：

- `C:\Users\phlas\miniconda3\envs\myo\python.exe`

推荐执行方式：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe myo_guided_collect.py
```

## 配置说明

主要配置在 `config.yaml`。

### 数据参数

- `data.fs = 200`
  含义：采样率 200Hz
- `data.window = 200`
  含义：每个样本窗口长度为 200 帧
- `data.step = 50`
  含义：滑窗步长 50 帧
- `data.channel = 8`
  含义：Myo 的 8 通道 EMG

### 采集参数

- `collect.guided.gestures = [1, 2, 3, 4, 5]`
- `collect.guided.repetitions = 50`
- `collect.guided.action_seconds = 1.0`
- `collect.guided.rest_seconds = 1.0`

这表示默认实验协议是：

- 共 5 个手势
- 每个手势采 50 次
- 每次动作保持 1 秒
- 每次动作之间休息 1 秒

### Myo 参数

- `myo.module_root = ./myo_support`
- `myo.dll_root = ./myo_support/bin`
- `myo.connect_timeout = 15.0`

### 部署参数

- `deploy.server_ip`
- `deploy.server_port`

这两个参数用于 PC 和 ESP32 通信。

## 数据格式

### 原始动作段

采集时先记录一整段连续动作数据，形状为：

- `(frames, 8)`

### 训练窗口

切分后统一为：

- `(N, 200, 8)`

其中：

- `N` 是窗口个数
- `200` 是时间长度
- `8` 是通道数

模型输入时会进一步变成：

- `(N, 1, 200, 8)`

## 采集流程

使用：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe myo_guided_collect.py
```

采集脚本会做这些事：

1. 等待 Myo 连接
2. 按手势顺序提示你开始实验
3. 每次重复先休息，再准备，再执行动作
4. 记录休息段和动作段原始数据
5. 动作段结束后再切分成窗口
6. 保存到 `data/gesture_X.npy`

输出内容包括：

- `data/gesture_1.npy` 到 `data/gesture_5.npy`
- 每次 session 的原始段备份
- `protocol.csv`
- `session.json`

说明：

- 当前设置下，`1 秒动作` 配合 `window=200`，通常每次重复只产生 1 个完整窗口
- 如果以后想每次重复生成更多窗口，可以把动作时长增加到 `1.5` 到 `2.0` 秒

## 训练流程

使用：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe train_emg_model.py
```

训练脚本会：

1. 从 `data/gesture_*.npy` 读取样本
2. 划分训练集和测试集
3. 训练 CNN 模型
4. 保存模型到 `runs/.../checkpoints`
5. 导出一份测试集到 `test_data/`

典型输出：

- `runs/<experiment>/checkpoints/best.pt`
- `runs/<experiment>/checkpoints/epoch_XXX.pt`
- `test_data/<experiment>_X.npy`
- `test_data/<experiment>_y.npy`

## 模型导出流程

使用：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe export_model_to_esp.py
```

这个脚本用于把训练结果转换为 ESP32 可部署模型。

## ESP32 回放评估流程

使用：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe esp32_replay_eval.py
```

这个脚本的输入不是实时 Myo，而是已经保存好的测试窗口。

它会：

1. 读取 `test_data/*.npy`
2. 把每个窗口发给 ESP32
3. 接收 ESP32 返回的类别
4. 统计正确率
5. 保存实验 CSV

适合用于：

- 板端精度验证
- 量化前后对比
- 固定测试集复现实验

## 在线实时识别流程

使用：

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe myo_realtime_infer.py
```

这个脚本会：

1. 直接连接 Myo
2. 连续采 EMG
3. 切成窗口
4. 发给 ESP32
5. 接收预测结果
6. 做简单投票平滑
7. 保存实验日志

它和 `esp32_replay_eval.py` 的区别是：

- `esp32_replay_eval.py` 用保存好的测试集做离线回放评估
- `myo_realtime_infer.py` 用 Myo 实时数据做在线识别

## 文件命名说明

当前入口脚本已经统一成更直白的风格：

- `myo_guided_collect.py`
- `train_emg_model.py`
- `export_model_to_esp.py`
- `esp32_replay_eval.py`
- `myo_realtime_infer.py`

这样从名字上就能直接看出用途。

## 常用命令

```powershell
C:\Users\phlas\miniconda3\envs\myo\python.exe myo_guided_collect.py
C:\Users\phlas\miniconda3\envs\myo\python.exe train_emg_model.py
C:\Users\phlas\miniconda3\envs\myo\python.exe export_model_to_esp.py
C:\Users\phlas\miniconda3\envs\myo\python.exe esp32_replay_eval.py
C:\Users\phlas\miniconda3\envs\myo\python.exe myo_realtime_infer.py
```

## 目前建议

- 先按默认协议完成 5 个手势的数据采集
- 先用 `esp32_replay_eval.py` 做稳定精度验证
- 板端精度稳定后，再重点看 `myo_realtime_infer.py` 的实时体验

## 备注

如果以后要扩展手势数、修改动作时长、调整窗口大小，优先改 `config.yaml`，不要直接改各个脚本里的常量。
