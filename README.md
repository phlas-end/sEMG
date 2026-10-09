# Myo sEMG 项目归档

Myo 方向于 2026-10-09 整理封版。保留采集、训练、模型导出和验证代码供复现；后续开发转入独立 ESP32 工程。

## 项目内容

历史流程：Myo 8 通道肌电 → 电脑采集/切窗 → PyTorch 离线训练 → ESPDL int8 导出 → ESP32-S3 推理 → 数据回放或实时对照。

最初为 5 类手势，最终为 6 类：`0=静息态，1~5=手势`。当前能量门控关闭，静息态是模型输出。没有在线训练。

当前板端仍是 TCP 窗口推理服务，电脑完成采集和切窗。直接采集、端侧切窗和本地显示属于后续工作。

## 当前配套版本

| 项目 | 六分类版本 |
| --- | --- |
| 模型目录 | `runs/R6_r6_fp24_fc1024_20260413-154153/` |
| checkpoint / 配置 | 上述目录的 `checkpoints/best.pt`、`checkpoints/config.yaml` |
| PyTorch 验证结果 | `82.18% (249/303)`，本轮重新加载原模型与验证集复核 |
| 板端 ESPDL | `C++/main/models/s3/sEMG.espdl`，1,618,144 bytes |
| 原始 ESPDL 归档 | 上述目录的 `espdl/sEMG.espdl`，与板端文件 SHA-256 相同 |
| 验证集 | 上述目录的 `test_split/X.npy`、`y.npy` |
| 结构 | 三层卷积，`final_pool=[24,1]`，`fc_hidden=1024` |
| 输入 | 200 Hz、8 通道、窗口 `200×8`；模型输入 `(N,1,200,8)` |
| 缩放 | 原始 Myo 数值除以 `128.0`；保存的验证集已经缩放 |

根 `config.yaml` 已与实际默认模型对齐。六分类板端准确率须通过对应验证集实机回放确认，不能借用五分类数字。

历史五分类基线为 PyTorch `91.58% (272/297)`、ESP32 int8 `79.46% (236/297)`；配套归档为 `runs/candidates/best_int8_20260410_011824/`。

| 六分类结构：final_pool / fc_hidden | 历史最好 PyTorch 验证结果 | 本机 ESPDL |
| --- | --- | --- |
| `[24,1]` / 1024 | 82.18% | 已保留，当前默认 |
| `[24,1]` / 768 | 81.19% | 未发现成功导出的文件 |
| `[48,1]` / 1024 | 76.24% | 未发现成功导出的文件 |
| `[48,1]` / 768 | 76.24% | 未发现成功导出的文件 |

完整本机模型清单见 `reports/model_inventory.json`；本轮复核见 `reports/verification.json`。历史训练按窗口随机划分，并使用同一验证集选择 best；默认量化校准可能包含验证窗口。这些结果不能解释为严格无泄漏的独立测试或跨受试者结果。

## 文件和本地数据

| 文件或目录 | 用途 |
| --- | --- |
| `myo_guided_collect.py`、`myo_runtime.py`、`myo_support/` | 归档的 Myo 采集和原依赖；依赖文件保持原样 |
| `train_emg_model.py`、`model.py`、`emg_pipeline.py` | 离线训练、轻量 CNN、布局/缩放/切窗 |
| `export_model_to_esp.py` | ESPDL 导出，默认读取 checkpoint 同目录的配置 |
| `esp32_client.py`、`esp32_replay_eval.py` | TCP 客户端和保存验证集的回放 |
| 三个 `*realtime_infer.py` 及 `dual_realtime_infer.py` | 归档的 Myo 实时对照入口 |
| `C++/`、`scripts/esp32.cmd` | 历史板端工程和 CMD 工具链入口 |
| `data/` | 早期五分类数据 |
| `datasets/collection_20260413/` | 后期数据与六分类训练数据 |
| `runs/` | 各实验 best checkpoint、配置、ESPDL、验证集、日志与结果 |
| `test_data/` | 早期导出的验证数据；保留原名以维持可追溯性 |
| `reports/` | 模型清单、复核摘要和清理记录，供报告使用 |

原始数据、各次有效实验的 `best.pt`、配套配置、有效 ESPDL 和验证集均保留。删除逐 epoch checkpoint、ONNX/JSON/INFO 中间文件和过时入口；临时实验配置移入本机备份。

源码、Git 历史及固件构建备份位于 `D:\Project\emg-esp32_archives\20261009\`。备份包含原本地配置，不能直接公开上传。

`data/`、`datasets/`、`runs/`、`test_data/` 不上传 Git。只克隆源码无法复现训练，需要取回配套本地实验文件。板端 ESPDL 与不含原始采集信号的结果摘要随代码保存。

## 环境与复现

以下命令在仓库根目录的 CMD 中执行。常规 Python 使用 `myo`，导出使用 `espdl-int8`；固件使用 ESP-IDF 5.5。源码板型为 ESP32-S3、8 MB Flash、八线 PSRAM。

```bat
set "PY=%USERPROFILE%\miniconda3\envs\myo\python.exe"
set "EXPORT_PY=%USERPROFILE%\miniconda3\envs\espdl-int8\python.exe"
```

历史采集协议为每类 50 次、每次 1 秒、间隔休息 1 秒，采集后切窗。若复现采集，请指定新数据集名称。

```bat
"%PY%" myo_guided_collect.py --dataset-name new_collection
"%PY%" train_emg_model.py --config config.yaml
```

训练保存验证集和 best checkpoint，不再每五个 epoch 保存大文件。

导出明确指定 checkpoint；未传 `--config` 时读取旁边的 `config.yaml`。默认输出会覆盖板端文件，复现建议先导出到实验目录。

```bat
"%EXPORT_PY%" export_model_to_esp.py --checkpoint runs/R6_r6_fp24_fc1024_20260413-154153/checkpoints/best.pt --output runs/R6_r6_fp24_fc1024_20260413-154153/espdl/sEMG.espdl
```

复制 `C++/main/network_config.h.example` 为同目录的 `network_config.h` 并填写 Wi-Fi。密码文件被 Git 忽略；本机路径可在 `local_settings.cmd` 设置，格式见 `local_settings.example.cmd`。

```bat
scripts\esp32.cmd probe COM10
scripts\esp32.cmd build
scripts\esp32.cmd flash COM10
scripts\esp32.cmd monitor COM10
```

`COM10` 为本轮识别的 USB 板端端口；后续以实际枚举为准。串口监视用 `Ctrl+]` 退出。

固定验证集回放：保存的 `X.npy` 已经除以 128，不能重复缩放。验证文件缺失时脚本报错，不改用训练数据计算准确率。

```bat
"%PY%" esp32_replay_eval.py --config runs/R6_r6_fp24_fc1024_20260413-154153/checkpoints/config.yaml --run-dir runs/R6_r6_fp24_fc1024_20260413-154153 --server-ip BOARD_IP
"%PY%" -m unittest discover -s tests -v
```

历史实时脚本默认使用根配置中明确指定的 checkpoint。切换模型时同时传入配套 `--config` 与 `--checkpoint`，避免五/六分类或结构混用。

## ESP32 后续主线

Gitee 的 `codex/esp32-standalone` 分支保存独立工程，个人私有仓库为 `phlas-end/esp32-standalone`。独立工程包括固件、模型规格、编译/烧录、回放验证和模型导出工具，不带 Myo DLL。

后续按新硬件补齐直接输入、端侧预处理和本地输出；重新核对采样率、单位、电极位置和输入分布。
