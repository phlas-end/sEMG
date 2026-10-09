# AGENTS.md

- 不用 PowerShell，使用 CMD 或指定 Python。
- Myo 方向于 2026-10-09 封版；此目录保留历史复现代码。后续在 `codex/esp32-standalone` 和个人私有仓库开发 ESP32。
- 不修改 `myo_support/` 的原依赖，不重建采集架构。
- 当前板端对应 `R6_r6_fp24_fc1024_20260413-154153`，六分类，`final_pool=[24,1]`、`fc_hidden=1024`。0 为静息态，1~5 为手势；能量门控关闭。
- 输入 `(N,200,8)`，模型输入 `(N,1,200,8)`；原始 Myo 数值除以 128，保存的验证集已缩放。
- 五分类历史板端 79.46% 不能作为六分类结果。模型、配置和验证集必须配套。
- 保留本地原始数据、各实验 best checkpoint、配置、有效 ESPDL、验证集和结果。实验目录默认不上传 Git。
- Wi-Fi 密码只放在被忽略的 `C++/main/network_config.h`；本机路径只放在 `local_settings.cmd`。
- 基准板型：ESP32-S3、8 MB Flash、八线 PSRAM。测试板差异不能改变基准规格。
- 普通 Python：`%USERPROFILE%\miniconda3\envs\myo\python.exe`；导出：`%USERPROFILE%\miniconda3\envs\espdl-int8\python.exe`；IDF：`%USERPROFILE%\esp\v5.5\esp-idf`。
- 使用说明和实验边界见 README；报告材料见 `reports/model_inventory.json`、`reports/verification.json`。
- 本机 `D:\Project\emg-esp32_archives\20261009\` 含原始本地配置，不公开上传。
