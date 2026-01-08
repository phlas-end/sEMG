# sEMG 信号处理与模型部署项目

## 项目概述
本项目是一个完整的表面肌电信号(sEMG)处理系统，实现了从模型训练到ESP32设备部署的全流程解决方案。

## 版本控制文件结构
当前已纳入版本控制的文件包括：
### 1. 核心Python文件
```
├── model.py             # 模型定义文件
├── sEMG.py             # 主程序实现
├── model2esp.py        # 模型转ESP格式工具
├── pt2onnx.py          # PyTorch模型转ONNX工具
└── wifi_connect.py     # WiFi连接实用工具
```

### 2. 配置文件
```
├── config.yaml          # 项目配置文件
├── launch_tensorboard.bat  # TensorBoard启动脚本
└── onnx2ncnn.txt       # ONNX转NCNN配置说明
```

### 3. ESP32实现 (C++/)
```
C++/
├── CMakeLists.txt                    # 主CMake配置
├── partitions.csv                    # 分区表配置
├── sdkconfig.defaults                # 默认SDK配置
├── sdkconfig.defaults.esp32p4        # ESP32-P4配置
├── sdkconfig.defaults.esp32s3        # ESP32-S3配置
└── main/
    ├── CMakeLists.txt               # 组件CMake配置
    ├── app_main.cpp                 # 应用主程序
    ├── idf_component.yml            # 组件配置
    └── models/                      # 模型文件
        ├── p4/
        │   └── model.espdl         # ESP32-P4模型
        └── s3/
            ├── model.espdl         # ESP32-S3模型
            └── sEMG.espdl         # sEMG专用模型
```

## 主要功能模块

### 1. Python核心模块
- `model.py`: 定义了深度学习模型架构
- `sEMG.py`: 实现主要的训练和处理逻辑
- `model2esp.py`: 将训练好的模型转换为ESP设备可用格式
- `pt2onnx.py`: 实现PyTorch模型到ONNX格式的转换
- `wifi_connect.py`: 提供WiFi连接功能支持

### 2. ESP32部署模块
- 完整的ESP32应用实现
- 支持ESP32-S3和ESP32-P4两种型号
- 包含特定模型部署文件（.espdl格式）
- 完整的构建系统配置（CMake，分区表等）

### 3. 工具支持
- TensorBoard集成（通过launch_tensorboard.bat）
- 模型转换工具链
- SDK配置文件

## 核心功能
1. 深度学习模型训练
2. 多平台模型转换
   - PyTorch → ONNX
   - ONNX → ESP专用格式
3. ESP32设备支持
   - ESP32-S3适配
   - ESP32-P4适配
4. 开发工具支持
   - TensorBoard可视化
   - WiFi连接工具
   - 构建系统配置

## 使用说明
1. 环境配置
   - 参考`config.yaml`进行基本配置
   - 确保ESP-IDF环境已正确设置（用于ESP32开发）

2. 模型开发流程
   - 使用`model.py`定义模型结构
   - 通过`sEMG.py`进行模型训练
   - 使用`launch_tensorboard.bat`监控训练过程

3. 模型转换
   - 使用`pt2onnx.py`将PyTorch模型转换为ONNX格式
   - 使用`model2esp.py`将模型转换为ESP设备格式
   - 参考`onnx2ncnn.txt`了解更多转换细节

4. ESP32部署
   - 根据目标设备选择适当的`sdkconfig.defaults`
   - 使用CMake进行项目构建
   - 将转换后的模型放置在正确的目录下
   - 编译并烧录代码到设备

## 贡献者
- phlas-end
