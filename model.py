import torch
import torch.nn as nn

class EMG2DCNN(nn.Module):
    def __init__(self, input_shape, model_cfg, num_classes):
        super().__init__()

        in_channels = input_shape[0]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # 新增 BatchNorm 模块列表
        self.pools = nn.ModuleList()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        for layer_cfg in model_cfg["conv_layers"]:
            out_channels = layer_cfg["out_channels"]
            kernel_size = layer_cfg["kernel_size"]
            pool_size = layer_cfg.get("pool_size", None)

            conv = nn.Conv2d(in_channels, out_channels, kernel_size)
            bn = nn.BatchNorm2d(out_channels)  # 新增 BatchNorm2d
            self.convs.append(conv)
            self.bns.append(bn)

            if pool_size:
                pool = nn.MaxPool2d(pool_size)
            else:
                pool = None
            self.pools.append(pool)

            in_channels = out_channels

        # 计算全连接输入尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = dummy
            for conv, bn, pool in zip(self.convs, self.bns, self.pools):
                x = conv(x)
                x = bn(x)
                x = self.relu(x)
                if pool:
                    x = pool(x)
            fc_input_dim = x.numel()

        self.fc1 = nn.Linear(fc_input_dim, model_cfg["fc_hidden"])
        self.dropout_fc = nn.Dropout(model_cfg["dropout_rate"])
        self.fc2 = nn.Linear(model_cfg["fc_hidden"], num_classes)

    def forward(self, x):
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
            if pool:
                x = pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
