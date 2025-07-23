import torch
import torch.nn as nn

# -----------------------------
# CNN模型（动态 ModuleList 版本，加入 Dropout）
# -----------------------------
class EMG2DCNN(nn.Module):
    def __init__(self, input_shape, model_cfg, num_classes):
        super().__init__()

        in_channels = input_shape[0]
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropouts = nn.ModuleList()

        dropout_rate = model_cfg.get("dropout", 0.5)

        for layer_cfg in model_cfg["conv_layers"]:
            out_channels = layer_cfg["out_channels"]
            kernel_size = layer_cfg["kernel_size"]
            pool_size = layer_cfg.get("pool_size", None)

            conv = nn.Conv2d(in_channels, out_channels, kernel_size)
            self.convs.append(conv)

            if pool_size:
                pool = nn.MaxPool2d(pool_size)
            else:
                pool = None
            self.pools.append(pool)

            # 每个卷积层后加 Dropout
            self.dropouts.append(nn.Dropout2d(dropout_rate))

            in_channels = out_channels

        # 计算全连接输入尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = dummy
            for conv, pool, dropout in zip(self.convs, self.pools, self.dropouts):
                x = conv(x)
                x = self.relu(x)
                x = dropout(x)
                if pool:
                    x = pool(x)
            fc_input_dim = x.numel()

        self.fc1 = nn.Linear(fc_input_dim, model_cfg["fc_hidden"])
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(model_cfg["fc_hidden"], num_classes)

    def forward(self, x):
        for conv, pool, dropout in zip(self.convs, self.pools, self.dropouts):
            x = conv(x)
            x = self.relu(x)
            x = dropout(x)
            if pool:
                x = pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
