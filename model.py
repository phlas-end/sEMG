import torch
import torch.nn as nn


class EMG2DCNN(nn.Module):
    def __init__(self, input_shape, model_cfg, num_classes):
        super().__init__()

        in_channels = input_shape[0]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.final_pool = None

        for layer_cfg in model_cfg["conv_layers"]:
            out_channels = layer_cfg["out_channels"]
            kernel_size = layer_cfg["kernel_size"]
            pool_size = layer_cfg.get("pool_size", None)

            conv = nn.Conv2d(in_channels, out_channels, kernel_size)
            bn = nn.BatchNorm2d(out_channels)
            self.convs.append(conv)
            self.bns.append(bn)

            if pool_size:
                pool = nn.MaxPool2d(kernel_size=(pool_size, 1))
            else:
                pool = None
            self.pools.append(pool)

            in_channels = out_channels

        final_pool = model_cfg.get("final_pool")
        if final_pool:
            self.final_pool = nn.AdaptiveAvgPool2d(tuple(final_pool))

        # Keep the fully-connected layer size derived from the configured input shape.
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = dummy
            for conv, bn, pool in zip(self.convs, self.bns, self.pools):
                x = conv(x)
                x = bn(x)
                x = self.relu(x)
                if pool:
                    x = pool(x)
            if self.final_pool:
                x = self.final_pool(x)
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
        if self.final_pool:
            x = self.final_pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
