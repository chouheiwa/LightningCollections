import torch
from torch import nn
import lightning as L

from lib.utils import expand_as_one_hot


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleConv, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            SingleConv(in_channels, mid_channels),
            SingleConv(mid_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class PoolConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PoolConv, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class SqueezeExcitationLayer(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels):
        super(SqueezeExcitationLayer, self).__init__()
        self.conv = DoubleConv(in_channels + in_channels_skip, out_channels)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.dense_layer = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels // 4),
            nn.ReLU(),
            nn.Linear(in_features=out_channels // 4, out_features=out_channels),
            nn.Sigmoid()
        )

    def forward(self, x, skip_data):
        concatenate = self.conv(torch.cat([x, skip_data], dim=1))
        squeeze = self.squeeze(concatenate)
        squeeze = squeeze.view(squeeze.size(0), -1)
        # 由 (b ,out_channels, 1, 1) 展开成为 (b, out_channels)
        dense = self.dense_layer(squeeze)
        dense = dense.view(dense.size(0), dense.size(1), 1, 1)
        scale = concatenate * dense
        return torch.cat([x, scale], dim=1)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.se_layer = SqueezeExcitationLayer(in_channels, in_channels_skip, out_channels)
        self.out_conv = DoubleConv(in_channels + out_channels, out_channels)

    def forward(self, x, skip_data):
        x = self.up(x)
        se_layer = self.se_layer(x, skip_data)
        return self.out_conv(se_layer)


class DSEUNet(L.LightningModule):
    def __init__(self, num_classes, image_size):
        super(DSEUNet, self).__init__()
        self.n_classes = num_classes

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        dims = [64, 128, 128, 256, 256, 512, 512, 1024]

        self.down_blocks = nn.ModuleList()

        self.down_blocks.append(DoubleConv(3, dims[0]))

        for i in range(1, len(dims)):
            self.down_blocks.append(PoolConv(dims[i - 1], dims[i]))

        self.up_blocks = nn.ModuleList()
        self.predict_outs = nn.ModuleList()

        for i in range(len(dims) - 1, 0, -1):
            self.up_blocks.append(
                UpConvBlock(dims[i], dims[i - 1], dims[i - 1])
            )

        self.out_conv = nn.Conv2d(in_channels=dims[0], out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1),
                                  padding=(0, 0))

    def forward(self, x):
        down_list = []
        conv = x
        for down in self.down_blocks:
            conv = down(conv)
            down_list.append(conv)

        up = down_list[-1]
        for i in range(len(self.up_blocks)):
            up = self.up_blocks[i](up, down_list[-2 - i])
        return self.out_conv(up)
