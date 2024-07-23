import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class SingleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = SingleConvBlock(in_channels, out_channels)
        self.conv2 = SingleConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PoolConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(PoolConvBlock, self).__init__()

        if kernel_size is int:
            kernel_size = (kernel_size, kernel_size)

        self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        pool = self.pool(x)
        conv = self.conv(pool)
        return conv, pool


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class UpData1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpData1, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, data, skipdata):
        data = F.interpolate(data, size=(skipdata.size(2), skipdata.size(3)))
        x = torch.cat([skipdata, data], dim=1)
        x = self.conv_block(x)
        return x


class UpSampleDataBlock(nn.Module):
    def __init__(self, conv_in_channels, data_in_channels, out_channels, up_conv_times):
        super(UpSampleDataBlock, self).__init__()
        datas = []
        if up_conv_times > 0:
            for _ in range(up_conv_times):
                datas.append(UpConvBlock(conv_in_channels, conv_in_channels))
        datas.append(nn.Upsample(scale_factor=4))
        self.up = nn.Sequential(*datas)
        self.updata = UpData1(conv_in_channels * 2 + data_in_channels, out_channels)

    def forward(self, data, conv1, conv2):
        conv1 = self.up(conv1)
        temp = torch.cat([conv2, conv1], dim=1)
        x = self.updata(data, temp)
        return x


class DeepUNet7(L.LightningModule):
    def __init__(self, num_classes):
        super(DeepUNet7, self).__init__()
        self.conv1 = ConvBlock(3, 64)

        self.conv2 = PoolConvBlock(64, 128, 2)  # 128

        self.conv31 = PoolConvBlock(64, 64, 4)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128
        self.conv3_temp = SingleConvBlock(in_channels=64, out_channels=32)
        self.conv3 = ConvBlock(32 + 128, 128)  # conv3 = conv3_temp(32) + pool2(128) => 128

        self.conv4 = PoolConvBlock(128, 256, 2)
        self.conv41 = PoolConvBlock(128, 128, 4)
        self.conv42 = PoolConvBlock(64, 64, 2)

        self.conv51 = PoolConvBlock(128, 128, 4)
        self.conv52 = PoolConvBlock(128, 128, 2)
        self.conv53 = PoolConvBlock(64, 64, 2)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256 => conv4
        self.conv5_temp = SingleConvBlock(in_channels=128, out_channels=32)
        self.conv5 = ConvBlock(32 + 256, 256)  # conv3 = conv3_temp(32) + pool2(128) => 128

        self.conv6 = PoolConvBlock(256, 512, 2)
        self.conv61 = PoolConvBlock(256, 256, 4)
        self.conv62 = PoolConvBlock(128, 128, 2)
        self.conv63 = PoolConvBlock(128, 128, 2)
        self.conv64 = PoolConvBlock(64, 64, 2)

        self.conv71 = PoolConvBlock(256, 256, 4)
        self.conv72 = PoolConvBlock(256, 256, 2)
        self.conv73 = PoolConvBlock(128, 128, 2)
        self.conv74 = PoolConvBlock(128, 128, 2)
        self.conv75 = PoolConvBlock(64, 64, 2)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7_temp = SingleConvBlock(in_channels=256, out_channels=32)
        self.conv7 = ConvBlock(32 + 512, 512)

        self.conv8 = PoolConvBlock(512, 1024, 2)
        self.conv81 = PoolConvBlock(512, 512, 4)
        self.conv82 = PoolConvBlock(256, 256, 2)
        self.conv83 = PoolConvBlock(256, 256, 2)
        self.conv84 = PoolConvBlock(128, 128, 2)
        self.conv85 = PoolConvBlock(128, 128, 2)
        self.conv86 = PoolConvBlock(64, 64, 2)

        # 6
        self.up1 = UpData1(1024 + 512 + 256 + 256 + 128 + 128 + 64 + 512, 512)

        # 12
        self.up2 = UpSampleDataBlock(conv_in_channels=512, data_in_channels=512, out_channels=512, up_conv_times=0)

        # 24
        self.up3 = UpSampleDataBlock(conv_in_channels=256, data_in_channels=512, out_channels=256, up_conv_times=1)

        self.up4 = UpSampleDataBlock(
            conv_in_channels=256, # conv1->conv83(256, 2, 2)
            data_in_channels=256, # data->up3(256, 16, 16)
            out_channels=256,
            up_conv_times=2
        )  # => 256, 32, 32
        self.up5 = UpSampleDataBlock(
            conv_in_channels=128, # conv1->conv84(128, 2, 2)
            data_in_channels=256, # data->up4(256, 32, 32)
            out_channels=128,
            up_conv_times=3
        )  # => 128, 64, 64
        self.up6 = UpSampleDataBlock(
            conv_in_channels=128, # conv1->conv85(128, 2, 2)
            data_in_channels=128, # data->up5(128, 64, 64)
            out_channels=128,
            up_conv_times=4
        ) # => 128, 128, 128
        self.up7 = UpSampleDataBlock(
            conv_in_channels=64, # conv1->conv86(64, 2, 2)
            data_in_channels=128, # data->up6(128, 128, 128)
            out_channels=64,
            up_conv_times=5
        ) # => 64, 256, 256

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)

        conv2, _ = self.conv2(conv1)

        conv31, pool21 = self.conv31(conv1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(torch.cat([
            self.conv3_temp(pool21),
            pool2
        ], dim=1))

        conv4, _ = self.conv4(conv3)
        conv41, _ = self.conv41(conv2)
        conv42, _ = self.conv42(conv31)

        conv51, pool41 = self.conv51(conv3)
        conv52, _ = self.conv52(conv41)
        conv53, _ = self.conv53(conv42)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(torch.cat([
            self.conv5_temp(pool41),
            pool4
        ], dim=1))

        conv6, _ = self.conv6(conv5)
        conv61, _ = self.conv61(conv4)
        conv62, _ = self.conv62(conv51)
        conv63, _ = self.conv63(conv52)
        conv64, _ = self.conv64(conv53)

        conv71, pool61 = self.conv71(conv5)
        conv72, _ = self.conv72(conv61)
        conv73, _ = self.conv73(conv62)
        conv74, _ = self.conv74(conv63)
        conv75, _ = self.conv75(conv64)
        pool6 = self.pool6(conv6)
        conv7 = self.conv7(torch.cat([
            self.conv7_temp(pool61),
            pool6
        ], dim=1))

        conv8, _ = self.conv8(conv7)
        conv81, _ = self.conv81(conv6)
        conv82, _ = self.conv82(conv71)
        conv83, _ = self.conv83(conv72)
        conv84, _ = self.conv84(conv73)
        conv85, _ = self.conv85(conv74)
        conv86, _ = self.conv86(conv75)

        up1 = self.up1(torch.cat([
            conv8, conv81, conv82, conv83, conv84, conv85, conv86
        ], dim=1), conv7)

        up2 = self.up2(up1, conv81, conv6)
        up3 = self.up3(up2, conv82, conv5)
        up4 = self.up4(up3, conv83, conv4)
        up5 = self.up5(up4, conv84, conv3)
        up6 = self.up6(up5, conv85, conv2)
        up7 = self.up7(up6, conv86, conv1)

        return self.out_conv(up7)
