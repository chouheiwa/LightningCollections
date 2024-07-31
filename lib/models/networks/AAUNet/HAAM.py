import torch
import torch.nn as nn


def expend_as(tensor, rep):
    return tensor.repeat(1, rep, 1, 1)

class Flatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return torch.flatten(x, self.start_dim, self.end_dim)

class ChannelBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.mediator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(1),
            nn.Linear(out_channels * 2, out_channels),  # conv1 + conv2
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)

        conv2 = self.conv2(x)

        sigm = self.mediator(torch.cat([conv1, conv2], dim=1))
        a = sigm.view(-1, sigm.size(1), 1, 1)
        a1 = 1 - a
        y = conv1 * a
        y1 = conv2 * a1
        return self.conv3(torch.cat([y, y1], dim=1))


class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(SpatialBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=size,
                      padding=(size // 2)),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x, channel_data):
        conv = self.conv(x)
        conv = self.conv1(conv)
        conv2 = self.conv2(channel_data + conv)

        a = expend_as(conv2, channel_data.size(1))
        y = a * channel_data

        a1 = 1 - a
        y1 = a1 * conv

        combined = torch.cat([y, y1], dim=1)

        return self.conv3(combined)


class HAAM(nn.Module):
    def __init__(self, in_channels, out_channels, size=3):
        super(HAAM, self).__init__()
        self.channel_block = ChannelBlock(in_channels, out_channels)
        self.spatial_block = SpatialBlock(in_channels, out_channels, size)

    def forward(self, x):
        channel_data = self.channel_block(x)
        haam_data = self.spatial_block(x, channel_data)
        return haam_data
