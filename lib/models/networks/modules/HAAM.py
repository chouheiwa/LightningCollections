import torch
import torch.nn as nn
#AAU-net: An Adaptive Attention U-net for Breast Lesions Segmentation in Ultrasound Images
#https://arxiv.org/pdf/2204.12077
"""
Channelblock类：
这个类首先使用两个不同大小的卷积核对输入特征图进行卷积，生成两组特征图。一个卷积核大小为3x3，并具有膨胀率3（dilation=3），另一个卷积核大小为5x5。
然后，每组特征图分别通过批量归一化（Batch Normalization）和ReLU激活函数。
这两组处理过的特征图被拼接在一起，并通过全局平均池化层（Global Average Pooling）。
接下来，拼接后的特征图通过一个全连接层，然后是批量归一化和ReLU激活函数。
最后，再通过一个全连接层和Sigmoid激活函数，产生一个权重向量，用于对原始的特征图进行加权，生成加权后的特征图。

Spatialblock类：
这个类对输入特征图应用一个3x3卷积，接着是批量归一化和ReLU激活函数。
接下来，通过一个1x1卷积以及另一次批量归一化和ReLU激活函数，产生空间特征。
将来自Channelblock类的通道加权特征图和空间特征图相加，然后应用激活函数。
加权后的空间特征图和原始特征图进行了组合，然后使用一个大小为size x size的卷积核进行最终的卷积处理，并再次通过批量归一化。

HAAM类（混合自适应注意力模块）：
这个类首先使用Channelblock处理输入特征图，得到通道注意力加权后的特征图。
然后，这些加权特征图和原始的特征图一起供Spatialblock使用，生成了最终的混合注意力特征图。
HAAM模块结合了通道注意力和空间注意力，使得网络能够更加关注重要的特征，可能提升乳腺病变区域在超声图像中的分割精度。
"""
def expend_as(tensor, rep):
    return tensor.repeat(1, rep, 1, 1)

class Channelblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Channelblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=3, dilation=3)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.batch2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels * 2, out_channels)
        self.batch3 = nn.BatchNorm1d(out_channels)
        # 使用组归一化替换批归一化
        #self.batch3 = nn.GroupNorm(1, out_channels)  # 组大小为1，可根据通道数调整

        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv3 = nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1, padding=0)
        self.batch4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        batch1 = self.batch1(conv1)
        relu1 = self.relu1(batch1)

        conv2 = self.conv2(x)
        batch2 = self.batch2(conv2)
        relu2 = self.relu2(batch2)

        combined = torch.cat([relu1, relu2], dim=1)
        pooled = self.global_avg_pool(combined)
        pooled = torch.flatten(pooled, 1)
        fc1 = self.fc1(pooled)
        batch3 = self.batch3(fc1)
        relu3 = self.relu3(batch3)
        fc2 = self.fc2(relu3)
        sigm = self.sigmoid(fc2)

        a = sigm.view(-1, sigm.size(1), 1, 1)
        a1 = 1 - sigm
        a1 = a1.view(-1, a1.size(1), 1, 1)

        y = relu1 * a
        y1 = relu2 * a1

        combined = torch.cat([y, y1], dim=1)

        conv3 = self.conv3(combined)
        batch4 = self.batch4(conv3)
        relu4 = self.relu4(batch4)

        return relu4

class Spatialblock(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(Spatialblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.batch2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.final_conv = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=size, padding=(size//2))
        self.batch3 = nn.BatchNorm2d(out_channels)

    def forward(self, x, channel_data):
        conv1 = self.conv1(x)
        batch1 = self.batch1(conv1)
        relu1 = self.relu1(batch1)

        conv2 = self.conv2(relu1)
        batch2 = self.batch2(conv2)
        spatil_data = self.relu2(batch2)

        data3 = torch.add(channel_data, spatil_data)
        data3 = torch.relu(data3)
        data3 = nn.Conv2d(data3.size(1), 1, kernel_size=1, padding=0)(data3)
        data3 = torch.sigmoid(data3)

        a = expend_as(data3, channel_data.size(1))
        y = a * channel_data

        a1 = 1 - data3
        a1 = expend_as(a1, spatil_data.size(1))
        y1 = a1 * spatil_data

        combined = torch.cat([y, y1], dim=1)

        conv3 = self.final_conv(combined)
        batch3 = self.batch3(conv3)

        return batch3

class HAAM(nn.Module):
    def __init__(self, in_channels, out_channels, size = 3):
        super(HAAM, self).__init__()
        self.channel_block = Channelblock(in_channels, out_channels)
        self.spatial_block = Spatialblock(out_channels, out_channels, size)

    def forward(self, x):
        channel_data = self.channel_block(x)
        haam_data = self.spatial_block(x, channel_data)
        return haam_data


if __name__ == '__main__':
    # 创建示例输入张量
    batch_size = 2
    in_channels = 64  # 输入通道数
    height, width = 224, 224  # 输入图像的高度和宽度
    input_tensor = torch.randn(batch_size, in_channels, height, width)

    # 实例化 HAAM 模型
    out_channels = 64  # 输出通道数
    haam_model = HAAM(in_channels, out_channels)

    # 前向传播
    output_tensor = haam_model(input_tensor)

    # 打印输入输出的形状
    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output_tensor.shape)