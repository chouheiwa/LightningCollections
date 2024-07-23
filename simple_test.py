import torch

from lib.models.networks.TransFuse import TransFuse_S

if __name__ == '__main__':
    model = TransFuse_S().cuda()
    r1, r2, r3 = model.forward(torch.randn(1, 3, 224, 224).cuda())
    print("Shape: ", r1.shape, r2.shape, r3.shape)