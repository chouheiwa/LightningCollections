from torch import nn

from lib.utils import *
import lightning as L


class DiceLoss(L.LightningModule):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    Use pytorch lightning Module for automatic configure the device.
    """

    def __init__(self, classes=1, weight=None, sigmoid_normalization=False, mode="extension"):
        assert (weight is not None, "Weight is required for DiceLoss")
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.register_buffer('weight', torch.FloatTensor(weight), persistent=False)
        self.mode = mode

        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def dice(self, input, target):
        target = expand_as_one_hot(target.long(), self.classes)

        assert input.size() == target.size(), "Inconsistency of dimensions between predicted and labeled images after one-hot processing in dice loss"

        input = self.normalization(input)

        return compute_per_channel_dice(input, target, epsilon=1e-6, mode=self.mode)

    def forward(self, input, target):
        per_channel_dice = self.dice(input, target)

        real_weight = self.weight.clone()
        for i, dice in enumerate(per_channel_dice):
            if dice == 0:
                real_weight[i] = 0

        weighted_dsc = torch.sum(per_channel_dice * real_weight) / torch.sum(real_weight)

        loss = 1. - weighted_dsc

        return loss
