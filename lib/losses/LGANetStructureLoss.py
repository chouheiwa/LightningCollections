from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from lib.utils import expand_as_one_hot


class LGANetStructureLoss(L.LightningModule):
    def __init__(self, classes=1, need_expand=True):
        super(LGANetStructureLoss, self).__init__()
        self.classes = classes
        self.need_expand = need_expand

    def forward(self, pred, mask) -> Any:
        if (self.need_expand or self.classes > 1) and mask.shape != pred.shape:
            mask = expand_as_one_hot(mask.long(), self.classes)

        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce=None)
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()
