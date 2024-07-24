import torch

from lib.losses.LGANetStructureLoss import LGANetStructureLoss
from .TransFuse import TransFuse_S
import lightning as L


class TransFuse(L.LightningModule):
    def __init__(self, num_classes, **kwargs):
        super(TransFuse, self).__init__()
        self.model = TransFuse_S(num_classes=num_classes, **kwargs)
        self.use_custom_loss_function = True
        self.loss_func = LGANetStructureLoss(classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _on_step(self, batch, metrics, normalization):
        input_tensor, target, _ = batch
        lateral_map_4, lateral_map_3, lateral_map_2 = self(input_tensor)
        predict = normalization(lateral_map_2)
        predict = torch.argmax(predict, dim=1)
        metrics.update(predict.float(), target.int())
        loss2 = self.loss_func(lateral_map_2, target)
        return lateral_map_4, lateral_map_3, lateral_map_2, loss2, target

    def training_step(self, batch, batch_idx, train_metrics, normalization):
        lateral_map_4, lateral_map_3, lateral_map_2, loss2, target = self._on_step(batch, train_metrics, normalization)
        loss4 = self.loss_func(lateral_map_4, target)
        loss3 = self.loss_func(lateral_map_3, target)
        return 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4

    def validation_step(self, batch, batch_idx, valid_metrics, normalization):
        _, _, _, loss2, _ = self._on_step(batch, valid_metrics, normalization)
        return loss2

    def test_step(self, batch, batch_idx):
        img, msk, _ = batch
        _, _, msk_pred = self(img)
        return msk_pred
