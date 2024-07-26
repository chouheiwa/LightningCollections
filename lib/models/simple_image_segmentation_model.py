import os
from typing import Union, Sequence, Any

import lightning as L
import torch
import torch.nn as nn
import torchvision
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.model_helpers import is_overridden

from lib.metrics import get_binary_metrics, cal_params_flops
from lib import best_model_name, lr_scheduler
from lib.optimizer import get_optimizer


class SimpleImageSegmentationModel(L.LightningModule):
    def __init__(self, net, opt, loss_func):
        super().__init__()
        self.net = net
        self.loss_func = loss_func
        self.normalization = nn.Sigmoid()
        self.opt = opt
        metrics = get_binary_metrics()
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        self.train_losses = []
        self.valid_losses = []

    def forward(self, x):
        return self.net(x)

    def _on_step(self, batch, batch_idx, metrics):
        input_tensor, target, _ = batch
        output = self(input_tensor)
        predict = self.normalization(output)
        predict = torch.argmax(predict, dim=1)
        metrics.update(predict.float(), target.int())

        loss = self.loss_func(output, target)
        return loss

    def on_train_epoch_start(self):
        self.train_metrics.reset()
        self.train_losses.clear()

    def training_step(self, batch, batch_idx):
        has_step = is_overridden("training_step", self.net)

        if has_step:
            loss = self.net.training_step(batch, batch_idx,
                                          train_metrics=self.train_metrics,
                                          normalization=self.normalization)
        else:
            loss = self._on_step(batch, batch_idx, self.train_metrics)
        self.train_losses.append(loss.item())
        return loss
    def on_train_epoch_end(self):
        dic = self.train_metrics.compute()
        self.log_dict(dic)
        self.log('train_loss', torch.tensor(self.train_losses).mean(), prog_bar=True)
        self.log('train_IOU', dic['train/BinaryJaccardIndex'], prog_bar=True)

    def on_validation_epoch_start(self):
        self.valid_metrics.reset()
        self.valid_losses.clear()

    def validation_step(self, batch, batch_idx):
        has_step = is_overridden("validation_step", self.net)

        if has_step:
            loss = self.net.validation_step(batch, batch_idx, valid_metrics=self.valid_metrics,
                                            normalization=self.normalization)
        else:
            loss = self._on_step(batch, batch_idx, self.valid_metrics)
        self.valid_losses.append(loss.item())
        return loss
    def on_validation_epoch_end(self):
        dic = self.valid_metrics.compute()
        self.log_dict(dictionary=dic)
        self.log('val_IOU', dic['val/BinaryJaccardIndex'], prog_bar=True)
        self.log('val_loss', torch.tensor(self.valid_losses).mean(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        has_step = is_overridden("test_step", self.net)
        input_tensor, target, image_names = batch
        if has_step:
            output = self.net.test_step(batch, batch_idx)
        else:
            output = self(input_tensor)
        predict = self.normalization(output)
        predict = torch.argmax(predict, dim=1)
        # 将预测图像进行分割
        if not self.opt.forbid_metrics:
            self.test_metrics.update(predict.float(), target.int())

        for i in range(predict.size(0)):
            # predict[predict == 1] =
            base_path = self.opt.result_dir
            os.makedirs(base_path, exist_ok=True)

            torchvision.utils.save_image(
                predict[i].float(),
                os.path.join(
                    self.opt.result_dir,
                    f'{image_names[i]}.png'
                )
            )

    def on_test_epoch_end(self):
        if self.opt.forbid_metrics:
            return
        params, flops = cal_params_flops(self.net, self.device, self.opt.resize_shape)
        logger = CSVLogger(self.opt.result_dir, name='result', version="")
        dic = self.test_metrics.compute()
        self.log_dict(dic, prog_bar=True)
        dic['Parameters'] = params
        dic['FLOPs'] = flops
        logger.log_metrics(dic)
        logger.save()
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.opt, self.net)
        lr_schedulers = lr_scheduler.get_lr_scheduler(optimizer, self.opt)

        if lr_schedulers is None:
            return {
                "optimizer": optimizer,
            }

        lr_scheduler_lightning_config = self.opt.lr_scheduler_config.lightning_config
        lr_scheduler_lightning_config = lr_scheduler_lightning_config.get_dict() if lr_scheduler_lightning_config is not None else {}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedulers,
                **lr_scheduler_lightning_config,
            }
        }

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        callbacks = [
            ModelCheckpoint(
                filename=best_model_name,
                monitor='val/BinaryJaccardIndex',
                mode='max',
                save_top_k=1,
                save_last=True,
                save_weights_only=True,
                save_on_train_epoch_end=True,
                enable_version_counter=False,
            )
        ]
        if self.opt.need_early_stop:
            callbacks.append(EarlyStopping(monitor="val/BinaryJaccardIndex", mode="max", patience=20))
        return callbacks
