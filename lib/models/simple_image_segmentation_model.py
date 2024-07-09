import os
from typing import Union, Sequence, Any

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from lib.metrics import get_binary_metrics
from lib import best_model_name


class SimpleImageSegmentationModel(L.LightningModule):
    def __init__(self, net, loss_func, opt):
        super().__init__()
        self.net = net
        self.loss_func = loss_func
        self.normalization = nn.Sigmoid()
        self.opt = opt
        metrics = get_binary_metrics()
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

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

    def training_step(self, batch, batch_idx):
        return self._on_step(batch, batch_idx, self.train_metrics)

    def on_train_epoch_end(self) -> None:
        dic = self.train_metrics.compute()
        self.log_dict(dic)
        self.log('train_IOU', dic['train/BinaryJaccardIndex'], prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        return self._on_step(batch, batch_idx, self.valid_metrics)

    def on_validation_epoch_end(self):
        dic = self.valid_metrics.compute()
        self.log_dict(dictionary=dic)
        self.log('val_IOU', dic['val/BinaryJaccardIndex'], prog_bar=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        input_tensor, target, image_names = batch
        output = self(input_tensor)
        predict = self.normalization(output)
        # 将预测图像进行分割
        predict_latest = torch.argmax(predict, dim=1)
        self.test_metrics.update(predict_latest.float(), target.int())

        for i in range(predict_latest.size(0)):
            # predict[predict == 1] =
            base_path = self.opt.result_dir
            os.makedirs(base_path, exist_ok=True)

            torchvision.utils.save_image(
                predict_latest[i].float(),
                os.path.join(
                    self.opt.result_dir,
                    f'{image_names[i]}.png'
                )
            )

    def on_test_epoch_end(self):
        logger = CSVLogger(self.opt.result_dir, name='result', version="")
        dic = self.test_metrics.compute()
        logger.log_metrics(dic)
        logger.save()
        self.test_metrics.reset()

    def configure_optimizers(self):
        if self.opt["optimizer_name"] == "SGD":
            optimizer = optim.SGD(self.net.parameters(), lr=self.opt["learning_rate"], momentum=self.opt["momentum"],
                                  weight_decay=self.opt["weight_decay"])

        elif self.opt["optimizer_name"] == 'Adagrad':
            optimizer = optim.Adagrad(self.net.parameters(), lr=self.opt["learning_rate"],
                                      weight_decay=self.opt["weight_decay"])

        elif self.opt["optimizer_name"] == "RMSprop":
            optimizer = optim.RMSprop(self.net.parameters(), lr=self.opt["learning_rate"],
                                      weight_decay=self.opt["weight_decay"],
                                      momentum=self.opt["momentum"])

        elif self.opt["optimizer_name"] == "Adam":
            optimizer = optim.Adam(self.net.parameters(), lr=self.opt["learning_rate"],
                                   weight_decay=self.opt["weight_decay"])

        elif self.opt["optimizer_name"] == "AdamW":
            optimizer = optim.AdamW(self.net.parameters(), lr=self.opt["learning_rate"],
                                    weight_decay=self.opt["weight_decay"])

        elif self.opt["optimizer_name"] == "Adamax":
            optimizer = optim.Adamax(self.net.parameters(), lr=self.opt["learning_rate"],
                                     weight_decay=self.opt["weight_decay"])

        elif self.opt["optimizer_name"] == "Adadelta":
            optimizer = optim.Adadelta(self.net.parameters(), lr=self.opt["learning_rate"],
                                       weight_decay=self.opt["weight_decay"])

        else:
            raise RuntimeError(f"No {self.opt['optimizer_name']} optimizer available")
        return {
            "optimizer": optimizer,
        }

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        early_stop = EarlyStopping(monitor="val/BinaryJaccardIndex", mode="max", patience=20)
        best_check_point = ModelCheckpoint(
            filename=best_model_name,
            monitor='val/BinaryJaccardIndex',
            mode='max',
            save_top_k=1,
            save_on_train_epoch_end=False,
            enable_version_counter=False,
        )
        return [best_check_point, early_stop]
