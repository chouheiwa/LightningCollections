import os
import re

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .command_helper import Command
from .dataloaders import get_test_data_loader, get_data_loader
from .losses import get_loss_function
from .models import get_network_model, SimpleImageSegmentationModel
from .progress_bar import CustomProgressBar

best_model_name = 'best_model'
best_train_loss = 'best_train_loss'
best_val_loss = 'best_val_loss'


def get_best_model_checkpoint(opt):
    best_path = None
    best_number = -1000
    base_path = os.path.join(opt.run_dir, 'lightning_logs')
    for data in os.listdir(base_path):
        match = re.search(r'\d+', data)
        if match:
            number = int(match.group())
            if number > best_number:
                best_number = number
                best_path = data
    return best_number, os.path.join(base_path, best_path, 'checkpoints', f'{best_model_name}.ckpt'), os.path.join(
        base_path, best_path, 'checkpoints', f'last.ckpt')


def test(args=None):
    command = Command(isTest=True, args=args)

    best_number, best_ckpt_path, last_ckpt_path = get_best_model_checkpoint(command.params)

    if command.params.best_model_path is not None:
        best_ckpt_path = command.params.best_model_path

    command.params.run_dir = os.path.join(command.params.run_dir, 'lightning_logs', f'version_{best_number}')

    test_loader = get_test_data_loader(command.params)

    network = get_network_model(command.params, isTrain=False)

    model = SimpleImageSegmentationModel(net=network, loss_func=None, opt=command.params)

    trainer = Trainer(
        default_root_dir=command.params.run_dir,
        benchmark=True,
        inference_mode=False,
        callbacks=[CustomProgressBar(command.params.dataset_name, command.params.model_name)]
    )
    trainer.test(model, test_loader, ckpt_path=best_ckpt_path)


def train(args=None):
    command = command_helper.Command(args=args)

    if command.params.custom_seed is not None:
        seed_everything(command.params.custom_seed)

    train_loader, valid_loader = get_data_loader(command.params)

    network = get_network_model(command.params)

    use_custom_loss_function = False

    try:
        use_custom_loss_function = network.use_custom_loss_function
    except:
        pass

    if not use_custom_loss_function:
        loss_func = get_loss_function(command.params)
    else:
        loss_func = None

    model = models.SimpleImageSegmentationModel(net=network, loss_func=loss_func, opt=command.params)
    callbacks = [
        CustomProgressBar(command.params.dataset_name, command.params.model_name),
        ModelCheckpoint(
            filename=best_model_name,
            monitor='val/BinaryJaccardIndex',
            mode='max',
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
            save_on_train_epoch_end=True,
            enable_version_counter=False,
        )
    ]
    if command.params.need_early_stop:
        callbacks.append(EarlyStopping(monitor="val/BinaryJaccardIndex", mode="max", patience=30))

    trainer = Trainer(
        accelerator=command.params.accelerator,
        devices=command.params.devices,
        default_root_dir=command.params.run_dir,
        benchmark=True,
        max_epochs=command.params.end_epoch,
        callbacks=callbacks
    )
    trainer.fit(model, train_loader, valid_loader)
