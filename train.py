from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from lib import command_helper, dataloaders, models, losses, best_model_name
from lib.progress_bar import CustomProgressBar


def train(args=None):
    command = command_helper.Command(args=args)

    if command.params.custom_seed is not None:
        seed_everything(command.params.custom_seed)

    train_loader, valid_loader = dataloaders.get_data_loader(command.params)

    network = models.get_network_model(command.params)

    use_custom_loss_function = False

    try:
        use_custom_loss_function = network.use_custom_loss_function
    except:
        pass

    if not use_custom_loss_function:
        loss_func = losses.get_loss_function(command.params)
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


if __name__ == '__main__':
    train()
