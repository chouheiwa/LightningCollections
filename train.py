from lightning import Trainer

from lib import command_helper, dataloaders, models, losses
from lib.progess_bar import MyProgressBar


if __name__ == '__main__':
    command = command_helper.Command()

    train_loader, valid_loader = dataloaders.get_data_loader(command.params)

    network = models.get_network_model(command.params)

    loss_func = losses.get_loss_function(command.params)

    model = models.SimpleImageSegmentationModel(net=network, loss_func=loss_func, opt=command.params)

    trainer = Trainer(
        default_root_dir=command.params.run_dir,
        callbacks=[MyProgressBar()],
        benchmark=True,
        max_epochs=command.params.end_epoch,
    )
    trainer.fit(model, train_loader, valid_loader)
