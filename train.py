from lightning import Trainer

from lib import command_helper, dataloaders, models, losses

if __name__ == '__main__':
    command = command_helper.Command()

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

    trainer = Trainer(
        default_root_dir=command.params.run_dir,
        benchmark=True,
        max_epochs=command.params.end_epoch,
    )
    trainer.fit(model, train_loader, valid_loader)
