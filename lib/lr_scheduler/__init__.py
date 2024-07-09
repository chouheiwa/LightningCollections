import torch.optim as optim

def get_lr_scheduler(optimizer, config):
    if config.lr_scheduler_name == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

    elif config.lr_scheduler_name == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    elif config.lr_scheduler_name == "MultiStepLR":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)

    elif config.lr_scheduler_name == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max)

    elif config.lr_scheduler_name == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0,
                                                                      T_mult=config.T_mult)

    elif config.lr_scheduler_name == "OneCycleLR":
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate,
                                                     steps_per_epoch=config.steps_per_epoch, epochs=config.end_epoch,
                                                     cycle_momentum=False)

    elif config.lr_scheduler_name == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=config.mode, factor=config.factor,
                                                            patience=config.patience)
    else:
        raise RuntimeError(f"No {config.lr_scheduler_name} lr_scheduler available")

    return lr_scheduler