from torch.optim import lr_scheduler

def get_lr_scheduler(optimizer, config):
    name = config.lr_scheduler_config.name
    scheduler = config.lr_scheduler_config.scheduler
    scheduler = scheduler.get_dict() if scheduler is not None else {}

    if name == "ExponentialLR":
        lrs = lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

    elif name == "StepLR":
        lrs = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    elif name == "MultiStepLR":
        lrs = lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)

    elif name == "CosineAnnealingLR":
        lrs = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max)

    elif name == "CosineAnnealingWarmRestarts":
        lrs = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler)

    elif name == "OneCycleLR":
        lrs = lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate,
                                                     steps_per_epoch=config.steps_per_epoch, epochs=config.end_epoch,
                                                     cycle_momentum=False)

    elif name == "ReduceLROnPlateau":
        lrs = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler)
    else:
        raise RuntimeError(f"No {name} lr_scheduler available")

    return lrs