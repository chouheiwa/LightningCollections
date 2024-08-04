from torch.optim import lr_scheduler

def get_lr_scheduler(optimizer, config):
    if config.lr_scheduler_config is None:
        return None

    name = config.lr_scheduler_config.name
    scheduler = config.lr_scheduler_config.scheduler
    scheduler = scheduler.get_dict() if scheduler is not None else {}

    if name == "ExponentialLR":
        lrs = lr_scheduler.ExponentialLR(optimizer, **scheduler)

    elif name == "StepLR":
        lrs = lr_scheduler.StepLR(optimizer, **scheduler)

    elif name == "MultiStepLR":
        lrs = lr_scheduler.MultiStepLR(optimizer, **scheduler)

    elif name == "CosineAnnealingLR":
        lrs = lr_scheduler.CosineAnnealingLR(optimizer, **scheduler)

    elif name == "CosineAnnealingWarmRestarts":
        lrs = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler)

    elif name == "OneCycleLR":
        lrs = lr_scheduler.OneCycleLR(optimizer, **scheduler)

    elif name == "ReduceLROnPlateau":
        lrs = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler)
    else:
        raise RuntimeError(f"No {name} lr_scheduler available")

    return lrs