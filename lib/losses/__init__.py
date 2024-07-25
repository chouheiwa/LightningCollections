def get_loss_function(opt):
    if opt["loss_function_name"] == "DiceLoss":
        from .DiceLoss import DiceLoss
        return DiceLoss(opt.classes, weight=opt.loss_function_config.class_weight,
                                 sigmoid_normalization=False, mode=opt.loss_function_config.dice_loss_mode)

    if opt["loss_function_name"] == "LGANetStructureLoss":
        from .LGANetStructureLoss import LGANetStructureLoss
        return LGANetStructureLoss(opt.classes, need_expand=opt.loss_function_config.need_expand)

    if opt.loss_function_name == "CELoss":
        from .CELoss import CELoss
        return CELoss(opt.classes, **opt.loss_function_config.get_dict())

    else:
        raise RuntimeError(f"No {opt['loss_function_name']} is available")
