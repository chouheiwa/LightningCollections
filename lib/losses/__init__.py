def get_loss_function(opt):
    if opt["loss_function_name"] == "DiceLoss":
        from .DiceLoss import DiceLoss
        loss_function = DiceLoss(opt.classes, weight=opt.loss_function_config.class_weight,
                                 sigmoid_normalization=False, mode=opt.loss_function_config.dice_loss_mode)

    elif opt["loss_function_name"] == "LGANetStructureLoss":
        from .LGANetStructureLoss import LGANetStructureLoss
        loss_function = LGANetStructureLoss(opt.classes, need_expand=opt.loss_function_config.need_expand)

    else:
        raise RuntimeError(f"No {opt['loss_function_name']} is available")

    return loss_function
