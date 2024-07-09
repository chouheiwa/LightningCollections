import torch

from .DiceLoss import DiceLoss


def get_loss_function(opt):
    if opt["loss_function_name"] == "DiceLoss":
        loss_function = DiceLoss(opt.classes, weight=opt.loss_function_config.class_weight,
                                 sigmoid_normalization=False, mode=opt.loss_function_config.dice_loss_mode)

    else:
        raise RuntimeError(f"No {opt['loss_function_name']} is available")

    return loss_function
