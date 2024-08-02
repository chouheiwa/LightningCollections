# Use local imports to make that you can minimize install dependencies.
def get_network_model(opt, isTrain=True):
    simple_image_segmentation_datasets = ["ISIC", "BUSI"]
    for dataset in simple_image_segmentation_datasets:
        if dataset in opt["dataset_name"]:
            if opt.model_name == "UNet":
                from .UNet import UNet
                return UNet(n_channels=3, n_classes=opt.classes)
            if opt.model_name == "AttUNet":
                from .AttUNet import AttUNet
                return AttUNet(img_ch=3, output_ch=opt.classes)
            if opt.model_name == "PMFSNet":
                from .PMFSNet import PMFSNet
                return PMFSNet(opt)
            if opt.model_name == "LGANet":
                from .LGANet import LGANet
                return LGANet(channel=32, n_classes=opt.classes, pretrain_model_path=opt.pretrain_weight_path)
            if opt.model_name == "TransFuse":
                from .TransFuse import TransFuse
                return TransFuse(
                    num_classes=opt.classes,
                    image_size=opt.resize_shape,
                    pretrained_model_path=opt.pretrain_weight_path,
                    **opt.model_config.get_dict()
                )
            if opt.model_name == "FATNet":
                from .FATNet import FATNet
                return FATNet(n_classes=opt.classes)
            if opt.model_name == "SwinUnet":
                from .SwinUnet import SwinUnet
                return SwinUnet(
                    config=opt,
                    img_size=opt.resize_shape[0],
                    in_channels=opt.in_channels,
                    num_classes=opt.classes,
                    pretrained_path=opt.pretrain_weight_path
                )
            if opt.model_name == "NUNet":
                from .NUNet import NUNet
                return NUNet(num_classes=opt.classes)
            if opt.model_name == "AAUNet":
                from .AAUNet import AAUNet
                return AAUNet(n_channels=3, n_classes=opt.classes)
            if opt.model_name == "XboundFormer":
                from .XboundFormer import XboundFormer
                return XboundFormer(
                    n_classes=opt.classes,
                    image_size=opt.resize_shape[0],
                    pretrained_model_path=opt.pretrain_weight_path,
                    **opt.model_config.get_dict()
                )
            if opt.model_name == "MALUNet":
                from .MALUNet import MALUNet
                return MALUNet(num_classes=opt.classes, input_channels=3)
            raise RuntimeError(f"No {opt['model_name']} model available when initialize model")

    raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize model")
