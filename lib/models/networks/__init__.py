# Use local imports to make that you can minimize install dependencies.
def get_network_model(opt, isTrain=True):
    simple_image_segmentation_datasets = ["ISIC", "BUSI"]
    for dataset in simple_image_segmentation_datasets:
        if dataset in opt["dataset_name"]:
            if opt["model_name"] == "PMFSNet":
                from .PMFSNet import PMFSNet
                return PMFSNet(opt)
            if opt.model_name == "LGANet":
                from .LGANet import LGANet
                return LGANet(channel=32, n_classes=opt.classes, pretrain_model_path=opt.pretrain_weight_path)
            if opt.model_name == "FATNet":
                from .FATNet import FATNet
                return FATNet(n_classes=opt.classes)
            if opt.model_name == "FITNet":
                from .FITNet import FITNet
                return FITNet(n_classes=opt.classes)
            if opt.model_name == "SwinUnet":
                from .SwinUnet import SwinUnet
                return SwinUnet(
                    config=opt,
                    img_size=opt.resize_shape[0],
                    in_channels=opt.in_channels,
                    num_classes=opt.classes,
                    pretrained_path=opt.pretrain_weight_path
                )
            raise RuntimeError(f"No {opt['model_name']} model available when initialize model")

    raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize model")
