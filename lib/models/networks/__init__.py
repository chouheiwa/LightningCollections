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
            raise RuntimeError(f"No {opt['model_name']} model available when initialize model")

    raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize model")