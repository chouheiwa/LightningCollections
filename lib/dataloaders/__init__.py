

from torch.utils.data import DataLoader

from .simple_dataset import SimpleDataset


def get_data_loader(opt):
    """
    get dataloader
    Args:
        opt: params dict
    Returns:
    """
    if "ISIC" in opt["dataset_name"] or "BUSI" in opt["dataset_name"]:
        train_set = SimpleDataset(opt, mode="train")
        valid_set = SimpleDataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True, drop_last=opt.drop_last)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    opt["steps_per_epoch"] = len(train_loader)

    return train_loader, valid_loader


def get_test_data_loader(opt):
    """
    get test dataloader
    :param opt: params dict
    :return:
    """
    if "ISIC" in opt["dataset_name"] or "BUSI" in opt["dataset_name"]:
        valid_set = SimpleDataset(opt, mode="valid", auto_append=not opt.forbid_auto_append, need_metrics=not opt.forbid_metrics)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    return valid_loader