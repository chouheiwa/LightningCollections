from torch import optim


def get_optimizer(opt, net):
    name = opt.optimizer_config.name
    optimizer_config = opt.optimizer_config.optimizer.get_dict()
    if name == "SGD":
        optimizer = optim.SGD(net.parameters(), **optimizer_config)

    elif name == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), **optimizer_config)

    elif name == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), **optimizer_config)

    elif name == "Adam":
        optimizer = optim.Adam(net.parameters(), **optimizer_config)

    elif name == "AdamW":
        optimizer = optim.AdamW(net.parameters(), **optimizer_config)

    elif name == "Adamax":
        optimizer = optim.Adamax(net.parameters(), **optimizer_config)

    elif name == "Adadelta":
        optimizer = optim.Adadelta(net.parameters(), **optimizer_config)
    else:
        raise RuntimeError(f"No {opt['optimizer_name']} optimizer available")

    return optimizer
