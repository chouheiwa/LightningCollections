from torch import optim


def get_optimizer(opt, net):
    if opt["optimizer_name"] == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=opt["learning_rate"], momentum=opt["momentum"],
                              weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=opt["learning_rate"],
                                  weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=opt["learning_rate"],
                                  weight_decay=opt["weight_decay"],
                                  momentum=opt["momentum"])

    elif opt["optimizer_name"] == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=opt["learning_rate"],
                               weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=opt["learning_rate"],
                                weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "Adamax":
        optimizer = optim.Adamax(net.parameters(), lr=opt["learning_rate"],
                                 weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "Adadelta":
        optimizer = optim.Adadelta(net.parameters(), lr=opt["learning_rate"],
                                   weight_decay=opt["weight_decay"])
    else:
        raise RuntimeError(f"No {opt['optimizer_name']} optimizer available")

    return optimizer