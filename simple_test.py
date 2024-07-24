import torch

from lib.losses.LGANetStructureLoss import LGANetStructureLoss
from lib.models.networks.TransFuse import TransFuse_S
from lib.utils import expand_as_one_hot

if __name__ == '__main__':
    model = LGANetStructureLoss(classes=2)
    data = torch.tensor([[1, 1, 1, 1],[1, 1, 1, 1]])
    data = data.unsqueeze(0)
    data = expand_as_one_hot(data, 2)
    model.forward(torch.rand(1, 2, 4, 4), torch.rand(1, 4, 4))