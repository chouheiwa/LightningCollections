import torch
import torch.nn as nn
import lightning as L

from lib.utils import expand_as_one_hot


class CELoss(L.LightningModule):
    def __init__(self, num_classes, class_weight):
        assert (class_weight is not None, "class_weight is required for CELoss")
        super(CELoss, self).__init__()
        self.register_buffer('class_weight', torch.FloatTensor(class_weight), persistent=False)
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss(weight=self.class_weight)

    def forward(self, input, target):
        target = expand_as_one_hot(target.long(), self.num_classes)
        return self.loss(input, target)
