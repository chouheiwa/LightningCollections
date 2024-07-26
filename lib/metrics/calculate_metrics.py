import torch
from torchmetrics import Recall

from .official_metrics_torch import cal_jaccard_index, cal_dsc, cal_accuracy


class CalculateMetrics:
    def __init__(self, prefix):
        self.dsc = []
        self.iou = []
        self.acc = []
        self.ji = []
        self.asd = []
        self.recall = []
        self.precision = []
        self.sensitivity = []
        self.specificity = []
        self.prefix = prefix


    def update(self, predict: torch.Tensor, target: torch.Tensor):
        self.dsc.append(cal_dsc(predict, target))
        self.iou.append(cal_jaccard_index(predict, target))
        self.acc.append(cal_accuracy(predict, target))
        self.ji.append(cal_jaccard_index(predict, target))
        self.recall.append()