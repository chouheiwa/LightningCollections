import csv
import os

import torch
from torchmetrics import *
from thop import profile

class MetricsResult:
    def __init__(self, result, prefix, mission="Binary"):
        self.F1 = result[f"{prefix}/{mission}F1Score"].item()
        self.Accuracy = result[f"{prefix}/{mission}Accuracy"].item()
        self.Dice = result[f"{prefix}/Dice"].item()
        self.Precision = result[f"{prefix}/{mission}Precision"].item()
        self.Specificity = result[f"{prefix}/{mission}Specificity"].item()
        self.Recall = result[f"{prefix}/{mission}Recall"].item()
        self.JaccardIndex = result[f"{prefix}/{mission}JaccardIndex"].item()
        try:
            self.AUROC = result[f"{prefix}/{mission}AUROC"].item()
            self.AveragePrecision = result[f"{prefix}/{mission}AveragePrecision"].item()
        except:
            self.AUROC = 0
            self.AveragePrecision = 0

    def to_result_csv(self, path, model_name, flops=0, params=0):
        first_create = os.path.exists(path)
        with open(os.path.join(path), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            if not first_create:
                wr.writerow([
                    'Model',
                    'Miou(Jaccard Similarity)', 'F1_score', 'Accuracy', 'Specificity',
                    'Sensitivity', 'DSC', 'Precision',
                    'AP', 'AUC', 'Parameters（M）', 'FLOPs(G)'
                ])

            wr.writerow([
                model_name,
                self.JaccardIndex * 100, self.F1 * 100, self.Accuracy * 100, self.Specificity * 100,
                self.Recall * 100, self.Dice * 100, self.Precision * 100,
                self.AveragePrecision * 100, self.AUROC * 100,
                params / 1e6, flops / 1e9
            ])

    def to_log(self, type, epoch, end_epoch, tr_loss):
        return f'Epoch [{epoch + 1}' \
               f'/{end_epoch}], Loss: {tr_loss:.4f}, ' \
               f'[{type}] Acc: {self.Accuracy:.4f}, ' \
               f'SE: {self.Recall:.4f}, ' \
               f'SP: {self.Specificity:.4f}, ' \
               f'PC: {self.Precision:.4f}, ' \
               f'F1: {self.F1:.4f}, ' \
               f'DC: {self.Dice:.4f}, ' \
               f'MIOU: {self.JaccardIndex:.4f}'

    def cal_params_flops(self, model, size):
        input = torch.randn(1, 3, size, size).cuda()
        flops, params = profile(model, inputs=(input,))
        return params, flops

def get_binary_metrics():
    return MetricCollection(
        [
            F1Score(task="binary"),
            Accuracy(task="binary"),
            Dice(multiclass=False),
            Precision(task="binary"),
            Specificity(task="binary"),
            Recall(task="binary"),
            AUROC(task="binary"),
            AveragePrecision(task="binary"),
            # IoU
            JaccardIndex(task="binary", num_labels=2, num_classes=2),
        ]
    )