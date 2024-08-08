# MALUNet

|               Code Repository                |                  Paper                   | Pretrained Model |
|:--------------------------------------------:|:----------------------------------------:|:----------------:|
| [Link](https://github.com/JCruan519/MALUNet) | [Link](https://arxiv.org/abs/2211.01784) |       N/A        |

MALUNet is mainly workable for ISIC dataset. This network performance very bad on BUSI dataset.

We rewrite the `SpatialAttBridge` to make it accept dynamic input tensors, the origin code only accept 5 input tensors.

## Network Metrics

### BUSI

|    Dataset     | AUROC | Accuracy | AveragePrecision | F1Score | JaccardIndex | Precision | Recall | Specificity | Dice |
|:--------------:|:-----:|:--------:|:----------------:|:-------:|:------------:|:---------:|:------:|:-----------:|:----:|
|    BUSI_all    |
|    BUSI_bad    |
|  BUSI_benign   |
| BUSI_malignant |

### ISIC

|  Dataset  | AUROC | Accuracy | AveragePrecision | F1Score | JaccardIndex | Precision | Recall | Specificity | Dice | Best Model Link |
|:---------:|:-----:|:--------:|:----------------:|:-------:|:------------:|:---------:|:------:|:-----------:|:----:|:---------------:|
| ISIC-2018 |
| ISIC-2017 |
| ISIC-2016 |
