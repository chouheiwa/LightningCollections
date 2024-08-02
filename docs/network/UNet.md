# UNet
| Code Repository |                  Paper                   | Pretrained Model |
|:---------------:|:----------------------------------------:|:----------------:|
|       N/A       | [Link](https://arxiv.org/abs/1505.04597) |       N/A        |

**UNet** is an old network for image segmentation. It is widely used in medical image segmentation.

## Example Usage
You can check the example script for [train](../../example_script/unet_train.sh) and [test](../../example_script/unet_test.sh) in the `example_script` folder.

## Network Metrics

### BUSI

|    Dataset     | AUROC | Accuracy | AveragePrecision | F1Score | JaccardIndex | Precision | Recall | Specificity | Dice | Best Model Link |
|:--------------:|:-----:|:--------:|:----------------:|:-------:|:------------:|:---------:|:------:|:-----------:|:----:|:---------------:|
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
