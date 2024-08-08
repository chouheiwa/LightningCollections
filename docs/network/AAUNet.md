# AAUNet

|             Code Repository              |                  Paper                   | Pretrained Model |
|:----------------------------------------:|:----------------------------------------:|:----------------:|
| [Link](https://github.com/CGPxy/AAU-net) | [Link](https://arxiv.org/abs/2204.12077) |       N/A        |

## Network Description

The original code is written by `tensorflow`. We have rewritten it by `pytorch`. You can simply copy the `AAUNet.py` to
your project and use it.

The original code not implement the full network, so we used UNet as the backbone and change the double conv component to their **HAAM** component.  

## Network Usage Description

1. `--classes` is the number of classes in the dataset. For simple image segmentation, it is 2. For multi-classes, it is
   the number of classes.
2. `--drop_last True` AAUNet must use the `drop_last` to be `True` to make the network work, because in the `HAAM` module we have a `BatchNorm1d` layer that need the input batch size to be at least 2, but in some cases the last batch size is 1, so we need to drop it.
3. `--batch_size` for us, we set it to 8, and when in training process, we used almost **20GB** of GPU memory. So you need to adjust the batch size according to your GPU memory. But it must be at least 2.
## Usage Script

You can check the example script for [train](../../example_script/aaunet_train.sh)
and [test](../../example_script/aaunet_test.sh) in the `example_script` folder.

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
