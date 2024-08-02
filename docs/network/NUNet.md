# NU-Net
|             Code Repository             |                  Paper                   | Pretrained Model |
|:---------------------------------------:|:----------------------------------------:|:----------------:|
| [Link](https://github.com/CGPxy/NU-net) | [Link](https://arxiv.org/abs/2209.07193) |       N/A        |

## Network Description
The original code is written by `tensorflow`. We have rewritten it by `pytorch`. You can simply copy the `NUNet.py` to your project and use it.

We remove the original out `sigmoid Activation` and add `num_classes` parameters to help the network can make segmentation for multi-classes.  

## Network Usage Description
1. `--classes` is the number of classes in the dataset. For simple image segmentation, it is 2. For multi-classes, it is the number of classes.
2. `--image_size` is the size of the input image. The default value is 256. And the image size should be the multiple of **128**.

## Usage Script
You can check the example script for [train](../../example_script/nunet_train.sh) and [test](../../example_script/nunet_test.sh) in the `example_script` folder.

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
