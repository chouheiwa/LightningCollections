# DSEUNet

|              Code Repository              |                       Paper                        | Pretrained Model |
|:-----------------------------------------:|:--------------------------------------------------:|:----------------:|
| [Link](https://github.com/CGPxy/DSEU-net) | [Link](https://doi.org/10.1016/j.eswa.2023.119939) |       N/A        |

## Network Description

The original code is written by `tensorflow`. We have rewritten it by `pytorch`. You can simply copy the `DSEUNet.py` to
your project and use it.

## Network Usage Description

1. `--classes` is the number of classes in the dataset. For simple image segmentation, it is 2. For multi-classes, it is
   the number of classes.
2. `--image_size` is the size of the input image. The default value is 256. And the image size should be the multiple of
   **128**.

## Usage Script

You can check the example script for [train](../../example_script/dseunet_train.sh)
and [test](../../example_script/dseunet_test.sh) in the `example_script` folder.

## Network Metrics

### BUSI

|    Dataset     | AUROC | Accuracy | AveragePrecision | F1Score | JaccardIndex | Precision | Recall | Specificity | Dice |
|:--------------:|:-----:|:--------:|:----------------:|:-------:|:------------:|:---------:|:------:|:-----------:|:----:|
|    BUSI_all    |
|    BUSI_bad    |
|  BUSI_benign   |
| BUSI_malignant |

### ISIC

|  Dataset  | AUROC | Accuracy | AveragePrecision | F1Score | JaccardIndex | Precision | Recall | Specificity | Dice |
|:---------:|:-----:|:--------:|:----------------:|:-------:|:------------:|:---------:|:------:|:-----------:|:----:|
| ISIC-2018 |
| ISIC-2017 |
| ISIC-2016 |