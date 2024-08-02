# LGANet

|              Code Repository              |                        Paper                         |                                 Pretrained Model                                 |
|:-----------------------------------------:|:----------------------------------------------------:|:--------------------------------------------------------------------------------:|
| [Link](https://github.com/AHU-VRV/LGANet) | [Link](https://ieeexplore.ieee.org/document/9527678) | [Link](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV) |

## Network Usage Description

The `--image_size` is the input image size need to be 256, so you need to resize the image to 256x256, or else you need
to change the model structure.

## Usage Script

You can check the example script for [train](../../example_script/lganet_train.sh)
and [test](../../example_script/lganet_test.sh) in the `example_script` folder.

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
