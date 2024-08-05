# LGANet

|              Code Repository              |                                    Paper                                    |                                                                               Pretrained Model                                                                               |
|:-----------------------------------------:|:---------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [Link](https://github.com/SZUcsh/FAT-Net) | [Link](https://www.sciencedirect.com/science/article/pii/S1361841521003728) | [deit_model](https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth)<br/>[resnet_model](https://download.pytorch.org/models/resnet34-b627a593.pth) |

## Network Description

The original code get deit_model used by `torch.hub.load` and resnet_model used by `torchvision.models.resnet34`. We
have added loaded the pretrained model in the network local, so you can use the network with the pretrained model just
by downloading it by yourselves.

## Network Usage Description

1. `--classes` is the number of classes in the dataset. For simple image segmentation, it is 2. For multi-classes, it is
   the number of classes.
2. `--image_size` is the size of the input image. It must be 224.

## Example Usage

You can check the example script for [train](../../example_script/fatnet_train.sh)
and [test](../../example_script/fatnet_test.sh) in the `example_script` folder.

## Network Metrics

## BUSI

| Dataset | AUROC | Accuracy | AveragePrecision | F1Score | JaccardIndex | Precision | Recall | Specificity | Dice | Best Model Link |
|:-------:|:-----:|:--------:|:----------------:|:-------:|:------------:|:---------:|:------:|:-----------:|:----:|:---------------:|

## ISIC

| Dataset | AUROC | Accuracy | AveragePrecision | F1Score | JaccardIndex | Precision | Recall | Specificity | Dice | Best Model Link |
|:-------:|:-----:|:--------:|:----------------:|:-------:|:------------:|:---------:|:------:|:-----------:|:----:|:---------------:|