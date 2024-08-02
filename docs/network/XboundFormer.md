# XboundFormer

|                  Code Repository                  |                  Paper                   | Pretrained Model |
|:-------------------------------------------------:|:----------------------------------------:|:----------------:|
| [Link](https://github.com/jcwang123/xboundformer) | [Link](https://arxiv.org/abs/2206.00806) |       N/A        |

XboundFormer is not provide the pretrained model. But they use pvt_v2_b2 as the backbone of the network. You can use the
pretrained model of `pvt_v2_b2` to initialize the network. And [PMFSNet](PMFSNet.md) also provide the pretrained model
of `pvt_v2_b2`, so you can use it as the pretrain backbone model.

## Example Usage

You can check the example script for [train](../../example_script/XboundFormer_train.sh)
and [test](../../example_script/XboundFormer_test.sh) in the `example_script` folder.

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
