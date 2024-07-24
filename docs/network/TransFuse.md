# TransFuse
## Network Description
The original code repository is [here](https://github.com/Rayicer/TransFuse).

The user original paper is [here](https://arxiv.org/abs/2102.08005).

## Pretrained Models
| TransFuse_Name      | Resnet Model                                                           | DeiT Model                                                                              |
|---------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| **TransFuse_S**     | [resnet-34](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | [DeiT-small](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)   |
| **TransFuse_L**     | [resnet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth) | [DeiT-base](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)     |
| **TransFuse_L_384** | [resnet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth) | [DeiT-base-384](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth) |

By default, if you don't change the TransFuse model in [here](../../lib/models/networks/TransFuse/__init__.py) :`lib/models/networks/TransFuse/__init__.py`, we use the `TransFuse_S` model.

And you need to download the resnet model and DeiT model to the same folder that you config in the `--pretrain_weight_path`.

Note: When you download the model, you need to use the original name of the model.

If you want to get more DeiT model, you can find it at [here](https://github.com/facebookresearch/deit/blob/main/README_deit.md)

## Example Usage