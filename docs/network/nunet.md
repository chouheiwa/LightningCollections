## Network Description
The original code repository is [here](https://github.com/CGPxy/NU-net). It is written by `tensorflow`. I have rewritten it by `pytorch`. You can simply copy the `NUNet.py` to your project and use it.

The user original paper is [here](https://arxiv.org/abs/2209.07193)

We remove the original out `sigmoid Activation` and add `num_classes` parameters to help the network can make segmentation for multi-classes.  

## Network Usage Description
1. `--classes` is the number of classes in the dataset. For simple image segmentation, it is 2. For multi-classes, it is the number of classes.
2. `--image_size` is the size of the input image. The default value is 256. And the image size should be the multiple of **128**.


## Usage Script
You can check the example script for [train](../../example_script/nunet_train.sh) and [test](../../example_script/nunet_test.sh) in the `example_script` folder.