## Setup

### Requirements

- Python 3.10
- Pytorch >= 2.0
- lightning

#### Linux

We highly propose to use linux system such as ubuntu. You can simply run the following command to install the
requirements:

```shell
# For the pip install
sh setup/setup.sh 
# Run the base_script.sh to install the other requirements
sh setup/base_script.sh
``` 

```shell
# For the conda install
sh setup/conda_setup.sh

sh setup/base_script.sh
```

#### Windows

You can also open the `setup/setup.sh` or `setup/conda_setup.sh` and `setup/base_script.sh` and then copy the command to
the **cmd** to install the python requirements.

### Pretrained Models
#### LGANet
You can download the pretrained model from [here](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV).
#### SwinUnet
You can download the pretrained model from [here](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing).

### Data Preparation

#### Common Simple Image Segmentation Dataset
1. You need to prepare the dataset in the following structure:
```
└── train
│   ├── images
│   └── masks
└── val
    ├── images
    └── masks
```
#### BUSI Dataset

The original dataset is in
the [BUSI dataset](https://academictorrents.com/details/1f0b5b8b9d3f6f1b3e8f4baf1b7e3f3b6f3b7f1b) and you can download
it from the link. After downloading the dataset, you need to unzip the dataset. The dataset structure is as follows:

```
├── BUSI_all
│   ├── train
│   │   ├── images
│   │   └── masks
│   └── val
│       ├── images
│       └── masks
├── BUSI_bad
│   ├── train
│   │   ├── images
│   │   └── masks
│   └── val
│       ├── images
│       └── masks
├── BUSI_benign
│   ├── train
│   │   ├── images
│   │   └── masks
│   └── val
│       ├── images
│       └── masks
└── BUSI_maligant
    ├── train
    │   ├── images
    │   └── masks
    └── val
        ├── images
        └── masks
```

## Run

### Training

#### [PMFSNet](https://github.com/yykzjh/PMFSNet)

Training on BUSI dataset with PMFSNet model and DiceLoss loss function, you can run the following command:

```shell
python train.py \
--config configs/BUSI.yaml \
--dataset_name BUSI_all \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/BUSI_all \
--model_name PMFSNet \
--model_config_path ./configs/model_configs/PMFSNet.yaml \
--loss_function_name DiceLoss \
--loss_function_config_path configs/loss_configs/DICE.yaml \
--run_dir ./runs
```

Note:

1. You need to config `--dataset_path` to the real path in your machine.
2. `--srun_dir` is the directory to save the training logs and checkpoints, and you need to make sure when in test
   script, the `run_dir` is the same as the training script.
3. All the params except `--config` are optional, and you can config them in the config yaml file.
4. The params with the suffix `config_path` will be load with yaml and the params without the suffix add to the param,
   so if you don't want to use the yaml config path, you can just add them in your config file.

#### LGANet

Training on BUSI dataset with LGANet model doesn't need to config the loss function, you can run the following command:

```shell
python train.py \
--config configs/BUSI.yaml \
--dataset_name BUSI_all \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/BUSI_all \
--model_name LGANet \
--classes 1 \
--image_size 256 \
--pretrain_weight_path /home/chouheiwa/machine_learning/pretrained_models/pvt_v2_b2.pth
```

Note:

1. The `--classes` for image segmentation need to be 1, due to the train/validation/test compute loss logic.
2. The `--image_size` is the input image size need to be 256, so you need to resize the image to 256x256, or else you need to change the model structure.

#### [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet)
Training on BUSI dataset with SwinUnet model, you can run the following command:

```shell
python train.py \
--config configs/BUSI.yaml \
--dataset_name "${dataset_name}" \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/"${dataset_name}" \
--model_name SwinUnet \
--model_config_path configs/model_configs/swin_tiny_patch4_window7_224_lite.yaml \
--loss_function_name DiceLoss \
--loss_function_config_path configs/loss_configs/DICE.yaml \
--pretrain_weight_path /home/chouheiwa/machine_learning/pretrained_models/swin_tiny_patch4_window7_224.pth \
--optimizer_config_path configs/optimizer_configs/adam.yaml \
--lr_scheduler_config_path configs/lr_scheduler_configs/ReduceLROnPlateau.yaml \
--classes 2 \
--image_size 224 \
--run_dir ./runs
```

Note:

1. The `--image_size` is the input image size need to be 224, so you need to resize the image to 224x224, or else you need to change the model structure.

### Testing

#### PMFSNet

For example testing on BUSI dataset with PMFSNet model, you can run the following command:

```shell
python test.py \
--config configs/BUSI.yaml \
--dataset_name BUSI_all \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/BUSI_all \
--model_name PMFSNet \
--model_config_path ./configs/model_configs/PMFSNet.yaml \
--loss_function_name DiceLoss \
--loss_function_config_path configs/loss_configs/DICE.yaml \
--run_dir ./runs \
--result_dir ./results
```

#### LGANet

Testing on BUSI dataset with LGANet model doesn't need to config the loss function, you can run the following command:

```shell
python test.py \
--config configs/BUSI.yaml \
--dataset_name BUSI_all \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/BUSI_all \
--model_name LGANet \
--classes 1 \
--pretrain_weight_path /home/chouheiwa/machine_learning/pretrained_models/pvt_v2_b2.pth \
--run_dir ./runs \
--result_dir ./results
```