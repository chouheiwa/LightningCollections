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
--loss_function DiceLoss \
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
--pretrain_weight_path /home/chouheiwa/machine_learning/pretrained_models/pvt_v2_b2.pth
```

Note:
1. The `--classes` for image segmentation need to be 1, due to the train/validation/test compute loss logic.

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
--loss_function DiceLoss \
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