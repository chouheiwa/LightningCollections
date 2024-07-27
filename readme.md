## Introduction

Our project is a simple image segmentation project that we aim to implement more segmentation network that to help us to
write out essay easier.

We used pytorch lightning to implement the project, and we also provide the pretrained model for the network that we
implemented.

## Setup

### Basic Requirements

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

#### SwinUnet

You can download the pretrained model
from [here](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing).

### Dataset Preparation

Now we support the following datasets:

- [x] [BUSI](docs/dataset/BUSI.md)

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

## Run

### How to add a new network

#### Simple Steps
If your network is a simple network like `UNet`, then just follow the following steps:

1. Create a new python package in the `lib/models/networks` folder (if your network is just a simple pytorch file, just
   copy it to the folder).
2. Change your network base class to `LightningModule`.
3. *\[Optional\]* Create a new config **yaml** file in the `configs/model_configs` folder.
4. Edit the `__init__.py` file in the `lib/models/networks` folder to import the new network.
5. Create a new script file in the `example_script` folder to train the new network.

Note:
1. The `--model_config_path` is the path to the config file that you created in step 2. And it will finally convert to
the `opt.model_config` in the step 3 `__init__.py` file.
2. If your network need to use the pretrained model, you can add the `--pretrained_model_path` to the script file.

#### Complex Steps
If your network is a complex network like `LGANet`, you need to follow the following steps:

1. Create a new python package in the `lib/models/networks` folder (if your network is just a simple pytorch file, just
   copy it to the folder).
2. Change your network base class to `LightningModule`, and then rewrite the method `training_step`, `validation_step`,
   `test_step`, note that `training_step`, `validation_step` need to return the final `loss`, `test_step` need to return the predict result.
3. *\[Optional\]* Create a new config **yaml** file in the `configs/model_configs` folder.
4. Edit the `__init__.py` file in the `lib/models/networks` folder to import the new network.
5. Create a new script file in the `example_script` folder to train the new network.

### Network

Now we implement the following networks(you can click the name to see the detail information):

- [x] [UNet](docs/network/UNet.md)
- [x] [PMFSNet](docs/network/PMFSNet.md)
- [x] [NUNet](docs/network/NUNet.md)
- [x] [TransFuse](docs/network/TransFuse.md)
- [x] [LGANet](docs/network/LGANet.md)
- [x] [SwinUnet](docs/network/SwinUnet.md)