#!/bin/bash
dataset_name=${1:-"BUSI_all"}
dataset_path=${2:-"/home/chouheiwa/machine_learning/dataset/BUSI数据集/${dataset_name}"}
extra_params=${3:-"--run_dir /home/chouheiwa/machine_learning/models/runs"}

python train.py \
--config configs/Common.yaml \
--dataset_name "${dataset_name}" \
--dataset_path "${dataset_path}" \
--model_name PMFSNet \
--image_size 224 \
--model_config_path ./configs/model_configs/PMFSNet.yaml \
--loss_function_name DiceLoss \
--loss_function_config_path configs/loss_configs/DICE.yaml \
${extra_params}