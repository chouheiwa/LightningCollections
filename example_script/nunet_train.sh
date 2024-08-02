#!/bin/bash
dataset_name=${1:-"BUSI_all"}
extra_params=${2:-""}

python train.py \
--config configs/BUSI.yaml \
--dataset_name "${dataset_name}" \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/"${dataset_name}" \
--model_name NUNet \
--classes 2 \
--image_size 256 \
--batch_size 12 \
--loss_function_name DiceLoss \
--loss_function_config_path configs/loss_configs/DICE.yaml \
--optimizer_config_path configs/optimizer_configs/NUNet/adam.yaml \
--run_dir /home/chouheiwa/machine_learning/models/runs \
${extra_params}