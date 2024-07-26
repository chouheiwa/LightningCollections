#!/bin/bash
dataset_name=${1:-"BUSI_all"}

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
--run_dir /home/chouheiwa/machine_learning/models/runs