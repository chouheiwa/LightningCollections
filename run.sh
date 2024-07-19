#!/bin/bash

all_data_array=("all" "bad" "benign" "malignant")
#all_data_array=("malignant")


for data_type in "${all_data_array[@]}"; do
dataset_name="BUSI_${data_type}"
python train.py \
--config configs/BUSI.yaml \
--dataset_name "${dataset_name}" \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/"${dataset_name}" \
--model_name FATNet \
--classes 2 \
--optimizer_config_path configs/optimizer_configs/adam.yaml \
--lr_scheduler_config_path configs/lr_scheduler_configs/ReduceLROnPlateau.yaml \
--loss_function_name DiceLoss \
--loss_function_config_path configs/loss_configs/DICE.yaml \
--image_size 224 \
--run_dir /home/chouheiwa/machine_learning/models/runs
done