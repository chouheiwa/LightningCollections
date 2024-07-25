#!/bin/bash

all_data_array=("all" "bad" "benign" "malignant")
#all_data_array=("malignant")


for data_type in "${all_data_array[@]}"; do
dataset_name="BUSI_${data_type}"
python train.py \
--config configs/BUSI.yaml \
--dataset_name "${dataset_name}" \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/"${dataset_name}" \
--model_name NUNet \
--classes 2 \
--image_size 256 \
--batch_size 12 \
--end_epoch 1000 \
--loss_function_name DiceLoss \
--loss_function_config_path configs/loss_configs/DICE.yaml \
--optimizer_config_path configs/optimizer_configs/NUNet/adam.yaml \
--run_dir /home/chouheiwa/machine_learning/models/runs
done