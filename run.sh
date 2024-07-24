#!/bin/bash

all_data_array=("all" "bad" "benign" "malignant")
#all_data_array=("malignant")


for data_type in "${all_data_array[@]}"; do
dataset_name="BUSI_${data_type}"
python train.py \
--config configs/BUSI.yaml \
--dataset_name "${dataset_name}" \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/"${dataset_name}" \
--model_name TransFuse \
--model_config_path configs/model_configs/TransFuse.yaml \
--classes 2 \
--image_size 224 \
--pretrain_weight_path /home/chouheiwa/machine_learning/pretrained_models \
--optimizer_config_path configs/optimizer_configs/adam.yaml \
--lr_scheduler_config_path configs/lr_scheduler_configs/ReduceLROnPlateau.yaml \
--run_dir /home/chouheiwa/machine_learning/models/runs
done