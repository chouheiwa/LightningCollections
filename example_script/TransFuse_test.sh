#!/bin/bash
dataset_name=${1:-"BUSI_all"}

python test.py \
--config configs/BUSI.yaml \
--dataset_name "${dataset_name}" \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/"${dataset_name}" \
--model_name TransFuse \
--model_config_path configs/model_configs/TransFuse.yaml \
--classes 2 \
--image_size 224 \
--optimizer_config_path configs/optimizer_configs/adam.yaml \
--lr_scheduler_config_path configs/lr_scheduler_configs/ReduceLROnPlateau.yaml \
--run_dir /home/chouheiwa/machine_learning/models/runs