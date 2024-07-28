#!/bin/bash
dataset_name=${1:-"BUSI_all"}

python train.py \
--config configs/BUSI.yaml \
--dataset_name "${dataset_name}" \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/"${dataset_name}" \
--model_name XboundFormer \
--model_config_path configs/model_configs/XboundFormer.yaml \
--classes 2 \
--image_size 224 \
--run_dir ./runs \
--result_dir /home/chouheiwa/machine_learning/models/results