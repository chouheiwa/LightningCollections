#!/bin/bash
dataset_name=${1:-"BUSI_all"}

python test.py \
--config configs/BUSI.yaml \
--dataset_name "${dataset_name}" \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/"${dataset_name}" \
--model_name LGANet \
--classes 2 \
--image_size 256 \
--run_dir /home/chouheiwa/machine_learning/models/runs \
--result_dir /home/chouheiwa/machine_learning/models/results