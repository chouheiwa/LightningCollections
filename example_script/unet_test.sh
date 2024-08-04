#!/bin/bash
dataset_name=${1:-"BUSI_all"}
dataset_path=${2:-"/home/chouheiwa/machine_learning/dataset/BUSI数据集/${dataset_name}"}
extra_params=${3:-"--run_dir /home/chouheiwa/machine_learning/models/runs --result_dir /home/chouheiwa/machine_learning/models/results"}

python test.py \
--config configs/Common.yaml \
--dataset_name "${dataset_name}" \
--dataset_path "${dataset_path}" \
--model_name UNet \
--classes 2 \
--image_size 224 \
${extra_params}