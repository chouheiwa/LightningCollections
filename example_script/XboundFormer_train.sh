#!/bin/bash
dataset_name=${1:-"BUSI_all"}
dataset_path=${2:-"/home/chouheiwa/machine_learning/dataset/BUSI数据集/${dataset_name}"}
extra_params=${3:-"--run_dir /home/chouheiwa/machine_learning/models/runs"}

python train.py \
--config configs/Common.yaml \
--dataset_name "${dataset_name}" \
--dataset_path "${dataset_path}" \
--model_name XboundFormer \
--model_config_path configs/model_configs/XboundFormer.yaml \
--pretrain_weight_path /home/chouheiwa/machine_learning/pretrained_models/pvt_v2_b2.pth \
--classes 2 \
--image_size 224 \
--optimizer_config_path configs/optimizer_configs/adam.yaml \
--lr_scheduler_config_path configs/lr_scheduler_configs/ReduceLROnPlateau.yaml \
${extra_params}