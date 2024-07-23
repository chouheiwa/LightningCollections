#!/bin/bash

all_data_array=("all" "bad" "benign" "malignant")
#all_data_array=("malignant")


#for data_type in "${all_data_array[@]}"; do
#dataset_name="BUSI_${data_type}"
#python test.py \
#--config configs/BUSI.yaml \
#--dataset_name "${dataset_name}" \
#--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/"${dataset_name}" \
#--model_name LGANet \
#--classes 1 \
#--image_size 256 \
#--pretrain_weight_path /home/chouheiwa/machine_learning/pretrained_models/pvt_v2_b2.pth \
#--run_dir ./runs \
#--result_dir ./results
#done

for data_type in "${all_data_array[@]}"; do
dataset_name="BUSI_${data_type}"
python test.py \
--config configs/BUSI.yaml \
--dataset_name "${dataset_name}" \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/"${dataset_name}" \
--model_name NUNet \
--classes 2 \
--loss_function_name DiceLoss \
--loss_function_config_path configs/loss_configs/DICE.yaml \
--image_size 256 \
--run_dir /home/chouheiwa/machine_learning/models/runs \
--result_dir /home/chouheiwa/machine_learning/models/results
done