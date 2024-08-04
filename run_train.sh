#!/bin/bash

all_models_script=("unet_train.sh" "ours/conv_net_train.sh")
all_data_array=("all" "bad" "benign" "malignant")
#all_data_array=("bad")
#all_data_array=("malignant")

for model_script in "${all_models_script[@]}"; do
    for data_type in "${all_data_array[@]}"; do
        dataset_name="BUSI_${data_type}"
        dataset_path="/home/chouheiwa/machine_learning/dataset/BUSI数据集/${dataset_name}"
        echo "Running ${model_script} with BUSI_${data_type} data"
        example_script/${model_script} "BUSI_${data_type}" "${dataset_path}"
    done
done