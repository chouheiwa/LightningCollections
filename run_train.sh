#!/bin/bash

all_models_script=("malunet_train.sh")
all_data_array=("all" "bad" "benign" "malignant")
#all_data_array=("bad")
#all_data_array=("malignant")

for model_script in "${all_models_script[@]}"; do
    for data_type in "${all_data_array[@]}"; do
        echo "Running ${model_script} with BUSI_${data_type} data"
        example_script/${model_script} "BUSI_${data_type}" "--need_early_stop True"
    done
done