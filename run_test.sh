#!/bin/bash

all_models_script=("aaunet_test.sh")
all_data_array=("all" "benign" "malignant" "normal")
all_data_array=("bad")

for model_script in "${all_models_script[@]}"; do
    for data_type in "${all_data_array[@]}"; do
        dataset_name="BUSI_${data_type}"
        echo "Running ${model_script} with ${dataset_name} data"
        example_script/${model_script} "${dataset_name}"
    done
done