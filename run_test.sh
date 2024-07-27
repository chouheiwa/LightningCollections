#!/bin/bash

all_models_script=("nunet_test.sh" "swinunet_test.sh" "fatnet_test.sh" "lganet_test.sh" "TransFuse_test.sh")
#all_data_array=("all" "bad" "benign" "malignant")
all_data_array=("normal")

for model_script in "${all_models_script[@]}"; do
    for data_type in "${all_data_array[@]}"; do
        dataset_name="BUSI_${data_type}"
        echo "Running ${model_script} with ${dataset_name} data"
        example_script/${model_script} "${dataset_name}"
    done
done