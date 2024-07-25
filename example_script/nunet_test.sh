dataset_name="BUSI_all"
python test.py \
--config configs/BUSI.yaml \
--dataset_name "${dataset_name}" \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/"${dataset_name}" \
--model_name NUNet \
--classes 2 \
--loss_function_name CELoss \
--loss_function_config_path configs/loss_configs/CE.yaml \
--image_size 256 \
--run_dir /home/chouheiwa/machine_learning/models/runs \
--result_dir /home/chouheiwa/machine_learning/models/results