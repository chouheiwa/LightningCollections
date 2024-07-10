python train.py \
--config configs/BUSI.yaml \
--dataset_name BUSI_all \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/BUSI_all \
--model_name PMFSNet \
--model_config_path ./configs/model_configs/PMFSNet.yaml \
--loss_function DiceLoss \
--loss_function_config_path configs/loss_configs/DICE.yaml \
--run_dir ./runs