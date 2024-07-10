python train.py \
--config configs/BUSI.yaml \
--dataset_name BUSI_all \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/BUSI_all \
--model_name LGANet \
--classes 1 \
--pretrain_weight_path /home/chouheiwa/machine_learning/pretrained_models/pvt_v2_b2.pth \
--optimizer_config_path configs/optimizer_configs/adam.yaml \
--lr_scheduler_config_path configs/lr_scheduler_configs/ReduceLROnPlateau.yaml \
--run_dir ./runs