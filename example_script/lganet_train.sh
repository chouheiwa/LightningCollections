dataset_name="BUSI_all"
python train.py \
--config configs/BUSI.yaml \
--dataset_name "${dataset_name}" \
--dataset_path /home/chouheiwa/machine_learning/dataset/BUSI数据集/"${dataset_name}" \
--model_name LGANet \
--classes 2 \
--image_size 256 \
--pretrain_weight_path /home/chouheiwa/machine_learning/pretrained_models/pvt_v2_b2.pth \
--optimizer_config_path configs/optimizer_configs/adam.yaml \
--lr_scheduler_config_path configs/lr_scheduler_configs/ReduceLROnPlateau.yaml \
--run_dir /home/chouheiwa/machine_learning/models/runs