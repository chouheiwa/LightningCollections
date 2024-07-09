# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/29 01:02
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
from .transforms.two import *
from .image_loader import ImageLoader


class SimpleDataset(ImageLoader):
    """
    load ISIC 2018 dataset
    """

    def __init__(self, opt, mode):
        """
        initialize ISIC 2018 dataset
        :param opt: params dict
        :param mode: train/valid
        """
        if mode == "train":
            array = []
            if opt.use_image_transform:
                array.append(RandomResizedCrop(tuple(opt["resize_shape"]), scale=(0.4, 1.0),
                                                             ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'))
                array.append(ColorJitter(brightness=opt["color_jitter"], contrast=opt["color_jitter"],
                                                       saturation=opt["color_jitter"], hue=0))
                array.append(RandomGaussianNoise(p=opt["augmentation_p"]))
                array.append(RandomHorizontalFlip(p=opt["augmentation_p"]))
                array.append(RandomVerticalFlip(p=opt["augmentation_p"]))
                array.append(RandomRotation(opt["random_rotation_angle"]))
                array.append(Cutout(p=opt["augmentation_p"], value=(0, 0)))
            else:
                array.append(Resize(opt["resize_shape"]))

            array.append(ToTensor())
            array.append(Normalize(mean=opt["normalize_means"], std=opt["normalize_stds"]))

            trans = Compose(array)
        else:
            trans = Compose([
                Resize(opt["resize_shape"]),
                ToTensor(),
                Normalize(mean=opt["normalize_means"], std=opt["normalize_stds"])
            ])

        base_path = os.path.join(opt["dataset_path"], "train" if mode == "train" else "val")

        super(SimpleDataset, self).__init__(
            origin_image_path=os.path.join(base_path, "images"),
            gt_image_path=os.path.join(base_path, "masks"),
            mode=mode,
            transforms=trans,
            support_types=['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 'TIF', 'TIFF'],
            gt_format=opt["gt_format"]
        )
