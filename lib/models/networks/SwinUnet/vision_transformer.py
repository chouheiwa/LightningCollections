# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn
import lightning as L

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


class SwinUnet(L.LightningModule):
    def __init__(self, config, img_size=224, in_channels=3, num_classes=21843, zero_head=False, pretrained_path=None):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=config.model_config.SWIN.PATCH_SIZE,
            in_chans=in_channels,
            num_classes=self.num_classes,
            embed_dim=config.model_config.SWIN.EMBED_DIM,
            depths=config.model_config.SWIN.DEPTHS,
            num_heads=config.model_config.SWIN.NUM_HEADS,
            window_size=config.model_config.SWIN.WINDOW_SIZE,
            mlp_ratio=config.model_config.SWIN.MLP_RATIO,
            qkv_bias=config.model_config.SWIN.QKV_BIAS,
            qk_scale=config.model_config.SWIN.QK_SCALE,
            drop_rate=config.model_config.DROP_RATE,
            drop_path_rate=config.model_config.DROP_PATH_RATE,
            ape=config.model_config.SWIN.APE,
            patch_norm=config.model_config.SWIN.PATCH_NORM,
            use_checkpoint=False)
        self.load_from(pretrained_path=pretrained_path)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, pretrained_path):
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))

            pretrained_dict = torch.load(pretrained_path)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
