#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import lightning as L
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lib.losses.LGANetStructureLoss import LGANetStructureLoss
from lib.utils import expand_as_one_hot
from .pvtv2 import pvt_v2_b2


class LGANet(L.LightningModule):
    def __init__(self, channel=32, n_classes=1, pretrain_model_path=None):
        super(LGANet, self).__init__()
        self.use_custom_loss_function = True
        self.criteria = nn.BCELoss()
        self.n_classes = n_classes
        self.loss_func = LGANetStructureLoss(classes=n_classes)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        if pretrain_model_path is not None:
            # 打印path所指代的绝对路径
            save_model = torch.load(pretrain_model_path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        # self.Translayer1_1 = nn.Sequential(BasicConv2d(64, channel, 3,padding=1),
        #                                    BasicConv2d(channel,channel,3,padding=1))
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.atten2 = LFM_GAM(128, n_classes, 4, 128, 1024)
        self.atten3 = LFM_GAM(320, n_classes, 4, 320, 256)
        self.atten4 = LFM_GAM(512, n_classes, 4, 512, 64)

        # self.gloabl2 = Attention_Global(128,8)
        # self.gloabl3 = Attention_Global(320,8)
        # self.gloabl4 = Attention_Global(512,8)

        self.aggeration = Three_Cat_Aggeration(channel=32)

        self.outconv = nn.Conv2d(channel, n_classes, 1)

    def forward(self, x):
        _x, _attns = self.backbone(x)

        x1 = _x[0]  # (_,64,h/4,w/4)
        x2 = _x[1]  # (_,128,h/8,w/8)
        x3 = _x[2]  # (_,320,h/16,w/16)
        x4 = _x[3]  # (_,512,h/32,w/32)

        atten_x2, s2 = self.atten2(x2)
        atten_x3, s3 = self.atten3(x3)
        atten_x4, s4 = self.atten4(x4)

        # x1_t = self.Translayer1_1(x1)
        x2_t = self.Translayer2_1(atten_x2)
        x3_t = self.Translayer3_1(atten_x3)
        x4_t = self.Translayer4_1(atten_x4)
        # mid_out = x3_t
        cam_feature = self.aggeration(x2_t, x3_t, x4_t)
        # mid_out = cam_feature
        out = self.outconv(cam_feature)

        prediction = F.interpolate(out, scale_factor=8, mode='bilinear')

        return prediction, s2, s3, s4  # s2[x,1,8,8], s3[x,1,4,4], s4[x,1,2,2]

    def _preload_data(self, batch):
        img, msk, name = batch
        img = img.float()
        # msk = msk.to(torch.float32)
        # msk = msk.unsqueeze(1)
        return img, msk, name

    def training_step(self, batch, batch_idx, train_metrics, normalization):
        img, msk, _ = self._preload_data(batch)
        msk_expand = expand_as_one_hot(msk.long(), self.n_classes)
        msk_pool2 = rearrange(msk_expand, 'b c (h n) (w m) -> b c h w (n m)', h=8, w=8)  # patch_size=4
        msk_pool2 = torch.sum(msk_pool2, dim=-1)
        msk_pool2[msk_pool2 == 0] = 0
        msk_pool2[msk_pool2 == 16] = 0
        msk_pool2[msk_pool2 > 0] = 1

        msk_pool3 = rearrange(msk_expand, 'b c (h n) (w m) -> b c h w (n m)', h=4, w=4)  # patch_size=4
        msk_pool3 = torch.sum(msk_pool3, dim=-1)
        msk_pool3[msk_pool3 == 0] = 0
        msk_pool3[msk_pool3 == 16] = 0
        msk_pool3[msk_pool3 > 0] = 1

        msk_pool4 = rearrange(msk_expand, 'b c (h n) (w m) -> b c h w (n m)', h=2, w=2)  # patch_size=4
        msk_pool4 = torch.sum(msk_pool4, dim=-1)
        msk_pool4[msk_pool4 == 0] = 0
        msk_pool4[msk_pool4 == 16] = 0
        msk_pool4[msk_pool4 > 0] = 1

        msk_pred, s2, s3, s4 = self(img)
        predict = normalization(msk_pred)
        predict = torch.argmax(predict, dim=1).float()
        train_metrics.update(predict, msk.int())
        # msk_pred = torch.sigmoid(msk_pred)
        loss_seg = self.loss_func(msk_pred, msk)
        loss_score2 = self.criteria(s2, msk_pool2)
        loss_score3 = self.criteria(s3, msk_pool3)
        loss_score4 = self.criteria(s4, msk_pool4)
        return 0.7 * loss_seg + 0.1 * loss_score2 + 0.1 * loss_score3 + 0.1 * loss_score4

    def validation_step(self, batch, batch_idx, valid_metrics, normalization):
        img, msk, _ = self._preload_data(batch)
        msk_pred, _, _, _ = self(img)
        predict = normalization(msk_pred)
        predict = torch.argmax(predict, dim=1).float()
        valid_metrics.update(predict, msk.int())
        return self.loss_func(msk_pred, msk)

    def test_step(self, batch, batch_idx):
        img, msk, _ = batch
        msk_pred, _, _, _ = self(img)
        return torch.squeeze(msk_pred, dim=1)


class LFM_GAM(nn.Module):
    def __init__(self, in_channels, out_channels, win_size, dim, numbers):
        super(LFM_GAM, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.win_size = win_size
        # self.pool = nn.AdaptiveAvgPool2d(patch_num)
        self.sigmoid = nn.Sigmoid()
        # self.position_embedding = nn.Parameter(torch.zeros((1, 784, in_channels)))    #维度要改变
        self.layerNorm1 = nn.LayerNorm(dim, eps=1e-6)
        self.win = WindowAttention(win_size, dim=dim)
        self.GlobalAtten = Attention_Global(dim=dim, heads=8)

        self.dlightconv = DlightConv(dim, win_size)
        self.linear = nn.Linear(numbers, numbers)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)  # win_size = 4时
        # self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)   #win_size = 2时

    def forward(self, x):
        origin_x = x  # [B,C,H,W]
        x = self.conv(x)
        _, _, h, w = x.shape
        num_h = h // self.win_size
        num_w = w // self.win_size
        score = F.adaptive_avg_pool2d(x, (num_h, num_w))  # [B,1,patch_num,patch_num]
        score = self.sigmoid(score)
        score_1d = score[:, 0, :, :]  # b n   为patch中既有正例又有负例的概率

        # s = self.pred_class(x)
        # score = self.softmax(s)  #[B,C,H,W]

        x_3d = origin_x.flatten(2)  # (B,C,N)
        x_3d = x_3d.transpose(-2, -1)  # (B, N, C)

        # x = x + self.position_embedding
        x_3d = self.layerNorm1(x_3d)

        x_3d = self.win(x_3d, score_1d)  ## (b, p, p, win, c)  返回的是加上原来的特征的
        x_reshape = rearrange(x_3d, 'b h w (n1 n2) d -> b d (h n1) (w n2)', n1=self.win_size,
                              n2=self.win_size)  # reshape成原来的形状
        # mid_out = x_reshape
        # b, p, p, win, c = x_3d.shape
        # h = x_3d.view(b, p, p, int(np.sqrt(win)), int(np.sqrt(win)),
        #            c).permute(0, 1, 3, 2, 4, 5).contiguous()
        # h = h.view(b, p * int(np.sqrt(win)), p * int(np.sqrt(win)),
        #            c).permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)
        x_3d = self.dlightconv(x_3d)  # (b, n, n, h)   聚合每个小window为一个TOKEN
        x_3d = x_3d.permute(0, 3, 1, 2)
        x_3d = x_3d.flatten(2)
        x_3d = x_3d.transpose(-2, -1)

        x_reshape = x_reshape.flatten(2)
        x_reshape = self.linear(x_reshape)
        x_reshape = x_reshape.transpose(-2, -1)

        x_att = self.GlobalAtten(x_3d, x_reshape, score_1d)
        b1, n1, c1 = x_att.shape
        x_att = x_att.view(b1, int(np.sqrt(n1)), int(np.sqrt(n1)), c1).permute(0, 3, 1, 2).contiguous()
        x_att = self.up4(x_att)  # 注意修改
        out = x_att + origin_x

        return out, score


class Attention_Local(nn.Module):
    def __init__(self, dim, num_head):
        super(Attention_Local, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.attention_head_size = int(self.dim / self.num_head)
        self.all_head_size = self.num_head * self.attention_head_size

        self.query_layer = nn.Linear(self.dim, self.all_head_size)
        self.key_layer = nn.Linear(self.dim, self.all_head_size)
        self.value_layer = nn.Linear(self.dim, self.all_head_size)

        self.out = nn.Linear(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_head, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, x):
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        query_layer = self.transpose_for_scores(query).permute(
            0, 1, 2, 4, 3, 5).contiguous()  # (b, p, p, head, n, c)
        key_layer = self.transpose_for_scores(key).permute(
            0, 1, 2, 4, 3, 5).contiguous()
        value_layer = self.transpose_for_scores(value).permute(
            0, 1, 2, 4, 3, 5).contiguous()

        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        atten_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(
            atten_probs, value_layer)  # (b, p, p, head, win, h)
        context_layer = context_layer.permute(0, 1, 2, 4, 3,
                                              5).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)

        return attention_output


class Attention_Global(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        # inner_dim = dim_head * heads
        # project_out = not (heads == 1 and dim_head == dim)

        self.thresh = 0.5
        self.heads = heads
        self.attention_head_size = int(dim / self.heads)
        self.scale = self.attention_head_size ** -0.5
        inner_dim = self.heads * self.attention_head_size

        self.softmax = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x_t, x_p, score):
        q = self.to_q(x_t)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        k = self.to_k(x_p)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = self.to_v(x_p)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.softmax(dots)

        # b, g, n, _ = attn.shape
        # score = rearrange(score,'b h w -> b (h w)')
        # score = score[:,None,:].repeat(1,g,1)
        # attn[score < self.thresh,:] = 0

        attn_out = torch.matmul(attn, v)
        attn_out = rearrange(attn_out, 'b h n d -> b n (h d)')
        out = self.to_out(attn_out)

        return out


class DlightConv(nn.Module):
    def __init__(self, dim, win_size):
        super(DlightConv, self).__init__()
        self.linear = nn.Linear(dim, win_size * win_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = x
        avg_x = torch.mean(x, dim=-2)  # (b, n, n, 1, h)
        x_prob = self.softmax(self.linear(avg_x))  # (b, n, n, win)

        x = torch.mul(h,
                      x_prob.unsqueeze(-1))  # (b, p, p, 16, h) (b, p, p, 16)
        x = torch.sum(x, dim=-2)  # (b, n, n, 1, h)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Three_Cat_Aggeration(nn.Module):
    def __init__(self, channel=32):
        super(Three_Cat_Aggeration, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.fuse34 = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.fuse2 = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.fuse1 = BasicConv2d(channel * 2, channel, 3, padding=1)

    def forward(self, x2, x3, x4):
        x4 = self.upsample(x4)
        x34 = self.fuse34(torch.cat([x3, x4], dim=1))
        x34 = self.upsample(x34)
        x34_2 = self.fuse2(torch.cat([x34, x2], dim=1))
        # x34_2 = self.upsample(x34_2)
        # x_fuse = self.fuse1(torch.cat([x34_2,x1],dim=1))

        return x34_2


class WindowAttention(nn.Module):
    def __init__(self, win_size, dim):
        super().__init__()
        self.window_size = win_size
        self.attention = Attention_Local(dim, num_head=8)
        self.thresh = 0.5

    def forward(self, x, prob):
        # prob, [B, patch_num, patch_num]
        b, n, c = x.shape
        h, w = int(np.sqrt(n)), int(np.sqrt(n))
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, w)
        if h % self.window_size != 0:
            right_size = h + self.window_size - h % self.window_size
            new_x = torch.zeros((b, c, right_size, right_size))
            new_x[:, :, 0:x.shape[2], 0:x.shape[3]] = x[:]
            new_x[:, :, x.shape[2]:,
            x.shape[3]:] = x[:, :, (x.shape[2] - right_size):,
                           (x.shape[3] - right_size):]
            x = new_x
            b, c, h, w = x.shape
        x = x.view(b, c, h // self.window_size, self.window_size,
                   w // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5,
                      1).contiguous().view(b, h // self.window_size,
                                           w // self.window_size,
                                           self.window_size * self.window_size,
                                           c).cuda()

        origin_x = x
        atten_x = self.attention(x)  # (b, p, p, win, h)    怎么设置其他只有背景和前景的patch不参与计算
        atten_x[prob < self.thresh, :, :] = 0  # 怎么设置其他只有背景和前景的patch不参与计算
        x = origin_x + atten_x
        return x  ## (b, p, p, win, h)
