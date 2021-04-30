import torch.nn as nn
import torch
from lib.models.super_connect import *


class head_subnet(nn.Module):
    def __init__(self, module_dict):
        super(head_subnet, self).__init__()
        self.cls_tower = module_dict['cls_tower']
        self.reg_tower = module_dict['reg_tower']
        self.cls_perd = module_dict['cls_pred']
        self.reg_pred = module_dict['reg_pred']

    def forward(self, inp):
        oup = {}
        # cls
        cls_feat = self.cls_tower(inp['cls'])
        oup['cls'] = self.cls_perd(cls_feat)
        # reg
        reg_feat = self.reg_tower(inp['reg'])
        oup['reg'] = self.reg_pred(reg_feat)
        return oup


def get_towers(module_list: torch.nn.ModuleList, path_head, inchannels, outchannels, towernum=8, kernel_list=[3, 5, 0]):
    num_choice_kernel = len(kernel_list)
    for tower_idx in range(towernum):
        block_idx = path_head[1][tower_idx]
        kernel_sz = kernel_list[block_idx]
        if tower_idx == 0:
            assert (kernel_sz != 0)
            padding = (kernel_sz - 1) // 2
            module_list.append(SeparableConv2d_BNReLU(inchannels, outchannels, kernel_size=kernel_sz,
                                                      stride=1, padding=padding, dilation=1))
        else:
            if block_idx != num_choice_kernel - 1:  # else skip
                assert (kernel_sz != 0)
                padding = (kernel_sz - 1) // 2
                module_list.append(SeparableConv2d_BNReLU(outchannels, outchannels, kernel_size=kernel_sz,
                                                          stride=1, padding=padding, dilation=1))
    return module_list


def build_subnet_head(path_head, channel_list=[128, 192, 256], kernel_list=[3, 5, 0], inchannels=64, towernum=8,
                      linear_reg=False):
    channel_idx_cls, channel_idx_reg = path_head['cls'][0], path_head['reg'][0]
    num_channel_cls, num_channel_reg = channel_list[channel_idx_cls], channel_list[channel_idx_reg]
    tower_cls_list = nn.ModuleList()
    tower_reg_list = nn.ModuleList()
    # add operations
    tower_cls = nn.Sequential(
        *get_towers(tower_cls_list, path_head['cls'], inchannels, num_channel_cls, towernum=towernum,
                    kernel_list=kernel_list))
    tower_reg = nn.Sequential(
        *get_towers(tower_reg_list, path_head['reg'], inchannels, num_channel_reg, towernum=towernum,
                    kernel_list=kernel_list))
    # add prediction head
    cls_pred = cls_pred_head(inchannels=num_channel_cls)
    reg_pred = reg_pred_head(inchannels=num_channel_reg, linear_reg=linear_reg)

    module_dict = {'cls_tower': tower_cls, 'reg_tower': tower_reg, 'cls_pred': cls_pred, 'reg_pred': reg_pred}
    return head_subnet(module_dict)


########## BN adjust layer before Correlation ##########
class BN_adj(nn.Module):
    def __init__(self, num_channel):
        super(BN_adj, self).__init__()
        self.BN_z = nn.BatchNorm2d(num_channel)
        self.BN_x = nn.BatchNorm2d(num_channel)

    def forward(self, zf, xf):
        return self.BN_z(zf), self.BN_x(xf)


def build_subnet_BN_backup(path_ops, inp_c=(40, 80, 96)):
    num_channel = inp_c[path_ops[0] - 1]
    return BN_adj(num_channel)


def build_subnet_BN(path_ops, model_cfg):
    inc_idx = model_cfg.stage_idx.index(path_ops[0])
    num_channel = model_cfg.in_c[inc_idx]
    return BN_adj(num_channel)


class Point_Neck_Mobile_simple_DP(nn.Module):
    def __init__(self, num_kernel_list=(256, 64), cat=False, matrix=True, adjust=True, adj_channel=128):
        super(Point_Neck_Mobile_simple_DP, self).__init__()
        self.adjust = adjust
        '''Point-wise Correlation & Adjust Layer (unify the num of channels)'''
        self.pw_corr = torch.nn.ModuleList()
        self.adj_layer = torch.nn.ModuleList()
        for num_kernel in num_kernel_list:
            self.pw_corr.append(PWCA(num_kernel, cat=cat, CA=True, matrix=matrix))
            self.adj_layer.append(nn.Conv2d(num_kernel, adj_channel, 1))

    def forward(self, kernel, search, stride_idx):
        '''stride_idx: 0 or 1. 0 represents stride 8. 1 represents stride 16'''
        oup = {}
        corr_feat = self.pw_corr[stride_idx]([kernel], [search])
        if self.adjust:
            corr_feat = self.adj_layer[stride_idx](corr_feat)
        oup['cls'], oup['reg'] = corr_feat, corr_feat
        return oup


'''Point-wise Correlation & channel adjust layer'''


class PW_Corr_adj(nn.Module):
    def __init__(self, num_kernel=64, cat=False, matrix=True, adj_channel=128):
        super(PW_Corr_adj, self).__init__()
        self.pw_corr = PWCA(num_kernel, cat=cat, CA=True, matrix=matrix)
        self.adj_layer = nn.Conv2d(num_kernel, adj_channel, 1)

    def forward(self, kernel, search):
        '''stride_idx: 0 or 1. 0 represents stride 8. 1 represents stride 16'''
        oup = {}
        corr_feat = self.pw_corr([kernel], [search])
        corr_feat = self.adj_layer(corr_feat)
        oup['cls'], oup['reg'] = corr_feat, corr_feat
        return oup


def build_subnet_feat_fusor_backup(path_ops, num_kernel_list=(256, 64), cat=False, matrix=True, adj_channel=128):
    stride_idx = 0 if path_ops[0] == 1 else 1
    num_kernel = num_kernel_list[stride_idx]
    return PW_Corr_adj(num_kernel=num_kernel, cat=cat, matrix=matrix, adj_channel=adj_channel)


def build_subnet_feat_fusor(path_ops, model_cfg, cat=False, matrix=True, adj_channel=128):
    stride = model_cfg.strides[path_ops[0]]
    stride_idx = model_cfg.strides_use_new.index(stride)
    num_kernel = model_cfg.num_kernel_corr[stride_idx]
    return PW_Corr_adj(num_kernel=num_kernel, cat=cat, matrix=matrix, adj_channel=adj_channel)
