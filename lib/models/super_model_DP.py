import torch
import numpy as np
import torch.nn as nn
from lib.models.super_model import Super_model

'''2020.10.14 Super model that supports dynamic positions for the head'''


class Super_model_DP(Super_model):
    def __init__(self, search_size=256, template_size=128, stride=16):
        super(Super_model_DP, self).__init__(search_size=search_size, template_size=template_size, stride=stride)
        self.strides = [4, 8, 16, 16, 32]
        self.channel_back = [24, 40, 80, 96, 192]
        self.num_choice_back = 6

    def feature_extractor(self, x, cand_b, backbone_index):
        """cand_b: candidate path for backbone"""
        if self.retrain:
            return self.features(x, stride=self.stride)
        else:
            return self.features(x, cand_b, stride=self.stride, backbone_index=backbone_index)

    def template(self, z, cand_b, backbone_index):
        self.zf = self.feature_extractor(z, cand_b, backbone_index)

    def track(self, x, cand_b, cand_h_dict, backbone_index):
        # supernet backbone
        xf = self.feature_extractor(x, cand_b, backbone_index)
        # Batch Normalization before Corr
        zf, xf = self.neck(self.zf, xf, self.stage_idx.index(backbone_index[0]))
        # Point-wise Correlation
        stride = self.strides[backbone_index[0]]
        stride_idx = self.strides_use_new.index(stride)
        feat_dict = self.feature_fusor(zf, xf, stride_idx)
        # supernet head
        oup = self.supernet_head(feat_dict, cand_h_dict)
        return oup

    def forward(self, template, search, label=None, reg_target=None, reg_weight=None,
                cand_b=None, cand_h_dict=None, backbone_index=None):
        """backbone_index: which layer's feature to use"""
        zf = self.feature_extractor(template, cand_b, backbone_index)
        xf = self.feature_extractor(search, cand_b, backbone_index)
        # Batch Normalization before Corr
        zf, xf = self.neck(zf, xf, self.stage_idx.index(backbone_index[0]))
        # Point-wise Correlation
        stride = self.strides[backbone_index[0]]
        stride_idx = self.strides_use_new.index(stride)
        feat_dict = self.feature_fusor(zf, xf, stride_idx)
        # supernet head
        oup = self.supernet_head(feat_dict, cand_h_dict)
        if label is not None and reg_target is not None and reg_weight is not None:
            # compute loss
            reg_loss = self.add_iouloss(oup['reg'], reg_target, reg_weight)
            cls_loss = self._weighted_BCE(oup['cls'], label)
            return cls_loss, reg_loss
        return oup

    def get_attribute(self):
        return {'search_back': self.search_back, 'search_out': self.search_ops, 'search_head': self.search_head}

    def clean_module_BN(self, model):
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)

    def clean_BN(self):
        print('clear bn statics....')
        if self.search_back:
            print('cleaning backbone BN ...')
            self.clean_module_BN(self.features)
        if self.search_head:
            print('cleaning head BN ...')
            self.clean_module_BN(self.supernet_head)
        if self.search_ops:
            print('cleaning neck and feature_fusor BN ...')
            self.clean_module_BN(self.neck)
            self.clean_module_BN(self.feature_fusor)


'''2020.10.17 Compute MACs for DP networks'''


class Super_model_DP_MACs(nn.Module):
    def __init__(self, search_size=256, template_size=128, stride=16):
        super(Super_model_DP_MACs, self).__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.stride = stride

    def feature_extractor(self, x, cand_b, backbone_index):
        '''cand_b: candidate path for backbone'''
        return self.model.features.forward_backbone(x, cand_b, stride=self.stride, backbone_index=backbone_index)

    def forward(self, zf, search, cand_b, cand_h_dict, backbone_index):
        xf = self.model.feature_extractor(search, cand_b, backbone_index)
        # Batch Normalization before Corr
        neck_idx = self.model.stage_idx.index(backbone_index[0])
        zf, xf = self.model.neck(zf[neck_idx], xf, neck_idx)
        # Point-wise Correlation
        stride = self.model.strides[backbone_index[0]]
        stride_idx = self.model.strides_use_new.index(stride)
        feat_dict = self.model.feature_fusor(zf, xf, stride_idx)
        # supernet head
        oup = self.model.supernet_head(feat_dict, cand_h_dict)
        return oup


'''2020.10.14 Super model that supports dynamic positions for the head'''
'''2020.10.18 for retrain the searched model'''


class Super_model_DP_retrain(Super_model):
    def __init__(self, search_size=256, template_size=128, stride=16):
        super(Super_model_DP_retrain, self).__init__(search_size=search_size, template_size=template_size,
                                                     stride=stride)

    def template(self, z):
        self.zf = self.features(z)

    def track(self, x):
        # supernet backbone
        xf = self.features(x)
        # BN before Pointwise Corr
        zf, xf = self.neck(self.zf, xf)
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.head(feat_dict)
        return oup['cls'], oup['reg']

    def forward(self, template, search, label, reg_target, reg_weight):
        '''backbone_index: which layer's feature to use'''
        zf = self.features(template)
        xf = self.features(search)
        # Batch Normalization before Corr
        zf, xf = self.neck(zf, xf)
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # supernet head
        oup = self.head(feat_dict)
        # compute loss
        reg_loss = self.add_iouloss(oup['reg'], reg_target, reg_weight)
        cls_loss = self._weighted_BCE(oup['cls'], label)
        return cls_loss, reg_loss
