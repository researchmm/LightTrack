from lib.models.backbone import build_subnet
from lib.models.backbone import build_supernet_DP
from lib.models.super_connect import head_supernet, MC_BN, Point_Neck_Mobile_simple_DP
from lib.models.super_model_DP import Super_model_DP, Super_model_DP_MACs, Super_model_DP_retrain
from lib.utils.transform import name2path
from lib.models.submodels import build_subnet_head, build_subnet_BN, build_subnet_feat_fusor
import numpy as np
import random


class LightTrackM_Supernet(Super_model_DP):
    def __init__(self, search_size=256, template_size=128, stride=16, adj_channel=128, build_module=True):
        """subclass calls father class's __init__ func"""
        super(LightTrackM_Supernet, self).__init__(search_size=search_size, template_size=template_size,
                                                   stride=stride)  # ATTENTION
        # config #
        # which parts to search
        self.search_back, self.search_ops, self.search_head = 1, 1, 1
        # backbone config
        self.stage_idx = [1, 2, 3]  # which stages to use
        self.max_flops_back = 470
        # head config
        self.channel_head = [128, 192, 256]
        self.kernel_head = [3, 5, 0]  # 0 means skip connection
        self.tower_num = 8  # max num of layers in the head
        self.num_choice_channel_head = len(self.channel_head)
        self.num_choice_kernel_head = len(self.kernel_head)
        # Compute some values #
        self.in_c = [self.channel_back[idx] for idx in self.stage_idx]
        strides_use = [self.strides[idx] for idx in self.stage_idx]
        strides_use_new = []
        for item in strides_use:
            if item not in strides_use_new:
                strides_use_new.append(item)  # remove repeated elements
        self.strides_use_new = strides_use_new
        self.num_kernel_corr = [int(round(template_size / stride) ** 2) for stride in strides_use_new]
        # build the architecture #
        if build_module:
            self.features, self.sta_num = build_supernet_DP(flops_maximum=self.max_flops_back)
            self.neck = MC_BN(inp_c=self.in_c)  # BN with multiple types of input channels
            self.feature_fusor = Point_Neck_Mobile_simple_DP(num_kernel_list=self.num_kernel_corr, matrix=True,
                                                             adj_channel=adj_channel)  # stride=8, stride=16
            self.supernet_head = head_supernet(channel_list=self.channel_head, kernel_list=self.kernel_head,
                                               linear_reg=True, inchannels=adj_channel, towernum=self.tower_num)
        else:
            _, self.sta_num = build_supernet_DP(flops_maximum=self.max_flops_back)


class LightTrackM_FLOPs(Super_model_DP_MACs):
    def __init__(self, search_size=256, template_size=128, stride=16, adj_channel=128):
        '''subclass calls father class's __init__ func'''
        super(LightTrackM_FLOPs, self).__init__(search_size=search_size, template_size=template_size,
                                                stride=stride)  # ATTENTION
        self.model = LightTrackM_Supernet(search_size=search_size, template_size=template_size,
                                          stride=stride, adj_channel=adj_channel)


class LightTrackM_Subnet(Super_model_DP_retrain):
    def __init__(self, path_name, search_size=256, template_size=128, stride=16, adj_channel=128):
        """subclass calls father class's __init__ func"""
        super(LightTrackM_Subnet, self).__init__(search_size=search_size, template_size=template_size,
                                                 stride=stride)  # ATTENTION
        model_cfg = LightTrackM_Supernet(search_size=search_size, template_size=template_size,
                                         stride=stride, adj_channel=adj_channel, build_module=False)

        path_backbone, path_head, path_ops = name2path(path_name, sta_num=model_cfg.sta_num)
        # build the backbone
        self.features = build_subnet(path_backbone, ops=path_ops)  # sta_num is based on previous flops
        # build the neck layer
        self.neck = build_subnet_BN(path_ops, model_cfg)
        # build the Correlation layer and channel adjustment layer
        self.feature_fusor = build_subnet_feat_fusor(path_ops, model_cfg, matrix=True, adj_channel=adj_channel)
        # build the head
        self.head = build_subnet_head(path_head, channel_list=model_cfg.channel_head, kernel_list=model_cfg.kernel_head,
                                      inchannels=adj_channel, linear_reg=True, towernum=model_cfg.tower_num)


class LightTrackM_Speed(LightTrackM_Subnet):
    def __init__(self, path_name, search_size=256, template_size=128, stride=16, adj_channel=128):
        super(LightTrackM_Speed, self).__init__(path_name, search_size=search_size, template_size=template_size,
                                                stride=stride, adj_channel=adj_channel)

    def forward(self, x, zf):
        # backbone
        xf = self.features(x)
        # BN before Point-wise Corr
        zf, xf = self.neck(zf, xf)
        # Point-wise Correlation
        feat_dict = self.feature_fusor(zf, xf)
        # head
        oup = self.head(feat_dict)
        return oup


class SuperNetToolbox(object):
    def __init__(self, model):
        self.model = model

    def get_path_back(self, prob=None):
        """randomly sample one path from the backbone supernet"""
        if prob is None:
            path_back = [np.random.choice(self.model.num_choice_back, item).tolist() for item in self.model.sta_num]
        else:
            path_back = [np.random.choice(self.model.num_choice_back, item, prob).tolist() for item in
                         self.model.sta_num]
        # add head and tail
        path_back.insert(0, [0])
        path_back.append([0])
        return path_back

    def get_path_head_single(self):
        num_choice_channel_head = self.model.num_choice_channel_head
        num_choice_kernel_head = self.model.num_choice_kernel_head
        tower_num = self.model.tower_num
        oup = [random.randint(0, num_choice_channel_head - 1)]  # num of choices for head's channel
        arch = [random.randint(0, num_choice_kernel_head - 2)]
        arch += list(np.random.choice(num_choice_kernel_head, tower_num - 1))  # 3x3 conv, 5x5 conv, skip
        oup.append(arch)
        return oup

    def get_path_head(self):
        """randomly sample one path from the head supernet"""
        cand_h_dict = {'cls': self.get_path_head_single(), 'reg': self.get_path_head_single()}
        return cand_h_dict

    def get_path_ops(self):
        """randomly sample an output position"""
        stage_idx = random.choice(self.model.stage_idx)
        block_num = self.model.sta_num[stage_idx]
        block_idx = random.randint(0, block_num - 1)
        return [stage_idx, block_idx]

    def get_one_path(self):
        """randomly sample one complete path from the whole supernet"""
        cand_back, cand_OP, cand_h_dict = None, None, None
        tower_num = self.model.tower_num
        if self.model.search_back or self.model.search_ops:
            # backbone operations
            cand_back = self.get_path_back()
        if self.model.search_ops:
            # backbone output positions
            cand_OP = self.get_path_ops()
        if self.model.search_head:
            # head operations
            cand_h_dict = self.get_path_head()
        else:
            cand_h_dict = {'cls': [0, [0] * tower_num], 'reg': [0, [0] * tower_num]}  # use fix head (only one choice)
        return {'back': cand_back, 'ops': cand_OP, 'head': cand_h_dict}
