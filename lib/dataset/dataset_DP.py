from __future__ import division

import os
import cv2
import json
import torch
import random
import logging
import numpy as np
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
from os.path import join
from easydict import EasyDict as edict
from torch.utils.data import Dataset

import sys
from lib.utils.utils import *
from lib.core.config_ocean import config
from lib.dataset.dataset_base import OceanDataset

sample_random = random.Random()


class OceanDataset_DP(OceanDataset):
    """for providing dynamic labels (s=8, s=16)"""

    def __init__(self, cfg):
        super(OceanDataset_DP, self).__init__(cfg)
        self.strides = [16, 8]
        self.sizes = [round(self.search_size / stride) for stride in self.strides]

    def __getitem__(self, index):
        """
        pick a vodeo/frame --> pairs --> data aug --> label
        """
        index = self.pick[index]
        dataset, index = self._choose_dataset(index)

        template, search = dataset._get_pairs(index, dataset.data_name)
        template, search = self.check_exists(index, dataset, template, search)

        template_image = cv2.cvtColor(cv2.imread(template[0]), cv2.COLOR_BGR2RGB)
        search_image = cv2.cvtColor(cv2.imread(search[0]), cv2.COLOR_BGR2RGB)

        template_box = self._toBBox(template_image, template[1])
        search_box = self._toBBox(search_image, search[1])

        template, _, _ = self._augmentation(template_image, template_box, self.template_size)
        search, bbox, dag_param = self._augmentation(search_image, search_box, self.search_size, search=True)

        # from PIL image to numpy
        template = np.array(template)
        search = np.array(search)
        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])
        '''Normalization'''
        template, search = map(lambda x: self.transform_norm(torch.tensor(x) / 255.0), [template, search])
        label_list = []
        for self.size, self.stride in zip(self.sizes, self.strides):
            self.grids()
            '''2020.08.13 positive region is adaptive to the stride'''
            out_label = self._dynamic_label([self.size, self.size], dag_param.shift,
                                            rPos=16 / self.stride)
            reg_label, reg_weight = self.reg_label(bbox)
            label_list.append([out_label, reg_label, reg_weight])
        out_label_16, reg_label_16, reg_weight_16 = label_list[0]  # stride = 16
        out_label_8, reg_label_8, reg_weight_8 = label_list[1]  # stride = 8
        return template, search, out_label_16, reg_label_16, reg_weight_16, out_label_8, reg_label_8, reg_weight_8
