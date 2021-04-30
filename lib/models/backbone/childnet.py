import warnings

warnings.filterwarnings('ignore')

import os
import logging
import torch
from collections import OrderedDict
from lib.models.backbone.models import _gen_childnet


def build_subnet(path_backbone, ops=None):
    arch_list = path_backbone

    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25',
         'ir_r1_k3_s1_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s1_e4_c40_se0.25',
         'ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25',
         'ir_r1_k3_s1_e4_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25'],
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25',
         'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25'],
        # stage 5, 14x14in
        ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25',
         'ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25'],
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c320_se0.25'],
    ]

    model = _gen_childnet(
        arch_list,
        arch_def,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0,
        global_pool='avg',
        bn_momentum=None,
        bn_eps=None,
        pool_bn=False,
        zero_gamma=False,
        ops=ops)

    return model


def resume_checkpoint(model, checkpoint_path, ema=True):
    """2020.11.5 Modified from timm"""
    other_state = {}
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_name = 'state_dict_ema' if ema else 'state_dict'
        if isinstance(checkpoint, dict) and state_dict_name in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_name].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            try:
                model.load_state_dict(new_state_dict, strict=True)
            except:
                model.load_state_dict(new_state_dict, strict=False)
                print('strict = %s' % False)
            if 'optimizer' in checkpoint:
                other_state['optimizer'] = checkpoint['optimizer']
            if 'amp' in checkpoint:
                other_state['amp'] = checkpoint['amp']
            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save
            logging.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            logging.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return other_state, resume_epoch
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
