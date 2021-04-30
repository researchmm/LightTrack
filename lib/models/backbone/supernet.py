import numpy as np
import torch
from lib.models.backbone.models.hypernet import _gen_supernet


def set_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_supernet(flops_maximum=600):
    set_seed()
    model, sta_num, size_factor = _gen_supernet(
        flops_minimum=0,
        flops_maximum=flops_maximum,
        num_classes=1000,
        drop_rate=0.0,
        global_pool='avg',
        resunit=False,
        dil_conv=False,
        slice=4)

    return model, sta_num


def build_supernet_DP(flops_maximum=600):
    """Backbone with Dynamic output position"""
    set_seed()
    model, sta_num, size_factor = _gen_supernet(
        flops_minimum=0,
        flops_maximum=flops_maximum,
        DP=True,
        num_classes=1000,
        drop_rate=0.0,
        global_pool='avg',
        resunit=False,
        dil_conv=False,
        slice=4)

    return model, sta_num


if __name__ == '__main__':
    _, sta_num = build_supernet(flops_maximum=600)
    print(sta_num)
