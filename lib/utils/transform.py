import numpy as np
import random


def get_cand_with_prob(CHOICE_NUM, prob=None, sta_num=(4, 4, 4, 4, 4)):
    if prob is None:
        get_random_cand = [np.random.choice(CHOICE_NUM, item).tolist() for item in sta_num]
    else:
        get_random_cand = [np.random.choice(CHOICE_NUM, item, prob).tolist() for item in sta_num]
    # print(get_random_cand)
    return get_random_cand


def get_cand_head():
    oup = [random.randint(0, 2)]  # num of channels (3 choices)
    arch = [random.randint(0, 1)]
    arch += list(np.random.choice(3, 7))  # 3x3 conv, 5x5 conv, skip
    oup.append(arch)
    return oup


def get_cand_head_wo_ID():
    """2020.10.24 Without using IDentity"""
    oup = [random.randint(0, 2)]  # num of channels (3 choices)
    arch = []
    arch.append(random.randint(0, 1))  # 3x3 conv, 5x5 conv
    arch += list(np.random.choice(2, 7))  # 3x3 conv, 5x5 conv
    oup.append(arch)
    return oup


def get_oup_pos(sta_num):
    stage_idx = random.randint(2, 3)  # 1, 2, 3
    block_num = sta_num[stage_idx]
    block_idx = random.randint(0, block_num - 1)
    return [stage_idx, block_idx]


'''2020.10.5 name --> path'''
'''2020.10.17 modified version'''


def name2path_backhead(path_name, sta_num=(4, 4, 4, 4, 4), head_only=False, backbone_only=False):
    backbone_name, head_name = path_name.split('+cls_')
    if not head_only:
        # process backbone
        backbone_name = backbone_name.strip('back_')[1:-1]  # length = 20 when 600M, length = 18 when 470M
        backbone_path = [[], [], [], [], []]
        for stage_idx in range(len(sta_num)):
            for block_idx in range(sta_num[stage_idx]):
                str_idx = block_idx + sum(sta_num[:stage_idx])
                backbone_path[stage_idx].append(int(backbone_name[str_idx]))
        backbone_path.insert(0, [0])
        backbone_path.append([0])
    if not backbone_only:
        # process head
        cls_name, reg_name = head_name.split('+reg_')
        head_path = {}
        cls_path = [int(cls_name[0])]
        cls_path.append([int(item) for item in cls_name[1:]])
        head_path['cls'] = cls_path
        reg_path = [int(reg_name[0])]
        reg_path.append([int(item) for item in reg_name[1:]])
        head_path['reg'] = reg_path
    # combine
    if head_only:
        backbone_path = None
    if backbone_only:
        head_path = None
    return tuple([backbone_path, head_path])


'''2020.10.5 name --> path'''
'''2020.10.17 modified version'''


def name2path(path_name, sta_num=(4, 4, 4, 4, 4), head_only=False, backbone_only=False):
    if '_ops_' in path_name:
        first_name, ops_name = path_name.split('_ops_')
        backbone_path, head_path = name2path_backhead(first_name, sta_num=sta_num, head_only=head_only,
                                                      backbone_only=backbone_only)
        ops_path = (int(ops_name[0]), int(ops_name[1]))
        return backbone_path, head_path, ops_path
    else:
        return name2path_backhead(path_name, sta_num=sta_num, head_only=head_only, backbone_only=backbone_only)


def name2path_ablation(path_name, sta_num=(4, 4, 4, 4, 4), num_tower=8):
    back_path, head_path, ops_path = None, None, None
    if 'back' in path_name:
        back_str_len = sum(sta_num) + 2  # head0, tail0
        back_str = path_name.split('back_')[1][:back_str_len]
        back_str = back_str[1:-1]  # remove head0 and tail0
        back_path = [[], [], [], [], []]
        for stage_idx in range(len(sta_num)):
            for block_idx in range(sta_num[stage_idx]):
                str_idx = block_idx + sum(sta_num[:stage_idx])
                back_path[stage_idx].append(int(back_str[str_idx]))
        back_path.insert(0, [0])
        back_path.append([0])
    if 'cls' in path_name and 'reg' in path_name:
        head_path = {}
        cls_str_len = num_tower + 1  # channel idx
        cls_str = path_name.split('cls_')[1][:cls_str_len]
        cls_path = [int(cls_str[0]), [int(item) for item in cls_str[1:]]]
        head_path['cls'] = cls_path
        reg_str_len = num_tower + 1  # channel idx
        reg_str = path_name.split('reg_')[1][:reg_str_len]
        reg_path = [int(reg_str[0]), [int(item) for item in reg_str[1:]]]
        head_path['reg'] = reg_path
    if 'ops' in path_name:
        ops_str = path_name.split('ops_')[1]
        ops_path = (int(ops_str[0]), int(ops_str[1]))
    return {'back': back_path, 'head': head_path, 'ops': ops_path}


if __name__ == "__main__":
    for _ in range(10):
        print(get_cand_head())
