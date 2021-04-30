import _init_paths
import os
import cv2
import torch
import torch.utils.data
import random
import argparse
import numpy as np

import lib.models.models as models

from os.path import exists, join, dirname, realpath
from easydict import EasyDict as edict
from lib.utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

from lib.eval_toolkit.pysot.datasets import VOTDataset
from lib.eval_toolkit.pysot.evaluation import EAOBenchmark
from lib.tracker.lighttrack import Lighttrack


def parse_args():
    parser = argparse.ArgumentParser(description='Test LightTrack')
    parser.add_argument('--arch', dest='arch', help='backbone architecture')
    parser.add_argument('--resume', type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2019', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    parser.add_argument('--video', default=None, type=str, help='test a video in benchmark')
    parser.add_argument('--stride', type=int, help='network stride')
    parser.add_argument('--even', type=int, default=0)
    parser.add_argument('--path_name', type=str, default='NULL')
    args = parser.parse_args()

    return args


DATALOADER_NUM_WORKER = 2


class ImageDataset:
    def __init__(self, image_files):
        self.image_files = image_files

    def __getitem__(self, i):
        fname = self.image_files[i]
        im = cv2.imread(fname)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # align with training
        rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, rgb_im

    def __len__(self):
        return len(self.image_files)


def collate_fn(x):
    return x[0]


def track(siam_tracker, siam_net, video, args):
    start_frame, toc = 0, 0
    snapshot_dir = os.path.dirname(args.resume)
    result_dir = os.path.join(snapshot_dir, '../..', 'result')
    model_name = snapshot_dir.split('/')[-1]

    # save result to evaluate
    tracker_path = os.path.join(result_dir, args.dataset, model_name)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in args.dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return

    regions = []
    lost = 0

    image_files, gt = video['image_files'], video['gt']

    dataset = ImageDataset(image_files)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                             num_workers=DATALOADER_NUM_WORKER)

    with torch.no_grad():
        for f, x in enumerate(dataloader):
            im, rgb_im = x
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # align with training

            tic = cv2.getTickCount()
            if f == start_frame:  # init
                cx, cy, w, h = get_axis_aligned_bbox(gt[f])

                target_pos = np.array([cx, cy])
                target_sz = np.array([w, h])

                state = siam_tracker.init(rgb_im, target_pos, target_sz, siam_net)  # init tracker

                regions.append(1 if 'VOT' in args.dataset else gt[f])
            elif f > start_frame:  # tracking
                state = siam_tracker.track(state, rgb_im)

                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

                b_overlap = poly_iou(gt[f], location) if 'VOT' in args.dataset else 1
                if b_overlap > 0:
                    regions.append(location)
                else:
                    regions.append(2)
                    start_frame = f + 5
                    lost += 1
            else:
                regions.append(0)

            toc += cv2.getTickCount() - tic

    with open(result_path, "w") as fin:
        if 'VOT' in args.dataset:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {}'.format(video['name'], toc, f / toc, lost))


def main():
    args = parse_args()

    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.epoch_test = args.epoch_test
    info.stride = args.stride

    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = args.dataset
    siam_info.epoch_test = args.epoch_test
    siam_info.stride = args.stride
    # build tracker
    siam_tracker = Lighttrack(siam_info, even=args.even)
    # build siamese network
    if args.path_name != 'NULL':
        siam_net = models.__dict__[args.arch](args.path_name, stride=siam_info.stride)
    else:
        siam_net = models.__dict__[args.arch](stride=siam_info.stride)

    print('===> init Siamese <====')

    siam_net = load_pretrain(siam_net, args.resume)
    siam_net.eval()
    siam_net = siam_net.cuda()

    # prepare video
    dataset = load_dataset(args.dataset)
    video_keys = list(dataset.keys()).copy()

    if args.video is not None:
        track(siam_tracker, siam_net, dataset[args.video], args)
    else:
        for video in video_keys:
            track(siam_tracker, siam_net, dataset[video], args)


if __name__ == '__main__':
    main()
