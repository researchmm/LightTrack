from collections import defaultdict
from io import BytesIO, StringIO
import os
import lmdb
import numpy as np
import cv2
from PIL import Image
from lib.core.config import config as train_config
import atexit
import time

LMDB_ENVS = dict()
LMDB_HANDLES = dict()
LMDB_FILELISTS = dict()


def get_lmdb_handle(name):
    global LMDB_HANDLES, LMDB_FILELISTS
    item = LMDB_HANDLES.get(name, None)
    if item is None:
        env = lmdb.open(name, readonly=True, lock=False, readahead=False, meminit=False)
        LMDB_ENVS[name] = env
        item = env.begin(write=False)
        LMDB_HANDLES[name] = item
    return item


def get_lmdb_fname(fname, prefix='./data/', lower_name='y2b'):
    '''
    Input:
        fname: a normal path

    Return:
        lmdb filename, filename in lmdb
    '''
    fname = fname[len(prefix)+1:]
    lmdb_fname = os.path.join(prefix, '%s_lmdb'%lower_name)
    return lmdb_fname, fname


def my_open(fname, mode='r', prefix='./data/', lower_name='y2b'):
    assert mode in ['r', 'rb', 's']
    lmdb_fname, fname = get_lmdb_fname(fname, prefix=prefix, lower_name=lower_name)
    handle = get_lmdb_handle(lmdb_fname)
    binfile = handle.get(fname.encode())
    if binfile is None:
        return None
    if mode == 's':
        return binfile
    elif mode == 'r':
        return StringIO(binfile.decode('utf-8'))
    return BytesIO(binfile)


def my_cv2_imread(fname, prefix, lower_name):
    # return BGR
    if not fname.startswith('./data/') and os.path.exists(fname):
        return cv2.imread(fname)
    binfile = my_open(fname, 's',  prefix, lower_name)
    if binfile is None:
        return None
    s = np.frombuffer(binfile, np.uint8)
    x = cv2.imdecode(s, cv2.IMREAD_COLOR)
    return x


def my_cv2_imread_rgb(fname, prefix, lower_name):
    # return RGB
    bgr = my_cv2_imread(fname, prefix, lower_name)
    if bgr is None:
        return None
    rgb = np.ascontiguousarray(bgr[:, :, [2, 1, 0]])
    return rgb


def get_file_list(fname):
    global LMDB_FILELISTS
    lmdb_fname, _ = get_lmdb_fname(fname)
    items = LMDB_FILELISTS.get(lmdb_fname, None)
    if items is None:
        handle = get_lmdb_handle(lmdb_fname)
        print(f"Reading filelist {lmdb_fname}")
        items = set([k.decode() for k, _ in handle.cursor()])
        print(f"filelist {lmdb_fname}: {len(items)}")
        LMDB_FILELISTS[lmdb_fname] = items

    return items


def my_pil_image_open(fname):
    binfile = my_open(fname, 'rb')
    if binfile is None:
        return None
    return np.array(Image.open(binfile))


def _loader_exit():
    global LMDB_ENVS
    for key in list(LMDB_ENVS.keys()):
        print(f"close lmdb {key}")
        env = LMDB_ENVS[key]
        env.close()
        del LMDB_ENVS[key]


class _LOADER_EXIT_CLS:
    def __exit__(self):
        _loader_exit()


_loader_exit_cls = _LOADER_EXIT_CLS()


atexit.register(_loader_exit)
