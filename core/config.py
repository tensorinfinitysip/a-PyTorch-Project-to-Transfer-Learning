# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import yaml
from easydict import EasyDict as edict

__C = edict()
opt = __C
__C.seed = 0

__C.dataset = edict()
__C.dataset.train_path = 'data/train/'
__C.dataset.valid_path = 'data/valid/'
__C.dataset.train_valid_path = 'data/train_valid'
__C.dataset.test_path = 'data/test'
__C.dataset.num_classes = 12

__C.aug = edict()
__C.aug.resize_size = 224
__C.aug.color_jitter = [0.4, 0.4, 0.4]
__C.aug.random_erasing = True
__C.aug.random_mirror = True

__C.train = edict()
__C.train.optimizer = 'SGD'
__C.train.lr = 0.01
__C.train.wd = 5e-4
__C.train.momentum = 0.9
__C.train.step = [20, 30]
__C.train.warmup_epoch = 5
__C.train.warmup_begin_lr = 1e-4
__C.train.factor = 0.1
__C.train.num_epochs = 50
__C.train.batch_size = 64
__C.train.checkpoints = ''

__C.train.gpus = '0'
__C.train.workers = 8

__C.network = edict()
__C.network.name = 'resnet50'

__C.misc = edict()
__C.misc.log_interval = 50
__C.misc.eval_step = 5
__C.misc.save_step = 5
__C.misc.save_dir = 'checkpoints/'


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in __C:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        __C[k][vk] = vv
                else:
                    __C[k] = v
            else:
                raise ValueError("key must exist in configs.py")
