# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import sys
from pprint import pprint

import torch
from torch import nn
from torch.backends import cudnn

import network
from core.config import opt, update_config
from core.loader import get_data_provider
from core.solver import Solver
from utils.lr_scheduler import LRScheduler

FORMAT = '[%(levelname)s]: %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    stream=sys.stdout
)


def train(args):
    logging.info('======= user config ======')
    logging.info(pprint(opt))
    logging.info(pprint(args))
    logging.info('======= end ======')

    train_data, valid_data = get_data_provider(opt)

    net = getattr(network, opt.network.name)(classes=opt.dataset.num_classes)
    optimizer = getattr(torch.optim, opt.train.optimizer)(net.parameters(), lr=opt.train.lr,
                                                          weight_decay=opt.train.wd, momentum=opt.train.momentum)
    ce_loss = nn.CrossEntropyLoss()
    lr_scheduler = LRScheduler(base_lr=opt.train.lr, step=opt.train.step, factor=opt.train.factor,
                               warmup_epoch=opt.train.warmup_epoch, warmup_begin_lr=opt.train.warmup_begin_lr)
    net = nn.DataParallel(net)
    net = net.cuda()
    mod = Solver(opt, net)
    mod.fit(train_data=train_data, test_data=valid_data, optimizer=optimizer, criterion=ce_loss,
            lr_scheduler=lr_scheduler)


def main():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--config_file', type=str, default=None, help='optional config file for training')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='save model directory')

    args = parser.parse_args()
    if args.config_file is not None:
        update_config(args.config_file)
    opt.misc.save_dir = args.save_dir
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.train.gpus
    cudnn.benchmark = True
    train(args)


if __name__ == '__main__':
    main()
