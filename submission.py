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

import pandas as pd
import torch
from torch import nn
from torch.backends import cudnn

import network
from core.config import opt, update_config
from core.loader import get_test_provider

FORMAT = '[%(levelname)s]: %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    stream=sys.stdout
)


def submission(args):
    logging.info('======= user config ======')
    logging.info(pprint(opt))
    logging.info(pprint(args))
    logging.info('======= end ======')

    test_loader, idx_to_class = get_test_provider(opt)
    net = getattr(network, opt.network.name)(classes=opt.dataset.num_classes, pretrain=False)
    net.load_state_dict(torch.load(args.model_path)['state_dict'])
    net.eval()
    net = nn.DataParallel(net)
    net = net.cuda()

    preds = list()
    img_ids = list()
    for data, fname in test_loader:
        with torch.no_grad():
            scores = net(data)
        pred_labels = scores.max(1)[1].cpu().numpy().tolist()
        preds.extend(pred_labels)
        img_ids.extend(fname)

    df = pd.DataFrame({'file': img_ids, 'species': preds})
    df['species'] = df['species'].apply(lambda x: idx_to_class[x])
    df.to_csv('submission.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='model testing')
    parser.add_argument('--config_file', type=str, default=None, help='optional config file for testing')
    parser.add_argument('--model_path', type=str, default='checkpoints/model_best.pth.tar',
                        help='loading model path')
    args = parser.parse_args()
    if args.config_file is not None:
        update_config(args.config_file)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.train.gpus
    cudnn.benchmark = True
    submission(args)


if __name__ == '__main__':
    main()
