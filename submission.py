# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os

import pandas as pd
import torch
import torchvision.transforms as T
from torch.backends import cudnn
from torch.utils.data import DataLoader

from datasets import ImageDataset
from model import *


def submission(network, test_loader, label2name, device):
    network.eval()
    network = network.to(device)

    preds = list()
    img_ids = list()
    for data, fname in test_loader:
        with torch.no_grad():
            data = data.to(device)
            scores = network(data)
        pred_labels = scores.max(1)[1].cpu().numpy().tolist()
        preds.extend(pred_labels)
        img_ids.extend(fname)

    df = pd.DataFrame({'file': img_ids, 'species': preds})
    df['species'] = df['species'].apply(lambda x: label2name[x])
    df.to_csv('submission.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='submission result')
    parser.add_argument('--test_dir', type=str, default='datasets/test', help='test image dir')
    parser.add_argument('--model_path', type=str, default='logs/tmp/models/model_best.pth',
                        help='loading model path')
    parser.add_argument('--test_size', type=int, default=[224, 224], nargs='+', help='test image size')
    parser.add_argument('--test_bs', type=int, default=128, help='test images per batch')
    parser.add_argument('--no_gpu', action='store_true', help='whether use gpu')
    parser.add_argument('--gpus', type=str, default='0', help='gpus to use in training')
    args = parser.parse_args()

    device = "cuda:{}".format(args.gpus) if not args.no_gpu else "cpu"
    if not args.no_gpu:
        cudnn.benchmark = True

    totensor = T.ToTensor()
    normalize = T.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])
    test_tfms = list()
    test_size = (args.test_size[0], args.test_size[0]) if len(args.test_size) == 1 else args.test_size
    test_tfms.append(T.Resize(test_size))
    test_tfms.append(totensor)
    test_tfms.append(normalize)
    test_tfms = T.Compose(test_tfms)

    # get test list
    test_list = glob.glob(os.path.join(args.test_dir, '*.png'))
    test_list = [(t, t.split('/')[-1]) for t in test_list]

    testset = ImageDataset(test_list, test_tfms)
    test_loader = DataLoader(testset, batch_size=args.test_bs, shuffle=False, num_workers=8, pin_memory=True)

    save_dict = torch.load(args.model_path)
    label2name = save_dict['label2name']

    network = get_resnet50(len(label2name))
    network.load_state_dict(save_dict['state_dict'])

    submission(network, test_loader, label2name, device)


if __name__ == '__main__':
    main()
