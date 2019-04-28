# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import glob
import os

import numpy as np


# split train and valid list
def split_dataset(root_path, split_ratio):
    train_path = os.path.join(root_path, 'train')
    train_labels = os.listdir(train_path)
    name2label = dict((j, i) for i, j in enumerate(train_labels))
    label2name = dict((j, i) for i, j in name2label.items())

    # split train and validation list
    train_list = []
    valid_list = []
    for i in train_labels:
        label = name2label[i]
        imgs_in_class = glob.glob(os.path.join(train_path, i, '*.png'))
        valid_idx = np.random.choice(np.arange(len(imgs_in_class)),
                                     size=int(len(imgs_in_class) * split_ratio),
                                     replace=False)
        train_idx = list(set(np.arange(len(imgs_in_class))) - set(valid_idx))
        train_list.extend([(imgs_in_class[idx], label) for idx in train_idx])
        valid_list.extend([(imgs_in_class[idx], label) for idx in valid_idx])

    return train_list, valid_list, label2name
