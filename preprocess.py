# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import shutil

import numpy as np


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


train_img_path = 'data/train'
train_labels = os.listdir('data/train')
num_per_class = []
for i in train_labels:
    num_per_class.append(len(os.listdir(os.path.join(train_img_path, i))))
num_valid = int(min(num_per_class) * 0.2)


for i in train_labels:
    all_imgs = os.listdir(os.path.join(train_img_path, i))
    valid_imgs = np.random.choice(all_imgs, size=num_valid, replace=False)
    mkdir_if_not_exist(['data', 'valid', i])
    for img in valid_imgs:
        shutil.move(os.path.join('data/train', i, img), os.path.join('data/valid', i, img))
