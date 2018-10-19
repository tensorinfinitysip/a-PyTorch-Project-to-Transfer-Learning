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

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from utils.augmentor import RandomErasing


def get_data_provider(opt) -> tuple:
    random_mirror = opt.aug.get('random_mirror', False)
    random_erasing = opt.aug.get('random_erasing', False)

    tfms = [T.ToTensor(), T.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])]
    train_aug = list()
    train_aug.append(T.RandomResizedCrop(opt.aug.resize_size))

    if random_mirror:
        train_aug.append(T.RandomHorizontalFlip())
        train_aug.append(T.RandomVerticalFlip())
    train_aug.extend(tfms)
    if random_erasing:
        train_aug.append(RandomErasing())

    train_aug = T.Compose(train_aug)

    test_aug = list()
    test_aug.append(T.Resize((opt.aug.resize_size,) * 2))
    test_aug.extend(tfms)
    test_aug = T.Compose(test_aug)

    train_set = ImageFolder(opt.dataset.train_path, train_aug)

    valid_set = ImageFolder(opt.dataset.valid_path, test_aug)

    train_loader = DataLoader(train_set, opt.train.batch_size, shuffle=True,
                              num_workers=opt.train.workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, opt.train.batch_size, shuffle=False,
                              num_workers=opt.train.workers, pin_memory=True)
    return train_loader, valid_loader


def get_test_provider(opt):
    train_set = ImageFolder(opt.dataset.train_path, None)
    tfms = [T.ToTensor(), T.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])]
    test_aug = list()
    test_aug.append(T.Resize((opt.aug.resize_size,) * 2))
    test_aug.extend(tfms)
    test_aug = T.Compose(test_aug)

    test_set = TestSet(opt.dataset.test_path, test_aug)
    test_loader = DataLoader(test_set, opt.train.batch_size, num_workers=opt.train.workers, pin_memory=True)
    idx_to_class = dict((j, i) for i, j in train_set.class_to_idx.items())
    return test_loader, idx_to_class


class TestSet(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.img_list = os.listdir(root)
        self.transform = transform

    def __getitem__(self, item):
        fname = self.img_list[item]
        img_path = os.path.join(self.root, fname)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname

    def __len__(self):
        return len(self.img_list)
