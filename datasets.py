# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, img_list, transforms):
        self.img_list = img_list
        self.transforms = transforms

    def __getitem__(self, item):
        img_path, label = self.img_list[item]
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.img_list)

