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
        #########################################################################
        # TODO:
        # 根据 item 实现对应数据和label的读入 同时对图像进行预处理和数据增强
        #
        # 提示: 可以使用 PIL 从硬盘中读入图片 并且记住要转换成 RGB 的模式
        # 同时注意类初始化中的定义
        #########################################################################
        pass
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def __len__(self):
        return len(self.img_list)

