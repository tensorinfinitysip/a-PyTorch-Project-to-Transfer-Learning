# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn
from torchvision.models import *


def get_resnet50(classes, pretrain=True):
    if pretrain:
        net = resnet50(pretrained=True)
    else:
        net = resnet50()
    net.avgpool = nn.AdaptiveAvgPool2d(1)
    net.fc = nn.Linear(net.fc.in_features, classes)
    return net
