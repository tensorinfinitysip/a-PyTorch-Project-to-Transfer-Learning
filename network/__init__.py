# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch import nn
from torchvision import models


def resnet50(classes, pretrain=True):
    if pretrain:
        net = models.resnet50(pretrained=True)
    else:
        net = models.resnet50()
    net.avgpool = nn.AdaptiveAvgPool2d(1)
    net.fc = nn.Linear(net.fc.in_features, classes)
    return net
