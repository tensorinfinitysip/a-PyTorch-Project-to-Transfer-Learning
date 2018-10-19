# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import time

import numpy as np
import torch

from utils.meter import AverageValueMeter
from utils.serialization import save_checkpoint


class Solver(object):
    def __init__(self, opt, net):
        self.opt = opt
        self.net = net
        self.loss = AverageValueMeter()
        self.acc = AverageValueMeter()

    def fit(self, train_data, test_data, optimizer, criterion, lr_scheduler):
        best_test_acc = -np.inf
        for epoch in range(self.opt.train.num_epochs):
            self.loss.reset()
            self.acc.reset()
            self.net.train()

            lr = lr_scheduler.update(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logging.info('Epoch [%d] learning rate update to %.3e' % (epoch, lr))
            tic = time.time()
            btic = time.time()
            for i, data in enumerate(train_data):
                imgs, labels = data
                labels = labels.cuda()
                scores = self.net(imgs)
                loss = criterion(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.loss.add(loss.item())
                acc = (scores.max(1)[1] == labels.long()).float().mean()
                self.acc.add(acc.item())

                if self.opt.misc.log_interval and not (i + 1) % self.opt.misc.log_interval:
                    loss_mean = self.loss.value()[0]
                    acc_mean = self.acc.value()[0]
                    logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\tloss=%f\t'
                                 'acc=%f' % (
                                     epoch, i + 1, train_data.batch_size * self.opt.misc.log_interval / (time.time() - btic),
                                     loss_mean, acc_mean))
                    btic = time.time()

            loss_mean = self.loss.value()[0]
            acc_mean = self.acc.value()[0]
            throughput = int(train_data.batch_size * len(train_data) / (time.time() - tic))

            logging.info('[Epoch %d] training: loss=%f\tacc=%f' % (
                epoch, loss_mean, acc_mean))
            logging.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f' % (epoch, throughput, time.time() - tic))

            is_best = False
            if test_data is not None and self.opt.misc.eval_step and not (epoch + 1) % self.opt.misc.eval_step:
                test_acc = self.test_func(test_data)
                logging.info('[Epoch %d] test acc: %f' % (epoch, test_acc))
                is_best = test_acc > best_test_acc
                if is_best:
                    best_test_acc = test_acc
            state_dict = self.net.module.state_dict()
            if not (epoch + 1) % self.opt.misc.save_step:
                save_checkpoint({
                    'state_dict': state_dict,
                    'epoch': epoch + 1,
                }, is_best=is_best, save_dir=self.opt.misc.save_dir,
                    filename='model' + '.pth.tar')

    def test_func(self, test_data) -> float:
        num_correct = 0
        num_imgs = 0
        self.net.eval()
        for data in test_data:
            imgs, labels = data
            labels = labels.cuda()
            with torch.no_grad():
                scores = self.net(imgs)
            num_correct += (scores.max(1)[1] == labels).float().sum().item()
            num_imgs += imgs.shape[0]
        return num_correct / num_imgs
