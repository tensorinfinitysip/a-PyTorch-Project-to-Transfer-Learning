# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import time

import numpy as np
import torch
import torchvision.transforms as T
from torch.backends import cudnn
from torch.utils.data import DataLoader

from datasets import ImageDataset
from model import *
from utils.create_data_lists import split_dataset
from utils.lr_scheduler import LRScheduler
from utils.meter import AverageValueMeter
from utils.serialization import mkdir_if_missing, save_checkpoint
from utils.layer_group import flatten_model

def main():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--save_dir', type=str, default='logs/tmp', help='save model directory')
    # transforms
    parser.add_argument('--train_size', type=int, default=[224], nargs='+', help='train image size')
    parser.add_argument('--test_size', type=int, default=[224, 224], nargs='+', help='test image size')
    parser.add_argument('--h_filp', action='store_true', help='do horizontal flip')
    # dataset
    parser.add_argument('--dataset_dir', type=str, default='datasets', help='datasets path')
    parser.add_argument('--valid_pect', type=float, default=0.2, help='validation percent split from train')
    parser.add_argument('--train_bs', type=int, default=64, help='train images per batch')
    parser.add_argument('--test_bs', type=int, default=128, help='test images per batch')
    # training
    parser.add_argument('--no_gpu', action='store_true', help='whether use gpu')
    parser.add_argument('--gpus', type=str, default='0', help='gpus to use in training')
    parser.add_argument('--opt_func', type=str, default='Adam', help='optimizer function')
    parser.add_argument('--lr', type=float, default=0.1, help='base learning rate')
    parser.add_argument('--steps', type=int, default=(60, 90), nargs='+', help='learning rate decay strategy')
    parser.add_argument('--factor', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, help='training momentum')
    parser.add_argument('--max_epoch', type=int, default=120, help='number of training epochs')
    parser.add_argument('--log_interval', type=int, default=50, help='intermediate printing')
    parser.add_argument('--save_step', type=int, default=20, help='save model every save_step')

    args = parser.parse_args()

    mkdir_if_missing(args.save_dir)
    log_path = os.path.join(args.save_dir, 'log.txt')
    with open(log_path, 'w') as f:
        f.write('{}'.format(args))

    device = "cuda:{}".format(args.gpus) if not args.no_gpu else "cpu"
    if not args.no_gpu:
        cudnn.benchmark = True

    # define train transforms and test transforms
    totensor = T.ToTensor()
    normalize = T.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])
    train_tfms = list()
    train_size = args.train_size[0] if len(args.train_size) == 1 else args.train_size
    train_tfms.append(T.RandomResizedCrop(train_size))
    if args.h_filp:
        train_tfms.append(T.RandomHorizontalFlip())
    train_tfms.append(totensor)
    train_tfms.append(normalize)
    train_tfms = T.Compose(train_tfms)

    test_tfms = list()
    test_size = (args.test_size[0], args.test_size[0]) if len(args.test_size) == 1 else args.test_size
    test_tfms.append(T.Resize(test_size))
    test_tfms.append(totensor)
    test_tfms.append(normalize)
    test_tfms = T.Compose(test_tfms)

    # get dataloader
    train_list, valid_list, label2name = split_dataset(args.dataset_dir, args.valid_pect)
    trainset = ImageDataset(train_list, train_tfms)
    validset = ImageDataset(valid_list, test_tfms)

    train_loader = DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(validset, batch_size=args.test_bs, shuffle=False, num_workers=8, pin_memory=True)

    # define network
    net = get_resnet50(len(label2name), pretrain=True)
    # layer_groups = [nn.Sequential(*flatten_model(net))]

    # define loss
    ce_loss = nn.CrossEntropyLoss()


    # define optimizer and lr scheduler
    if args.opt_func == 'Adam':
        optimizer = getattr(torch.optim, args.opt_func)(net.parameters(), weight_decay=args.wd)
    else:
        optimizer = getattr(torch.optim, args.opt_func)(net.parameters(), weight_decay=args.wd, momentum=args.momentum)
    lr_scheduler = LRScheduler(base_lr=args.lr, step=args.steps, factor=args.factor)

    train(
        args=args,
        network=net,
        train_data=train_loader,
        valid_data=valid_loader,
        optimizer=optimizer,
        criterion=ce_loss,
        lr_scheduler=lr_scheduler,
        device=device,
        log_path=log_path,
        label2name=label2name,
    )


def train(args, network, train_data, valid_data, optimizer, criterion, lr_scheduler, device, log_path, label2name):
    network = network.to(device)
    best_test_acc = -np.inf
    losses = AverageValueMeter()
    acces = AverageValueMeter()
    for epoch in range(args.max_epoch):
        losses.reset()
        acces.reset()
        network.train()

        lr = lr_scheduler.update(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print_str = 'Epoch [%d] learning rate update to %.3e' % (epoch, lr)
        print(print_str)
        with open(log_path, 'a') as f: f.write(print_str + '\n')
        tic = time.time()
        btic = time.time()
        for i, data in enumerate(train_data):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            scores = network(imgs)
            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.add(loss.item())
            acc = (scores.max(1)[1] == labels.long()).float().mean()
            acces.add(acc.item())

            if (i + 1) % args.log_interval == 0:
                loss_mean = losses.value()[0]
                acc_mean = acces.value()[0]
                print_str = 'Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\tloss=%f\tacc=%f' % (
                    epoch, i + 1, train_data.batch_size * args.log_interval / (time.time() - btic),
                    loss_mean, acc_mean)
                print(print_str)
                with open(log_path, 'a') as f: f.write(print_str + '\n')
                btic = time.time()

        loss_mean = losses.value()[0]
        acc_mean = acces.value()[0]
        throughput = int(train_data.batch_size * len(train_data) / (time.time() - tic))

        print_str = '[Epoch %d] training: loss=%f\tacc=%f speed: %d samples/sec\ttime cost: %.3f' % (
            epoch, loss_mean, acc_mean, throughput, time.time() - tic)
        print(print_str)
        with open(log_path, 'a') as f:
            f.write(print_str + '\n')

        is_best = False
        if valid_data is not None:
            test_acc = test(network, valid_data, device)
            print_str = '[Epoch %d] test acc: %f' % (epoch, test_acc)
            print(print_str)
            with open(log_path, 'a') as f:
                f.write(print_str + '\n')
            is_best = test_acc > best_test_acc
            if is_best:
                best_test_acc = test_acc
        state_dict = network.state_dict()
        if (epoch + 1) % args.save_step == 0:
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch + 1,
                'label2name': label2name,
            }, is_best=is_best, save_dir=os.path.join(args.save_dir, 'models'), filename='model' + '.pth')


def test(network, test_data, device):
    num_correct = 0
    num_imgs = 0
    network.eval()
    for data in test_data:
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            scores = network(imgs)
        num_correct += (scores.max(1)[1] == labels).float().sum().item()
        num_imgs += imgs.shape[0]
    return num_correct / num_imgs


if __name__ == '__main__':
    main()
