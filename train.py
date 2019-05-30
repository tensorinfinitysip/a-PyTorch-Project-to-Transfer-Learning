# encoding: utf-8
"""
@author:  l1aoxingyu
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
    # dataset
    parser.add_argument('--dataset_dir', type=str, default='datasets', help='datasets path')
    parser.add_argument('--valid_pect', type=float, default=0.2, help='validation percent split from train')
    parser.add_argument('--train_bs', type=int, default=64, help='train images per batch')
    parser.add_argument('--test_bs', type=int, default=128, help='test images per batch')
    # training
    parser.add_argument('--no_gpu', action='store_true', help='whether use gpu')
    parser.add_argument('--gpus', type=str, default='0', help='gpus to use in training')
    parser.add_argument('--log_interval', type=int, default=20, help='intermediate printing')
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
    train_tfms.append(T.RandomResizedCrop((224, 224)))
    train_tfms.append(T.RandomHorizontalFlip())
    train_tfms.append(totensor)
    train_tfms.append(normalize)
    train_tfms = T.Compose(train_tfms)

    test_tfms = list()
    test_tfms.append(T.Resize((224, 224)))
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

    # define loss
    ce_loss = nn.CrossEntropyLoss()

    # base_params = list(net.parameters())[:-2]
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    # define optimizer and lr scheduler
    # if args.opt_func == 'Adam':
    #     optimizer = getattr(torch.optim, args.opt_func)(net.parameters(), weight_decay=args.wd)
    # else:
    #     optimizer = getattr(torch.optim, args.opt_func)(net.parameters(), weight_decay=args.wd, momentum=args.momentum)

    train(
        args=args,
        network=net,
        train_data=train_loader,
        valid_data=valid_loader,
        optimizer=optimizer,
        criterion=ce_loss,
        device=device,
        log_path=log_path,
        label2name=label2name,
    )


def train(args, network, train_data, valid_data, optimizer, criterion, device, log_path, label2name):
    lr_scheduler = LRScheduler(base_lr=0.01, step=(30, 60), factor=0.1)
    network = network.to(device)
    best_test_acc = -np.inf
    losses = AverageValueMeter()
    acces = AverageValueMeter()
    for epoch in range(120):
        losses.reset()
        acces.reset()
        network.train()

        lr = lr_scheduler.update(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print_str = 'Epoch [%d] learning rate update to %.3e' % (epoch, lr)
        # print(print_str)
        # with open(log_path, 'a') as f: f.write(print_str + '\n')
        tic = time.time()
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
                print_str = 'Epoch[%d] Batch [%d]\tloss=%f\tacc=%f' % (
                    epoch, i + 1, loss_mean, acc_mean)
                print(print_str)
                with open(log_path, 'a') as f: f.write(print_str + '\n')
                btic = time.time()

        loss_mean = losses.value()[0]
        acc_mean = acces.value()[0]

        print_str = '[Epoch %d] Training: loss=%f\tacc=%f\ttime cost: %.3f' % (
            epoch, loss_mean, acc_mean, time.time() - tic)
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
