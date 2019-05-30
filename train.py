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
from utils.create_data_lists import split_dataset
from utils.meter import AverageValueMeter
from utils.serialization import mkdir_if_missing, save_checkpoint


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
    parser.add_argument('--max_epoch', type=int, default=120, help='number of epochs for training')
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

    #########################################################################
    # TODO:
    # 定义训练集的数据增强操作和验证集的数据增强操作
    # 对于训练集来讲 最简单的数据增强是随机 resize 水平翻转 当然可以
    # 使用更多的数据增强操作
    # 对于验证集来讲 只需要 resize 到固定大小即可
    #
    # 提示：可以查看 torchvision.transforms 中的函数来实现数据增强
    # 别要忘记最好要将图片转换成 Tensor 同时用 ImageNet 的均值和方差做标准化
    #########################################################################
    pass
    train_tfms = T.Compose([])
    test_tfms = T.Compose([])
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################


    # get dataloader
    train_list, valid_list, label2name = split_dataset(args.dataset_dir, args.valid_pect)
    trainset = ImageDataset(train_list, train_tfms)
    validset = ImageDataset(valid_list, test_tfms)

    train_loader = DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(validset, batch_size=args.test_bs, shuffle=False, num_workers=0, pin_memory=True)

    #########################################################################
    # TODO:
    # 定义模型，可以使用 torchvision.models 里面定义好的模型 如 resnet18
    # 可以使用在 ImageNet 上预训练的模型 特别注意要修改最后一层的全连接层参数
    #
    # 根据问题 可以定义交叉熵作为损失函数 具体的函数名可以查看文档
    #
    # 定义网络的优化器 可以使用 SGD 也可以用 Adam 同时
    # 可以考虑固定住前面的预训练部分 也可以让他们和全连接层一起训练
    #
    # 提示：遇到问题要学会查阅文档 同时也可以查看官方教程
    #########################################################################
    pass
    net = None
    loss_func = None
    optimizer = None
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################


    train(
        args=args,
        network=net,
        train_data=train_loader,
        valid_data=valid_loader,
        optimizer=optimizer,
        criterion=loss_func,
        device=device,
        log_path=log_path,
        label2name=label2name,
    )


def train(args, network, train_data, valid_data, optimizer, criterion, device, log_path, label2name):
    network = network.to(device)
    best_test_acc = -np.inf
    losses = AverageValueMeter()
    acces = AverageValueMeter()
    for epoch in range(args.max_epoch):
        losses.reset()
        acces.reset()
        network.train()

        tic = time.time()
        for i, data in enumerate(train_data):
            imgs, labels = data
            #########################################################################
            # TODO:
            # 定义模型的训练逻辑 实现一个 batch 的数据的前向传播 反向传播和参数更新
            # 1. 将数据放到 GPU 上
            # 2. 将数据输入网络实现前向传播
            # 3. 根据损失函数计算交叉熵
            # 4. 将参数的梯度归 0
            # 5. 通过反向传播计算参数的梯度
            # 6. 进行参数的更新
            # 7. 计算 batch 训练数据预测的准确率
            #########################################################################
            pass
            loss = None
            acc = None
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            losses.add(loss.item())
            acces.add(acc.item())

            if (i + 1) % args.log_interval == 0:
                loss_mean = losses.value()[0]
                acc_mean = acces.value()[0]
                print_str = 'Epoch[%d] Batch [%d]\tloss=%f\tacc=%f' % (
                    epoch, i + 1, loss_mean, acc_mean)
                print(print_str)
                with open(log_path, 'a') as f: f.write(print_str + '\n')

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
        #########################################################################
        # TODO:
        # 定义模型的测试逻辑
        # 1. 将图片和标签放到 GPU 上
        # 2. 在不追踪梯度的情况下实现模型的前向传播 使用 torch.no_grad()
        # 3. 计算预测正确的样本数量
        #########################################################################
        pass
        #########################################################################
        #                       END OF YOUR CODE
        #########################################################################
    return num_correct / num_imgs


if __name__ == '__main__':
    main()
