这是一个 [PyTorch]() 的教程: a PyTorch Tutorial to Transfer Learning

这是 [a series of projects]() 中的第一个项目，从这个项目中我们会学习到如何使用迁移学习完成这个很棒的图像分类项目。

需要大家了解 PyTorch 的基本知识，同时要掌握卷积神经网络的知识。

项目使用 `PyTorch 1.0` 和 `python3.7`

# 目录

[**Objective**](https://github.com/L1aoXingyu/a-PyTorch-Tutorial-to-Transfer-Learning#objective)

[**Concepts**](https://github.com/L1aoXingyu/a-PyTorch-Tutorial-to-Transfer-Learning#concepts)

[**Overview**](https://github.com/L1aoXingyu/a-PyTorch-Tutorial-to-Transfer-Learning#overview)

[**Implementation**](https://github.com/L1aoXingyu/a-PyTorch-Tutorial-to-Transfer-Learning#implementation)

[**Training**](https://github.com/L1aoXingyu/a-PyTorch-Tutorial-to-Transfer-Learning#training)

[**Evaluation**](https://github.com/L1aoXingyu/a-PyTorch-Tutorial-to-Transfer-Learning#evaluation)

[**Inference**](https://github.com/L1aoXingyu/a-PyTorch-Tutorial-to-Transfer-Learning#inference)


# Objective
我们需要训练一个卷积神经网络来识别不同的植物种类。

在这个项目中，我们主要使用迁移学习和 fine-tune 的技术。

下面是几个例子，训练好的网络在测试集上的结果

<div align=center>
<img src='assets/demo1.png' width='800'>
</div>

<div align=center>
<img src='assets/demo2.png' width='800'>
</div>

<div align=center>
<img src='assets/demo3.png' width='800'>
</div>

# Concepts

- **图像分类** 给定任何一张图片，这个任务的目标是在给定的候选label中预测一个概率最大的label或者是预测一个概率分布。在2012年以前，图像分类基本都是基于传统的图像处理方法，比如通过梯度算子和颜色直方图等信息手动提取特征，接着使用SVM等线性分类器进行分类。到了2012年深度学习方法横空出世，超越了传统方法非常多的分数，从此之后深度学习的方法逐步开始统治CV中应用。在图像分类领域，越来越多的卷积网络结构被提出来，比如VGG，InceptionNet，ResNet，DenseNet等等，同时这些网络都作为backbone应用到了检测和分割等任务上。

<div align=center>
<img src='assets/cifar10.png' width='600'>
</div>

- **softmax** 在数学上softmax也被称为归一化指数函数，是logistic函数从2维到高维的一个推广，能将任何实数k维向量z归一化到另外一个k维实向量$\sigma(z)$

- 过拟合

- transfer leanring

- learning rate decay

# Overview


# Implementation


# Training

## 数据下载
通过[比赛界面](https://www.kaggle.com/c/plant-seedlings-classification/data)根据图片中的显示进行数据下载

<div align=center>
<img src='https://ws4.sinaimg.cn/large/006tNbRwly1fwdo7019xfj31kw13owgy.jpg' width='800'>
</div>

然后在项目的根目录中创建`datasets`文件夹，将下载好`train.zip`和`test.zip`文件放入`datasets`中

## 训练 baseline
运行下面的代码

```bash
python train.py 
```

就可以进行baseline训练, 这次提供的baseline是模块化的代码，所有的配置文件都在`core/config.py`中，大家可以自己去查看，同时因为数据和模型比较大，这次代码只支持GPU训练，在训练过程中，会自动创建`checkpoints`文件夹，训练的模型会自动保存在`checkpoints`中。


# Evaluation

## 提交结果
训练完成 baseline 之后，我们的模型会保存在 `checkpoints` 中，我们可以 load 我们想要的模型，进行结果的提交，运行下面的代码

```
python submission.py --model_path='logs/tmp/model/model_best.pth' 
```

我们会在本地创建一个预测的结果 `submission.csv`，我们将这个文件提交到 kaggle，可以得到类似下面的比赛结果。

<div align=center>
<img src='https://ws1.sinaimg.cn/large/006tNbRwly1fwdozwppbuj31iq0c2jry.jpg' width='800'>
</div>


# Inference