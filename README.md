# Kaggle Plant Seeding Classification 比赛

<div align=center>
<img src='https://ws2.sinaimg.cn/large/006tNbRwly1fwdpjbncupj31i40ew0vk.jpg' width='500'>
</div>

## 项目介绍

在本项目中，大家会学会使用成熟的卷积网络架构进行 kaggle 植物种子分类，不同于上一个比赛，在这个比赛中，数据量更大，图像分辨率更高，网络结构更深，给的baseline更加符合项目的模块化要求，大家可以使用任意的网络结构和预训练模型进行训练和测试，通过该项目，我们不仅能够学会调参，还能够学会如何写出模块化的代码。

## 项目下载

打开终端，运行

```bash
git clone https://github.com/L1aoXingyu/kaggle-plant-seeding.git
```

进行项目的下载或者通过网页版下载，接着进入到项目所在的目录，运行

```bash
pip3 install -r requirements.txt
```

进行依赖的安装。

## 数据下载
通过[比赛界面](https://www.kaggle.com/c/plant-seedlings-classification/data)根据图片中的显示进行数据下载

<div align=center>
<img src='https://ws4.sinaimg.cn/large/006tNbRwly1fwdo7019xfj31kw13owgy.jpg' width='500'>
</div>

然后在项目的根目录中创建`data`文件夹，将下载好`train.zip`和`test.zip`文件放入`data`中，接着运行下面的命令来得到预处理之后的数据

```bash
cd data;
unzip -q train.zip; cp -r train train_valid;
unzip -q test.zip
cd ..; python3 preprocess.py
```

## 训练 baseline
运行下面的代码

```bash
python3 train.py 
```

就可以进行baseline训练, 这次提供的baseline是模块化的代码，所有的配置文件都在`core/config.py`中，大家可以自己去查看，同时因为数据和模型比较大，这次代码只支持GPU训练，在训练过程中，会自动创建`checkpoints`文件夹，训练的模型会自动保存在`checkpoints`中。

## 提交结果
训练完成 baseline 之后，我们的模型会保存在 `checkpoints` 中，我们可以 load 我们想要的模型，进行结果的提交，运行下面的代码

```
python3 submission.py --model_path='checkpoints/model_best.pth.tar' 
```

我们会在本地创建一个预测的结果 `submission.csv`，我们将这个文件提交到 kaggle，可以得到类似下面的比赛结果。

<div align=center>
<img src='https://ws1.sinaimg.cn/large/006tNbRwly1fwdozwppbuj31iq0c2jry.jpg' width='800'>
</div>

希望大家能够尝试着自己调一调参，试一下模型融合的方式，得到更好的结果。