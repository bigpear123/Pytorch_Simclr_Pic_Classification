# pytorch 图像对比学习分类任务训练fintune 代码

此项目主要是更快、更方便的进行图片分类模型finetune的过程，模型涵盖resnet、alexnet、squeezenet、vgg、densenet、inception、googlenet、shufflenet、mobilenet_v2、mobilenet_v3_large、mobilenet_v3_small、resnext50_32x4d、wide_resnet50_2等较为主流的图片分类模型，同时也支持**对比学习（simclr）的模型训练**、finetune，用户只需修改相关配置文件，即可启动训练并得到finetune 后的模型。

## 详细介绍

### 具体代码文件

#### 训练配置文件 config.yaml

可以在config目录下，上传需要的新的训练config.yaml 文件


用户主要是修改这个配置文件，进而修改相关训练参数

```
# 分布式的相关参数，但是单机训练不需要修改
nodes: 1 
gpus: 1 # 单机训练不需要修改，分布式在调试中
nr: 0 # 单机训练不需要修改
dataparallel: 0 # Use DataParallel instead of DistributedDataParallel
workers: 1

# 训练集和测试集地址
dataset_dir: "./datasets"
valdata_dir: "./dddd"

# 训练相关参数
seed: 2 # sacred handles automatic seeding when passed in the config
batch_size: 8
image_size: 224
start_epoch: 0
# 无监督学习训练的训练epoch数目
unsupervised_epochs: 100
# 有监督学习的训练数目
supervised_epochs: 100
dataset: "GIF2" # STL10
pretrain: True

# 模型的相关选项
resnet: "resnet18" # 支持resnet、alexnet、squeezenet、vgg、densenet、inception、googlenet、shufflenet、mobilenet_v2、mobilenet_v3_large、mobilenet_v3_small、resnext50_32x4d、wide_resnet50_2，对比学习这个选项是backbone函数
projection_dim: 64 # 目前没有用

# 损失函数相关选项
optimizer: "Adam" # or LARS (experimental)
weight_decay: 1.0e-6 # "optimized using LARS [...] and weight decay of 10−6"
temperature: 0.5 # see appendix B.7.: Optimal temperature under different batch sizes

# 保存模型相关选项
unsupervised_model_path: "save" # set to the directory containing `checkpoint_##.tar`
supervised_model_path: "class_model"
reload: False

# logistic regression options
logistic_batch_size: 8
logistic_epochs: 50

```
#### 主代码文件

```
  cv_train_main.py # resnet等主流图片分类模型训练启动文件
  simclr_main.py   # 对比学习第一阶段，无监督学习训练流程
  simclr_class_training.py  # 对比学习第二阶段，有监督学习训练流程
  predict.py # 对比学习训练模型推理代码
```
执行相关的训练只需要：

```
# 执行resnet等的分类训练
python3 cv_train_main.py 
# 执行对比学习的分类训练
## 首先执行无监督训练代码：
python3  simclr_main.py
## 其次执行监督学习训练代码
python3  simclr_class_training.py
```
也可以不修改config 文件直接传参数，例如：

```
python3 cv_train_main.py  --dataset GIF2

```

#### cnn_learning & simlar

这两个文件夹是包含了各个任务相关的模型加载和图片预处理的文件


## 代办

- [x] 对比学习无监督训练过程调通（22.03.06）
- [x] 3个以上的主流图像分类任务代码流程合并开发、调试通过（22.03.09）
- [x] 对比学习分类监督任务开发完成（22.03.28）
- [x] 对比学习的推理过程开发（22.03.28）
- [ ] 主流图像分类任务推理过程开发
- [ ] 镜像开发


