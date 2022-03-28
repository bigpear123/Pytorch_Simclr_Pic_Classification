# !/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'haoli1'
# Created by haoli on 2022/03/09.
# 常见cnn网络的的主训练代码

# base
import os
import numpy as np
import torch
import torchvision
import argparse
import torch.nn as nn
import copy

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# CNN 模型

from cnn_learning.modules import get_models
from cnn_learning.transformations import TransformsCNN
from model import load_optimizer, save_model
from utils import yaml_config_hook

def set_parameters_require_grad(model, is_fixed):
    #默认parameter.requires_grad = True
    #当采用固定预训练模型参数的方法进行训练时，将预训练模型的参数设置成不需要计算梯度
    if(is_fixed):
        for parameter in model.parameters():
            parameter.requires_grad = False
            
def init_model(args,is_fixed=False, pretrained=True):
    model = get_models(args.resnet, pretrained=True)
    #设置参数是否需要计算梯度
    #is_fixed=True表示模型参数不需要计算梯度更新, False表示模型参数需要finetune（需要计算梯度）
    set_parameters_require_grad(model, is_fixed=False)
    in_features = model.fc.in_features  #取出全连接层的输入特征维度
    #重新定义resnet18模型的全连接层,使其满足新的分类任务
    #此时模型的全连接层默认需要计算梯度
    model.fc = nn.Linear(in_features, args.num_classes) 
    return model

def test(model, test_loader, loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for images, labels in test_loader:
        images = images.to(args.device)
        labels = labels.to(args.device)
        with torch.no_grad():
            outputs = model(images)
            loss = loss_func(outputs, labels)
            
        _, predicts = torch.max(outputs, 1)
        
        loss_val += loss.item() * images.size(0)
        corrects += torch.sum(predicts.view(-1) == labels.view(-1)).item()
        
    test_loss = loss_val / len(test_loader.dataset)
    test_acc = corrects / len(test_loader.dataset)
    
    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))
        
    return test_acc
def train(args, train_loader, test_loader, model, criterion, optimizer, writer):
    #初始化最好的验证准确率
    best_val_acc = 0.0
    args.global_step = 0
    args.current_epoch = 0
    lr = optimizer.param_groups[0]["lr"]
    #初始化最好的模型参数，采用deepcopy为防止优化过程中修改到best_model_params
    best_model_params = copy.deepcopy(model.state_dict())
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0
        for  step, (images, labels) in enumerate(train_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            #找出输出的最大概率所在的为
            #二分类中：如果第一个样本输出的最大值出现在第0为，则其预测值为0
            _, predicts = torch.max(outputs, 1) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #loss.item()为一个batch的平均loss的值
            #images.size(0)为当前batch中有多少样本量
            #loss.item() * images.size(0)表示一个batch的总loss值
            loss_val += loss.item() * images.size(0)
            
            #view(-1)表示将tensor resize成一个维度为[batch_size]的tensor
            #计算预测值与标签值相同的数量
            corrects += torch.sum(predicts.view(-1) == labels.view(-1)).item()

            if args.nr == 0 and step % 50 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")
            if args.nr == 0:
                writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
                args.global_step += 1
                
        
        #计算每个epoch的平均loss
        train_loss = loss_val / len(train_loader.dataset)
        #预测准确的数量除以总的样本量即为准确率
        train_acc = corrects / len(train_loader.dataset)
        
        print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))
        
        #调用测试
        test_acc = test(model, test_loader, criterion)
        #根据测试准确率跟新最佳模型的参数
        if(best_val_acc < test_acc):
            best_val_acc = test_acc
            best_model_params = copy.deepcopy(model.state_dict())

        if args.nr == 0:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            args.current_epoch += 1
    #将模型的最优参数载入模型    
    model.load_state_dict(best_model_params)
    return model



def main(args):
 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsCNN(size=args.image_size),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsCNN(size=args.image_size),
        )
    elif args.dataset == "OTHER":
        train_dataset = torchvision.datasets.ImageFolder(
                        args.dataset_dir,
                        transform=TransformsCNN(size=args.image_size),
        )
    else:
        raise NotImplementedError

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )
    test_dataset = torchvision.datasets.ImageFolder(args.valdata_dir,
                                      torchvision.transforms.Compose([
                                      torchvision.transforms.Resize(args.image_size),
                                        torchvision.transforms.CenterCrop(args.image_size),
                                          torchvision.transforms.ToTensor()
                                          ]))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,)

    # initialize model
    model = init_model(args)
    # 支持加载模型重新训练
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    
    model = model.to(args.device)
    # optimizer / loss

    optimizer, scheduler = load_optimizer(args, model)
    # 一般的分类模型就使用交叉熵函数
    criterion = nn.CrossEntropyLoss()

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter() 
    
    # args, train_loader, test_loader, model, criterion, optimizer, writer
    model = train(args, train_loader, test_loader, model,criterion, optimizer,writer)
    torch.save(model.state_dict(),"good_model.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PicClassification")
    config = yaml_config_hook("./config/config_cnn.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = 1

    main(args)
