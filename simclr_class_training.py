# !/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'haoli1'
# Created by haoli on 2022/03/28.
# 对比学习的stage2,有监督学习的分类任务主训练代码，
# 主要是替换无监督学习的全连接层，更换loss函数

from simclr import SimCLRStage2
from simclr.modules import get_resnet
import torch,argparse,os
import torchvision
import argparse
from utils import yaml_config_hook
from torch.utils.data import DataLoader
from simclr.modules.transformations import TransformsClassImage


def train(args):

    # load dataset for train and eval
    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            root='dataset',
            train=True,
            split="unlabeled",
            download=True,
            transform=TransformsClassImage(size=args.image_size,train=True),
        )
        eval_dataset= torchvision.datasets.STL10(root='dataset', 
                                                train=False, 
                                                transform=TransformsClassImage(size=args.image_size,train=False), 
                                                download=True)


    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root='dataset',
            download=True,
            transform=TransformsClassImage(size=args.image_size),
        )
        eval_dataset= torchvision.datasets.CIFAR10(
                                                root='dataset', 
                                                train=False, 
                                                transform=TransformsClassImage(size=args.image_size,train=False), 
                                                download=True)

    elif args.dataset == "GIF2":
        train_dataset = torchvision.datasets.ImageFolder(
                        args.dataset_dir,
                        transform=TransformsClassImage(size=args.image_size,train=True),
        )
        eval_dataset = torchvision.datasets.ImageFolder(
                        args.valdata_dir,
                        transform=TransformsClassImage(size=args.image_size,train=False),
        )

    else:
        raise NotImplementedError

    train_sampler = None

    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    eval_data =  torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    # Initialization Model
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features 
    model =SimCLRStage2(encoder, args.projection_dim, n_features,num_class=len(train_dataset.classes)).to(args.device.type)
    model_path = os.path.join(args.unsupervised_model_path, "checkpoint_{}.tar".format(args.unsupervised_epochs))
    model.load_state_dict(torch.load(model_path, map_location=args.device.type),strict=False)

    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)

    # os.makedirs(args.supervised_model_path, exist_ok=True)

    for epoch in range(1,args.supervised_epochs+1):
        model.train()
        total_loss=0
        for batch, (data, target) in enumerate(train_data):
            data, target = data.to(args.device.type), target.to(args.device.type)
            pred = model(data)

            loss = loss_criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("epoch",epoch,"loss:", total_loss / len(train_dataset)*args.batch_size)
        with open(os.path.join(args.supervised_model_path, "stage2_loss.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset)*args.batch_size) + " ")

        if epoch % 2==0:
            torch.save(model.state_dict(), os.path.join(args.supervised_model_path, 'model_stage2_epoch' + str(epoch) + '.pth'))

            model.eval()
            with torch.no_grad():
                print("batch", " " * 1, "top1 acc", " " * 1, "top5 acc")
                total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
                for batch, (data, target) in enumerate(eval_data):
                    data, target = data.to(args.device.type), target.to(args.device.type)
                    pred = model(data)

                    total_num += data.size(0)
                    prediction = torch.argsort(pred, dim=-1, descending=True)
                    top1_acc = torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    top5_acc = torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_correct_1 += top1_acc
                    total_correct_5 += top5_acc

                    print("  {:02}  ".format(batch + 1), " {:02.3f}%  ".format(top1_acc / data.size(0) * 100),
                          "{:02.3f}%  ".format(top5_acc / data.size(0) * 100))

                print("all eval dataset:", "top1 acc: {:02.3f}%".format(total_correct_1 / total_num * 100),
                          "top5 acc:{:02.3f}%".format(total_correct_5 / total_num * 100))
                with open(os.path.join(args.supervised_model_path, "stage2_top1_acc.txt"), "a") as f:
                    f.write(str(total_correct_1 / total_num * 100) + " ")
                with open(os.path.join(args.supervised_model_path, "stage2_top5_acc.txt"), "a") as f:
                    f.write(str(total_correct_5 / total_num * 100) + " ")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SimCLR Supervised  learning")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    if not os.path.exists(args.supervised_model_path):
        os.makedirs(args.supervised_model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    train(args)

