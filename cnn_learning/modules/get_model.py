# !/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'haoli1'
# Created by haoli on 2022/03/05.
# 倒入的基础模型代码

import torchvision.models as models



def get_models(name, pretrained=False):
    resnet = {
        "resnet18": models.resnet18(pretrained=pretrained),
        "resnet50": models.resnet50(pretrained=pretrained),
        "alexnet": models.alexnet(pretrained=True),
        "squeezenet" : models.squeezenet1_0(pretrained=True),
        "vgg16": models.vgg16(pretrained=True),
        "densenet" : models.densenet161(pretrained=True),
        "inception" : models.inception_v3(pretrained=True),
        "googlenet" : models.googlenet(pretrained=True),
        "shufflenet" : models.shufflenet_v2_x1_0(pretrained=True),
        "mobilenet_v2" : models.mobilenet_v2(pretrained=True),
        "mobilenet_v3_large" : models.mobilenet_v3_large(pretrained=True),
        "mobilenet_v3_small" : models.mobilenet_v3_small(pretrained=True),
        "resnext50_32x4d" : models.resnext50_32x4d(pretrained=True),
        "wide_resnet50_2" : models.wide_resnet50_2(pretrained=True),
    }
    if name not in resnet.keys():
        raise KeyError(f"{name} is not a valid Model version")
    return resnet[name]
