# !/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'haoli1'
# Created by haoli on 2022/03/05.
# 对比学习的loss函数
# 代码来源都是：https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py 
# 增加一些备注和个人理解
# 参考git：https://github.com/Spijkervet/SimCLR

import torch
import torch.nn as nn
from .gather import GatherLayer

# NT-Xent 全称：(the normalized temperature-scaled cross entropy loss) 
# "归一化的带温度交叉熵"
# 主要是 postive pair 和 negative pair
# 一个batch是由N张图片通过两组不同的增强变成2N张并且穿插排列
class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature #温度系数,温度小于 1，softmax 预测变得更有信心, T 大于 1，则预测的可信度较低（也称为“软”概率）
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        # 因为主对角元素值（自己和自己的度量值）没用
        # 用mask掩膜对原来shape为[2*Batch_size,2*Batch_size]的sim_matrix做处理（
        # 把sim_matrix的主对角线元素删掉，然后形成shape为[2*Batch_size,2*Batch_size-1]的sim_matrix）
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2) # 余旋距离，计算相似性


    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size # world_size 是 分布式训练中的参数，这里是单机代码，为1，可以忽略
        mask = torch.ones((N, N), dtype=bool) # [2 * batch_size, 2 * batch_size] 数组初始化为1
        mask = mask.fill_diagonal_(0) # fill_diagonal 对角线填充为0
        for i in range(batch_size * world_size): 
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # zi,zj 是simclr 模型的输出的embedding ，是postive  pair 
        N = 2 * self.batch_size * self.world_size
        #

        z = torch.cat((z_i, z_j), dim=0) 

        # z.unsqueeze(1):[,1,]
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature # batch size内两两之间的距离
        
        sim_i_j = torch.diag(sim, self.batch_size * self.world_size) #取对角线元素
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long() #[batch_size,2*batch_size]
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss



