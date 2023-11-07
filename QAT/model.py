# -*- coding:utf-8 -*-
# Author : Yao Huang
# Date : 2023.10.11

import torch.nn as nn

from torch.quantization import QuantStub, DeQuantStub

class depthwise_separable_conv(nn.Module):
    '''
    Depthwise Separable Convolution with Batch Normlization and ReLU 
    '''
    def __init__(self, ch_in, ch_out, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        # [B, C_in, H_in, W_in] => [B, C_in, H_out, W_out]
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size = kernel_size, groups = ch_in)
        self.bn1 = nn.BatchNorm2d(ch_in)
        self.relu1 = nn.ReLU()
        # [B, C_in, H_out, W_out] => [B, C_out, H_out, W_out]
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size = 1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.point_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
    
    
class RexKWSModel(nn.Module):
    '''
    REEXEN KWS model.
    Special parameters below for CIFAR10 exam.
    '''
    def __init__(self):
        super(RexKWSModel, self).__init__()
        # [-1, 3, 32, 32] => [-1, 64, 28, 28]
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 5)
        self.relu = nn.ReLU()
        self.layer = nn.ModuleList([])
        # [-1, 64, 28, 28] => [-1, 64, 20, 20]
        for i in range(4):
            self.layer.append(depthwise_separable_conv(64, 64, 3))
        # [-1, 64, 20, 20] => [-1, 64, 4, 4]
        self.avg_pool = nn.AvgPool2d(kernel_size = 8, stride = 4)
        # [-1, 64, 9, 9] => [-1, 1024]
        self.flatten = nn.Flatten()
        # [-1, 1024] => [-1, 10]
        self.linear = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        for module in self.layer:
            x = module(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x    
    

class RexKWSModelQuant(nn.Module):
    '''
    REEXEN KWS model with quantization.
    Special parameters below for CIFAR10 exam.
    '''
    def __init__(self):
        super(RexKWSModelQuant, self).__init__()
        self.quant = QuantStub()
        self.backbone = RexKWSModel()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.backbone(x)
        x = self.dequant(x)
        return x