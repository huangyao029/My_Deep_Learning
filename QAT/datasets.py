# -*- coding:utf-8 -*-
# Author : Yao Huang
# Date : 2023.10.12

import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataset():
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    trainset = torchvision.datasets.CIFAR10(
        root = './data',
        train = True,
        download = True,
        transform = transform
    )
    
    testset = torchvision.datasets.CIFAR10(
        root = './data',
        train = False,
        download = True,
        transform = transform
    )
    
    return trainset, testset