# -*- coding:utf-8 -*-
# Author : Yao Huang
# Date : 2023.10.11

import os
import datetime
import torch
import torch.nn as nn

from datasets import get_cifar10_dataset
from model import RexKWSModelQuant

def train():
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    quant_model = RexKWSModelQuant()
    quant_model.eval()
    
    # op fuse
    op_fused_lst =  [
        ['backbone.conv1', 'backbone.relu'],
        ['backbone.layer.0.depth_conv', 'backbone.layer.0.bn1', 'backbone.layer.0.relu1'],
        ['backbone.layer.0.point_conv', 'backbone.layer.0.bn2', 'backbone.layer.0.relu2'],
        ['backbone.layer.1.depth_conv', 'backbone.layer.1.bn1', 'backbone.layer.1.relu1'],
        ['backbone.layer.1.point_conv', 'backbone.layer.1.bn2', 'backbone.layer.1.relu2'],
        ['backbone.layer.2.depth_conv', 'backbone.layer.2.bn1', 'backbone.layer.2.relu1'],
        ['backbone.layer.2.point_conv', 'backbone.layer.2.bn2', 'backbone.layer.2.relu2'],
        ['backbone.layer.3.depth_conv', 'backbone.layer.3.bn1', 'backbone.layer.3.relu1'],
        ['backbone.layer.3.point_conv', 'backbone.layer.3.bn2', 'backbone.layer.3.relu2']
    ]
    quant_fuse_model = torch.quantization.fuse_modules(quant_model, op_fused_lst)

    quant_fuse_model.train()
    
    # quantization configuration
    quant_conf = torch.quantization.QConfig(
        activation = torch.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.with_args(
            observer = torch.quantization.observer.MovingAverageMinMaxObserver,
            quant_min = 0,
            quant_max = 255,
            reduce_range = True
        ),
        weight = torch.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.with_args(
            observer = torch.quantization.observer.MovingAveragePerChannelMinMaxObserver,
            quant_min = 0,
            quant_max = 255,
            dtype = torch.quint8,
            qscheme = torch.per_channel_affine
        )
    )
    
    quant_fuse_model.qconfig = quant_conf
    quant_fuse_model = torch.quantization.prepare_qat(quant_fuse_model, inplace = True)
    quant_fuse_model.to(device)
    
    trainset, testset = get_cifar10_dataset()
    
    # hyperparameters
    batch_size = 16
    init_lr = 0.001
    total_epoch = 1
    
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 4
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        quant_fuse_model.parameters(),
        lr = init_lr
    )
    
    best_acc = 0.
    save_dir = './saved_model'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok = True)
    for epoch_idx in range(total_epoch):
        # training
        quant_fuse_model.train()
        running_loss = 0.
        running_acc = 0.
        epoch_running_loss = 0.
        for step_idx, data in enumerate(trainloader):
            x, y = data
            optimizer.zero_grad()
            logits = quant_fuse_model(x.to(device))
            loss = criterion(logits, y.to(device))
            #pred_y = torch.max(logits, dim = 1)[1]
            #running_loss += loss.item()
            #running_acc += (pred_y.to(device) == y.to(device)).sum().item()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            for val_data in testloader:
                val_x, val_y = val_data
                outputs = quant_fuse_model(val_x.to(device))
                pred_y = torch.max(outputs, dim = 1)[1]
                running_acc += (pred_y.to(device) == val_y.to(device)).sum().item()
                running_loss += criterion(outputs, val_y.to(device)).item()
                
        print('[%s] epoch:%03d loss:%.4f acc:%.4f'%(datetime.datetime.now(), epoch_idx+1, running_loss / len(testloader), running_acc / (batch_size*len(testloader))))
        running_loss = 0.
        running_acc = 0.
        torch.save(quant_fuse_model.state_dict(), '%s/epoch_%03d_uint8.pth'%(save_dir, epoch_idx+1))
        
        
def eval():
    in_model_pth = './saved_model/epoch_001_uint8.pth'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    quant_model = RexKWSModelQuant()
    quant_model.eval()
    
    # op fuse
    op_fused_lst =  [
        ['backbone.conv1', 'backbone.relu'],
        ['backbone.layer.0.depth_conv', 'backbone.layer.0.bn1', 'backbone.layer.0.relu1'],
        ['backbone.layer.0.point_conv', 'backbone.layer.0.bn2', 'backbone.layer.0.relu2'],
        ['backbone.layer.1.depth_conv', 'backbone.layer.1.bn1', 'backbone.layer.1.relu1'],
        ['backbone.layer.1.point_conv', 'backbone.layer.1.bn2', 'backbone.layer.1.relu2'],
        ['backbone.layer.2.depth_conv', 'backbone.layer.2.bn1', 'backbone.layer.2.relu1'],
        ['backbone.layer.2.point_conv', 'backbone.layer.2.bn2', 'backbone.layer.2.relu2'],
        ['backbone.layer.3.depth_conv', 'backbone.layer.3.bn1', 'backbone.layer.3.relu1'],
        ['backbone.layer.3.point_conv', 'backbone.layer.3.bn2', 'backbone.layer.3.relu2']
    ]
    quant_fuse_model = torch.quantization.fuse_modules(quant_model, op_fused_lst)
    quant_fuse_model.train()
    
    # quantization configuration
    quant_conf = torch.quantization.QConfig(
        activation = torch.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.with_args(
            observer = torch.quantization.observer.MovingAverageMinMaxObserver,
            quant_min = 0,
            quant_max = 255,
            reduce_range = True
        ),
        weight = torch.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize.with_args(
            observer = torch.quantization.observer.MovingAveragePerChannelMinMaxObserver,
            quant_min = 0,
            quant_max = 255,
            dtype = torch.quint8,
            qscheme = torch.per_channel_symmetric,
            reduce_range = False,
            ch_axis = 0
        )
    )
    
    quant_fuse_model.qconfig = quant_conf
    quant_fuse_model = torch.quantization.prepare_qat(quant_fuse_model, inplace = True)

    quant_fuse_model.load_state_dict(torch.load(in_model_pth))
    
    quant_fuse_model.eval()
    
    # for param in quant_fuse_model.parameters():
    #     print(param)
    
    saved_int_model = './saved_int_model'
    if not os.path.exists(saved_int_model):
        os.makedirs(saved_int_model, exist_ok = True)
    int_model = torch.quantization.convert(quant_fuse_model, inplace = True)
    torch.save(int_model.state_dict(), '%s/%s.pth'%(saved_int_model, 'epoch_001_uint8_quant'))

if __name__ == '__main__':
    train()
    #eval()