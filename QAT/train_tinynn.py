# -*- coding:utf-8 -*-
# Author : Yao Huang
# Date : 2023.10.12

import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

import os
import datetime
import torch
import torch.nn as nn

from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.tracer import model_tracer

from datasets import get_cifar10_dataset
from model import RexKWSModel

def train():
    
    with model_tracer():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = RexKWSModel()
        dummy_input = torch.randn((1, 3, 32, 32))
        
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
        
        quantizer = QATQuantizer(model, dummy_input, work_dir = 'out', config={'asymmetric': False, 'per_tensor': True})
        # windows
        quantizer.backend = 'fbgemm'
        qat_model = quantizer.quantize()
        
    qat_model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        qat_model.parameters(),
        lr = init_lr
    )
    
    save_dir = './saved_model_1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok = True)
    for epoch_idx in range(total_epoch):
        # training
        qat_model.train()
        running_loss = 0.
        running_acc = 0.
        for step_idx, data in enumerate(trainloader):
            x, y = data
            optimizer.zero_grad()
            logits = qat_model(x.to(device))
            loss = criterion(logits, y.to(device))
            #pred_y = torch.max(logits, dim = 1)[1]
            #running_loss += loss.item()
            #running_acc += (pred_y.to(device) == y.to(device)).sum().item()
            loss.backward()
            optimizer.step()
            
        # with torch.no_grad():
        #     for val_data in testloader:
        #         val_x, val_y = val_data
        #         outputs = qat_model(val_x.to(device))
        #         pred_y = torch.max(outputs, dim = 1)[1]
        #         running_acc += (pred_y.to(device) == val_y.to(device)).sum().item()
        #         running_loss += criterion(outputs, val_y.to(device)).item()
                
        # print('[%s] epoch:%03d loss:%.4f acc:%.4f'%(datetime.datetime.now(), epoch_idx+1, running_loss / len(testloader), running_acc / (batch_size*len(testloader))))
        # running_loss = 0.
        # running_acc = 0.
        # torch.save(qat_model.state_dict(), '%s/epoch_%03d_uint8.pth'%(save_dir, epoch_idx+1))
        
        with torch.no_grad():
            qat_model.eval()
            qat_model.cpu()
            qat_model = quantizer.convert(qat_model)
            torch.backends.quantized.engine = quantizer.backend
            converter = TFLiteConverter(qat_model, dummy_input, tflite_path='out/qat_model.tflite')
            converter.convert()
    
if __name__ == '__main__':
    train()