#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:46:07 2020

@author: jonona
"""

from data_for_torch import loaders, Net
from func import plotting
import torch
#from torch import data_utils
#import torch.nn as nn
import torch.nn.functional as F
import copy
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(20)

#all possible columns
#col=["Month","Weekday","TwoDaysBefore","DayBefore","TempMax","TempAvg","TempMin","HumMax","HumAvg","HumMin","WindMax","WindAvg","WindMin","PresMax","PresAvg","PresMin","Target"]
#choose columns
col=["Month", "Weekday", "Target"]

if "Hour" in col: unit='hour'
else: unit='day'

batch_size=25

train_loader, test_loader, input_features = loaders(col,unit,batch_size,factors=True)

model=Net(input_features)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
loss_func = torch.nn.MSELoss()

def train(epoch):
    model.train()
    total_loss=0
    
    for i,(data,target) in enumerate(train_loader):
        optimizer.zero_grad()
        loss = loss_func(model(data).squeeze(), target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if i == (len(train_loader)-1):
            print('[{}/{}] Loss: {:.4f}'.format(i + 1, len(train_loader), total_loss / len(train_loader)))
    
    return total_loss / len(train_loader)
        


def test(loader,epoch):
    model.eval()
    pred=torch.zeros([1,1])
    targets=torch.zeros([1,1])
    
    for data, target in loader:
        with torch.no_grad():
            pred = torch.cat((pred,model(data)))
            targets = torch.cat((targets, target.unsqueeze(1)))
    plotting(np.exp(pred[1:]), np.exp(targets[1:]), epoch, unit)
    #plotting(pred[1:], targets[1:], epoch, 'day (log(N))', 'Coffee', 'log(Sales)')
    plt.show()
    targets.squeeze()
    error=np.abs(targets[1:]-pred[1:]).sum()/(targets[1:].sum())
    error_abs=np.abs(np.exp(targets[1:])-np.exp(pred[1:])).sum()/(np.exp(targets[1:]).sum())
    print("Test Relative Error for Logarithms: {:.04f} \nTest Relative Error for Absolute Values: {:.04f}".format(error, error_abs))
    


max_epoch=30
train_loss=np.zeros(max_epoch)

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1000000


for epoch in range(1, max_epoch+1):
    print('Epoch: {:03d}'.format(epoch))
    loss=train(epoch)
    train_loss[epoch-1]=loss
    
    test(test_loader,epoch)
    if loss<best_loss:
        best_loss = loss
        best_model_wts = copy.deepcopy(model.state_dict())
    exp_lr_scheduler.step()
    
    
#Save best model
model.load_state_dict(best_model_wts)
#torch.save(model, osp.join(osp.dirname(osp.abspath(__file__)),'model.pth'))


#Plot learning curves
plt.figure(figsize=[10,10])
plt.plot(np.arange(1,max_epoch), train_loss[1:],  color='blue', label='train loss')
plt.legend(loc='best')
plt.xlabel('Epoch')
#plt.ylim((0,1))
plt.show()