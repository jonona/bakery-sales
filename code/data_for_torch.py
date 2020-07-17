#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:11:19 2020

@author: jonona
"""
from func import clean_data_hourly, clean_data_daily
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.random.seed(20)

def loaders(col,unit='day',batch_size=50,factors=False):
    batch_size=batch_size
    if unit=='day':
        trainset, testset = clean_data_daily(col=col, factors=factors)
    else:
        trainset, testset = clean_data_hourly(col=col, factors=factors)
    
    #train
    train_target = torch.tensor(np.log(trainset['Target'].values)).float()
    trainset = torch.tensor(trainset.drop('Target', axis = 1).values).float()
    train_tensor = TensorDataset(trainset, train_target) 
    train_loader = DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)
    
    #test
    test_target = torch.tensor(np.log(testset['Target'].values)).float() 
    testset = torch.tensor(testset.drop('Target', axis = 1).values).float()
    test_tensor = TensorDataset(testset, test_target) 
    test_loader = DataLoader(dataset = test_tensor, batch_size = batch_size, shuffle = False)
    
    input_features=testset.size()[1]
    
    return train_loader, test_loader, input_features

class Net(nn.Module):
    def __init__(self, input_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_features,10)
        self.fc2 = nn.Linear(10,1)
    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x