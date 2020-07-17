#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:42:35 2020

@author: jonona
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from func import *

np.random.seed(20)

#all_columns=["Weekday","TwoDaysBefore","DayBefore","TempMax","TempAvg","TempMin","HumMax","HumAvg","HumMin","WindMax","WindAvg","WindMin","PresMax","PresAvg","PresMin","Target"]
col=["Month", "Weekday", "DayBefore", "TempAvg", "HumMax", "Target"]

item='Coffee'

#new data
trainset,testset = clean_data_daily(col,True,item)
#trainset,testset = clean_data_hourly(["Month","Weekday","Hour","HumMax","WindMin","Target"],True,item)


# #save to reproduce
# trainset.to_pickle('trainset.pkl')
# testset.to_pickle('testset.pkl')

#load data
# trainset=pd.read_pickle('trainset.pkl')
# testset=pd.read_pickle('testset.pkl')


#train_target = np.log(trainset['Target'].values)
train_target = trainset['Target'].values
trainset = trainset.drop('Target', axis = 1)#.values


X = trainset
y = train_target


# statsmodel
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model
print(model.summary())

#plotting(pred=np.exp(predictions), target=np.exp(y), epoch=0, unit='hour', itemName=item)
plotting(pred=predictions, target=y, epoch=0, unit='day', itemName=item)



# testing
#test_target = np.log(testset['Target'].values)
test_target = testset['Target'].values
testset = testset.drop('Target', axis = 1).values

X = testset
y = test_target

predictions = model.predict(X)

#plotting(pred=np.exp(predictions), target=np.exp(y), epoch=0, unit='day (log(N))', itemName=item, 'log(Sales)')
plotting(predictions, y, 0, 'day', 'Coffee')


#error_abs=np.abs(np.exp(y)-np.exp(predictions)).sum()/np.exp(y).sum()
error=np.abs(y-predictions).sum()/(y[1:].sum())
rsquared=1-((predictions-test_target)**2).sum()/((y-train_target.mean())**2).sum()
#rmse=np.sqrt(((np.exp(predictions)-np.exp(test_target))**2).sum()/len(testset))
rmse=np.sqrt(((predictions-test_target)**2).sum()/len(testset))

#print("Test Error for Logarithms: {:.04f} \nTest Error for Absolute Values: {:.04f} \nR squared: {:.04f} \nRMSE: {:.04f}".format(error, error_abs, rsquared, rmse))
print("Test Error for Absolute Values: {:.04f} \nTest R squared: {:.04f} \nTest RMSE: {:.04f}".format(error, rsquared, rmse))

#model.save('linear_regression.pickle')