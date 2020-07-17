#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:32:12 2020

@author: jonona
"""


from sklearn import tree
import pandas as pd
import numpy as np
#import statsmodels.api as sm
from func import *
from sklearn.ensemble import RandomForestRegressor


np.random.seed(20)

#all_columns=["Weekday","Temp,Mx","TempMvg","TempMin","Hum,Mx","HumAvg","HumMin","WindMax","WindAvg","WindMin","PresMax","PresAvg","PresMin","Target"]
col=["Month", "Weekday", "DayBefore", "TempAvg", "HumMax", "Target"]

item='Coffee'
unit='day'

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


# regression tree
#model = tree.DecisionTreeRegressor(min_samples_leaf=10)
model=RandomForestRegressor(n_estimators=100, min_samples_leaf=10)
model = model.fit(X, y)
predictions = model.predict(X) # make the predictions by the model

# _, ax = plt.subplots(figsize=(15, 15))
# tree.plot_tree(model, ax=ax)
# plt.show()


plotting(pred=predictions, target=y, epoch=0, unit=unit, itemName=item)

base=train_target.mean()
error=np.abs(y-predictions).sum()/(y[1:].sum())
rsquared=model.score(X,y)
rmse=np.sqrt(((predictions-train_target)**2).sum()/len(trainset))

print("Baseline prediction: {:.04f} \nTrain Error for Absolute Values: {:.04f} \nTrain R squared: {:.04f} \nTrain RMSE: {:.04f}".format(base, error, rsquared, rmse))


# testing
#test_target = np.log(testset['Target'].values)
test_target = testset['Target'].values
testset = testset.drop('Target', axis = 1).values

X = testset
y = test_target

predictions = model.predict(X)

#plotting(pred=np.exp(predictions), target=np.exp(y), epoch=0, unit='day', itemName=item)
plotting(predictions, y, 0, 'day', 'Coffee', 'N')


error=np.abs(y-predictions).sum()/(y[1:].sum())
rsquared=model.score(X,y)
rmse=np.sqrt(((predictions-test_target)**2).sum()/len(testset))


print("Test Error for Absolute Values: {:.04f} \nTest R squared: {:.04f} \nTest RMSE: {:.04f}".format(error, rsquared, rmse))

#model.save('regression_tree.pickle')




