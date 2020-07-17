# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(20)

def as_factor(data):
    try:
        data.Month=data.Month.astype("object")
        dummyCols=pd.get_dummies(data["Month"],prefix="Month")
        data=data.join(dummyCols)
        del data["Month"]
    except:
        print("no month")
    try:
        data.Weekday=data.Weekday.astype(object)
        dummyCols=pd.get_dummies(data["Weekday"],prefix="Weekday")
        data=data.join(dummyCols)
        del data["Weekday"]
    except:
        print("no weekday")
    try:
        data.Hour=data.Hour.astype(object)
        dummyCols=pd.get_dummies(data["Hour"],prefix="Hour")
        data=data.join(dummyCols)
        del data["Hour"]
    except:
        print("no hour")
    return data


def del_outliers(data):
    y = data['Target']
    removed_outliers = y.between(y.quantile(.15), y.quantile(.85))
    index_names = data[~removed_outliers].index
    data.drop(index_names, inplace=True)
    return data

def expandto24(data):
    # add non-bussiness hours for every day with zero sales
    dates=data["Day"].unique()
    for date in dates:
        for i in [random.randrange(0,7), random.randrange(0,7), random.randrange(21,24)]:
            row=pd.DataFrame({"Day":date,"ItemCount":int(0),"Hour":int(i)}, index=[0])
            data=pd.concat([data,row], join='inner')
    return data

def normalize(data):
    #data.Hour=data.Hour/24
    t=data["TempMax"].max()
    data['TempAvg']=data['TempAvg']/t
    data['TempMin']=data['TempMin']/t
    data['TempMax']=data['TempMax']/t
    data['HumAvg']=data['HumAvg']/100
    data['HumMin']=data['HumMin']/100
    data['HumMax']=data['HumMax']/100
    w=data["WindMax"].max()
    data['WindAvg']=data['WindAvg']/w
    data['WindMin']=data['WindMin']/w
    data['WindMax']=data['WindMax']/w
    p=data["PresMax"].max()
    data['PresAvg']=data['PresAvg']/p
    data['PresMin']=data['PresMin']/p
    data['PresMax']=data['PresMax']/p
    data['Month']=data['Month']/12
    data['Weekday']=data['Weekday']/7
    return data

def clean_data_hourly(col,factors=False,Name="Coffee"):
    
    path=osp.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    
    # get sales data
    inputData = pd.read_csv(path+"/BreadBasket_DMS.csv")
    inputData['Date']=inputData['Date'].astype('datetime64[ns]')
    
    # clean weather data
    weather=pd.read_csv(path+"/weather.csv", sep='\t')
    for column in weather.columns[3:]:
        weather[column] = weather[column].apply(lambda x: float(x.replace(',', '.')))
    weather.insert(0,"Date",pd.to_datetime(weather[weather.columns[:3]]))
    weather.drop(columns=weather.columns[1:4], inplace=True)
    
    # get dates
    newDateTime = inputData.Date.astype('str') +' '+inputData.Time
    inputData.index = (pd.to_datetime(newDateTime))
    inputData.drop(["Time","Date","Transaction"],axis=1,inplace=True)
    data=inputData[inputData.Item == Name]
    # group by hours
    group=pd.DataFrame({"ItemCount":data.groupby([data.index.map(lambda t: (t.date(),t.hour)),"Item"]).size()}).reset_index();
    group[['Date','Hour']] = pd.DataFrame(group.level_0.values.tolist(), index= group.index)
    group.Date=group.Date.astype('datetime64')
    # get months and weekdays
    group["Weekday"] = group.Date.dt.weekday
    group["Month"] = group.Date.dt.month
    # merging weather
    data=pd.merge(left=group, right=weather, left_on='Date', right_on='Date')
    # shuffle, normalize and delete outliers
    data = data.sample(frac=1)
    data=normalize(data)
    data.rename(columns={"ItemCount" : "Target"}, inplace=True)
    data=del_outliers(data)
    #filter out columns
    data=pd.DataFrame(data[col])
    
    if factors==True:
        data=as_factor(data)
    
    #trainset
    trainset = data[:int(np.ceil(len(data)*0.85))]
    trainset = trainset.sample(frac=1).reset_index(drop=True)
    
    #testset
    testset = data[int(np.ceil(len(data)*0.85)):]
    testset = testset.sample(frac=1).reset_index(drop=True)
    
    return trainset,testset


def clean_data_daily(col,factors=False,Name="Coffee"):
    
    path=osp.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    
    # get sales data
    inputData = pd.read_csv(path+"/BreadBasket_DMS.csv")
    inputData['Date']=inputData['Date'].astype('datetime64[ns]')
    
    # clean weather data
    weather=pd.read_csv(path+"/weather.csv", sep='\t')
    for column in weather.columns[3:]:
        weather[column] = weather[column].apply(lambda x: float(x.replace(',', '.')))
    weather.insert(0,"Date",pd.to_datetime(weather[weather.columns[:3]]))
    weather.drop(columns=weather.columns[1:4], inplace=True)
    
    # get dates and time
    newDateTime = inputData.Date.astype('str') +' '+inputData.Time
    inputData.index = (pd.to_datetime(newDateTime))
    inputData.drop(["Time","Date","Transaction"],axis=1,inplace=True)
    data=inputData[inputData.Item == Name]
    # group by days
    group=pd.DataFrame({"ItemCount":data.groupby([data.index.map(lambda t: t.date()),"Item"]).size()}).reset_index();
    #group[['Day','Hour']] = pd.DataFrame(group.level_0.values.tolist(), index= group.index)
    group.rename(columns={"level_0" : "Date"}, inplace=True)
    group.Date=group.Date.astype('datetime64')
    # get months and weekdays
    group["Weekday"] = group.Date.dt.weekday
    group["Month"] = group.Date.dt.month
    # add data from past
    group["DayBefore"]=group.shift(periods=1, fill_value=group["ItemCount"].mean())["ItemCount"]
    group["TwoDaysBefore"]=group.shift(periods=2, fill_value=group["ItemCount"].mean())["ItemCount"]
    # add weather data
    data=pd.merge(left=group, right=weather, left_on='Date', right_on='Date')
    # shuffle, normalize and delete outliers
    data = data.sample(frac=1)
    data=normalize(data)
    data.rename(columns={"ItemCount" : "Target"}, inplace=True)
    data=del_outliers(data)
    #filter out columns
    data=pd.DataFrame(data[col])
    
    #convert month, weekday and hour to factors
    if factors==True:
        data=as_factor(data)
    
    
    #trainset
    trainset = data[:int(np.ceil(len(data)*0.85))]
    trainset = trainset.sample(frac=1).reset_index(drop=True)
    
    #testset
    testset = data[int(np.ceil(len(data)*0.85)):]
    testset = testset.sample(frac=1).reset_index(drop=True)
    
    return trainset,testset



def plotting(pred,target,epoch,unit,itemName='Coffee',ylab='Sales'):
    fig = plt.figure(figsize = (15,5))
    ax = fig.gca()
    n=len(pred)
    x = range(n)
    yTrue = target
    yPred = pred
    plt.plot(x,yTrue,label="True", marker='o', color='green')
    plt.plot(x,yPred,label="Predicted", marker='s', color='red')
    plt.legend()
    plt.xlabel('Time span',fontsize=10)
    plt.ylabel(ylab,fontsize=10)
    ax.tick_params(labelsize=10)
    plt.ylim(ymin=-1, ymax=target.max()+1)
    plt.title('Number of {} sold in one {}, Epoch: {:03d}'.format(itemName,unit,epoch),fontsize=20)
    plt.grid()
    plt.ioff()
