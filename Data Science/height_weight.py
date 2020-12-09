# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:38:01 2020

@author: aswin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
datasheet=pd.read_csv("height_weight.csv")
datasheet.isnull().sum()
datasheet.mean()
datasheet.info()
datasheet.mean()
datasheet.median()
datasheet.fillna(datasheet.mean(),inplace=True)
#doubt......datasheet[height] not working
x=datasheet.iloc[:,0:1].values
y=datasheet.iloc[:,1].values
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.40,random_state=0)
from sklearn.linear_model import LinearRegression
prediction=LinearRegression()
prediction.fit(xtrain,ytrain)
result=prediction.predict(xtest)
