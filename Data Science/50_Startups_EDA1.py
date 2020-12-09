# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:11:18 2020

@author: aswin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
datasheet=pd.read_csv("50_Startups_EDA.csv") 
datasheet.fillna(datasheet.mean(),inplace=True)
x=datasheet.iloc[:,0:-1].values
y=datasheet.iloc[:,5].values
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
A=make_column_transformer((OneHotEncoder(categories='auto'), [4]),remainder="passthrough")
x=A.fit_transform(x)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=0)
from sklearn.linear_model import LinearRegression
prediction=LinearRegression()
prediction.fit(xtrain,ytrain)
result=prediction.predict(xtest)
prediction.score(xtrain,ytrain)
prediction.score(xtest,ytest)
import statsmodels.api as sm
x=np.append(arr=np.ones(shape=(60,1),dtype=int),values=x,axis=1)
xnew1= np.array(x[:,[0,2,3,4,5,6,7]], dtype=int)
model = sm.OLS(y,xnew1)
results1=model.fit()
results1.summary()
xnew11=xnew1[:,1:]
xtrain,xtest,ytrain,ytest=train_test_split(xnew11,y,test_size=0.30,random_state=0)
prediction=LinearRegression()
prediction.fit(xtrain,ytrain)
result=prediction.predict(xtest)
prediction.score(xtrain,ytrain)
prediction.score(xtest,ytest)

model2 = sm.OLS(y,xnew2)
results2=model2.fit()
results2.summary()
xnew22=xnew2[:,1:]

model3 = sm.OLS(y,xnew3)
results3=model3.fit()
results3.summary()
xnew33=xnew3[:,1:]

model4 = sm.OLS(y,xnew4)
results4=model4.fit()
results4.summary()
xnew44=xnew4[:,1:]

model5 = sm.OLS(y,xnew5)
results5=model5.fit()
results5.summary()
xnew55=xnew5[:,1:]



