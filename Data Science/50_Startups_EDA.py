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
x1train,x1test,y1train,y1test=train_test_split(xnew11,y,test_size=0.30,random_state=0)
prediction1=LinearRegression()
prediction1.fit(x1train,y1train)
prediction1.score(x1train,y1train)
prediction1.score(x1test,y1test)
xnew2= np.array(x[:,[0,3,4,5,6,7]], dtype=int)
model2 = sm.OLS(y,xnew2)
results2=model2.fit()
results2.summary()
xnew22=xnew2[:,1:]
x2train,x2test,y2train,y2test=train_test_split(xnew22,y,test_size=0.30,random_state=0)
prediction2=LinearRegression()
prediction2.fit(x2train,y2train)
prediction2.score(x2train,y2train)
prediction2.score(x2test,y2test)
xnew3= np.array(x[:,[0,3,4,5,7]], dtype=int)
model3 = sm.OLS(y,xnew3)
results3=model3.fit()
results3.summary()
xnew33=xnew3[:,1:]
x3train,x3test,y3train,y3test=train_test_split(xnew33,y,test_size=0.30,random_state=0)
prediction3=LinearRegression()
prediction3.fit(x3train,y3train)
prediction3.score(x3train,y3train)
prediction3.score(x3test,y3test)
xnew4= np.array(x[:,[0,4,5,7]], dtype=int)
model4 = sm.OLS(y,xnew4)
results4=model4.fit()
results4.summary()
xnew44=xnew4[:,1:]
x4train,x4test,y4train,y4test=train_test_split(xnew44,y,test_size=0.30,random_state=0)
prediction4=LinearRegression()
prediction4.fit(x4train,y4train)
prediction4.score(x4train,y4train)
prediction4.score(x4test,y4test)
xnew5= np.array(x[:,[0,4,5]], dtype=int)
model5 = sm.OLS(y,xnew5)
results5=model5.fit()
results5.summary()
xnew55=xnew5[:,1:]
x5train,x5test,y5train,y5test=train_test_split(xnew55,y,test_size=0.30,random_state=0)
prediction5=LinearRegression()
prediction5.fit(x5train,y5train)
prediction5.score(x5train,y5train)
prediction5.score(x5test,y5test)


