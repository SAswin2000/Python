 # -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 09:59:03 2020

@author: aswin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
datasheet=pd.read_csv("Data_preprocessing.csv")
datasheet.isnull().sum()
x=datasheet.iloc[:,0:3].values
y=datasheet.iloc[:,3:4].values
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label=LabelEncoder()
x[:,0]=label.fit_transform(x[:,0])
encode=OneHotEncoder(categories='0')
x=encode.fit_transform(x).toarray()
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.20,random_state=0)
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
xtrain=scale.fit_transform(xtrain)
xtest=scale.transform(xtest)

