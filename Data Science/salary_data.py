import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
datasheet=pd.read_csv("salary_data.csv")
x=datasheet.iloc[:,0:1].values
y=datasheet.iloc[:,1].values
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=0)
from sklearn.linear_model import LinearRegression
prediction=LinearRegression()
prediction.fit(xtrain,ytrain)
result=prediction.predict(xtest)
