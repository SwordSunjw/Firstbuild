import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_excel("E:\Datapython\Pytorch_Learn\BJweather_1678065761546.xlsx")
print(data.head(10))
#更改文件夹名字
#缺失值
print(data.isnull().sum())
#data['白天天气'].fillna(data['白天天气'].mean(),inplace=True)
data.dropna(inplace=True)#delete null
#重复值
print(data.duplicated().sum())
sunny = data[data['白天天气']=='晴']['白天气温'].mean()
print("sunny day mean is:",sunny)
##predict&regression
data_encoded = pd.get_dummies(data,columns = ['白天天气'],drop_first=True)
print(data_encoded)
X=data_encoded[['白天气温','昨天气温','前天气温']]
y=data_encoded[['白天天气_中雪','白天天气_多云','白天天气_小雨','白天天气_小雪','白天天气_扬沙','白天天气_晴','白天天气_阴','白天天气_阵雨','白天天气_雨夹雪','白天天气_雷阵雨','白天天气_雾']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=21)
model = LinearRegression()
model.fit(X_train,y_train)

y_predit=model.predict(X_test)
print("we predict results is:",y_predit)
y_predit = pd.DataFrame(y_predit, columns=['白天天气_中雪','白天天气_多云','白天天气_小雨','白天天气_小雪','白天天气_扬沙','白天天气_晴','白天天气_阴','白天天气_阵雨','白天天气_雨夹雪','白天天气_雷阵雨','白天天气_雾'])
y_predit.to_excel('预测白天气温.xlsx', index=True)














































































































