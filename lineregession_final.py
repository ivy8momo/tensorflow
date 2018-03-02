# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split # 训练集与测试集拆分
from sklearn import linear_model # 线性回归模型1
import statsmodels.api as sm     # 多元线性回归2-对应sm.ols
from sklearn.metrics import mean_squared_error # 误差平方和

# 读取数据
iris = pd.read_csv('e:\data\iris.csv')

# 拆分训练集和测试集
train,test = train_test_split(iris,train_size=0.8, random_state=1234)
train_X = train.iloc[:,2:5]
train_Y = train.iloc[:,[1]]
test_X = test.iloc[:,2:5]
test_Y = test.iloc[:,[1]]

# 训练模型1
train_X = sm.add_constant(train_X)
linreg_1 = sm.OLS(train_Y,train_X).fit()
print(linreg_1.summary())
# 预测1
test_X = sm.add_constant(test_X)
pred = linreg_1.predict(test_X)
RMSE = np.sqrt(mean_squared_error(test.SepalLength,pred))
print('模型的预测效果：RMES=%.4f\n' %RMSE)

# 训练-LinearRegression()
linreg_1 = linear_model.LinearRegression()
fit4.fit(x,y)
c,d = fit4.coef_,fit4.intercept_
print(c,d)
# 测试-LinearRegression()
x3_test = Test.iloc[:,2:5]
pred4 = fit4.predict(x3_test)
RMSE4 = np.sqrt(mean_squared_error(Test.SepalLength,pred4))
print('第四个模型的预测效果：RMES=%.4f\n' %RMSE4)
