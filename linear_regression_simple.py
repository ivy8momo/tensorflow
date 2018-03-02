# -*- coding: utf-8 -*-

import pandas as pd
import sklearn.cross_validation as cv
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error  # 误差平方和
import matplotlib.pyplot as plt  # 画图
import matplotlib.font_manager as fm  # 中文显示

# 读取数据
iris = pd.read_csv('e:\data\iris.csv')

# 查看数据
print(iris.head())
print(iris.describe())

# 建模数据
X = iris.loc[:, ['SepalWidth']]
Y = iris.loc[:, ['SepalLength']]

# 验证：训练集和测试集
train_X, test_X, train_Y, test_Y = cv.train_test_split(X, Y, test_size=0.2, random_state=1234)
# LinearRegression()
linreg = LinearRegression()
model = linreg.fit(train_X, train_Y)
a, b = model.coef_, model.intercept_
print(a, b)
pred_y = model.predict(test_X)
MES = np.sqrt(mean_squared_error(test_Y, pred_y))
score = model.score(test_X, test_Y)
print('测试集的预测效果：MES=%.4f\n' % MES)
print('测试集的预测效果：R方=%.4f\n' % score)
# ols
train_X = sm.add_constant(train_X)
model_ols = sm.OLS(train_Y, train_X).fit()
print(model_ols.summary())
test_X = sm.add_constant(test_X)
pred_ols = model_ols.predict(test_X)
# Rsqrt  'OLSResults' object has no attribute 'score'没有R方的值
MES_ols = np.sqrt(mean_squared_error(test_Y, pred_ols))
print('测试集的预测效果：MES=%.4f\n' % MES_ols)
# print('测试集的预测效果：R方=%.4f\n' % Rsqrt)
import urllib2 url='http://www.baidu.com/'

# 交叉验证-LinearRegression()
predicted = cv.cross_val_predict(linreg, X, Y, cv=5)
print("10折交叉验证MSE:", mean_squared_error(Y, predicted))
print("10折交叉验证RMSE:", np.sqrt(mean_squared_error(Y, predicted)))
scores = cv.cross_val_score(linreg, X, Y, cv=5)
print("交叉验证R方值：", scores)
print("交叉验证R方均值：", np.mean(scores))

# 设置中文正常显示
font1 = fm.FontProperties(fname='C:\Windows\Fonts\msyh.ttc')
fig, ax = plt.subplots()
ax.scatter(Y, predicted, label = '观测点')
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4, label = '拟合标准')
ax.set_xlabel(u'真实值',fontproperties=font1)
ax.set_ylabel(u'预测值',fontproperties=font1)
# 添加图例
plt.legend(loc = 'upper left',prop = font1)
plt.show()