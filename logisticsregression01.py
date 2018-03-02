# -*- coding: utf-8 -*-

# 导入第三方包
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from ggplot import *

# 读取数据集
purchase = pd.read_csv('e:\data\iris02.csv')
# 查看数据类型
purchase.dtypes
# 查看各变量的缺失情况
purchase.isnull().sum()

print(pd.pivot_table(purchase,
                     index = ['Species'],
                     values = ['ID'],
                     aggfunc=len))

purchase['if_true'] = 0

# 对Gender变量作哑变量处理
# dummy = pd.get_dummies(purchase.Gender)
# 为防止多重共线性，将哑变量中的Female删除
# dummy_drop = dummy.drop('Female', axis = 1)

# 剔除用户ID和Gender变量
# purchase = purchase.drop(['User ID','Gender'], axis = 1)
# 如果调用Logit类，需要给原数据集添加截距项
purchase['Intercept'] = 1

# 哑变量和原数据集合并
# model_data = pd.concat([dummy_drop,purchase], axis = 1)
# model_data
model_data = purchase

# 标记目标变量为逻辑值
true_data = purchase[purchase['Species'] == 'setosa']
true_data['if_true'] = 1
false_data = purchase[purchase['Species'] != 'setosa']
false_data['if_true'] = 0
# 将数据集拆分为训练集和测试集
X = model_data.drop('Purchased', axis = 1)
y = model_data['Purchased']
# 训练集与测试集的比例为75%和25%
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state=0)

# 根据训练集构建Logistic分类器
logistic = smf.Logit(y_train,X_train).fit()
logistic.summary()