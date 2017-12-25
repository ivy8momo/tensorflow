# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf # 线性回归模型
from sklearn import linear_model # 线性回归模型
import statsmodels.api as sm # 多元线性回归-对应smf.ols
from sklearn.cross_validation import train_test_split # 训练集与测试集拆分
from sklearn.metrics import mean_squared_error # 误差平方和
import matplotlib.pyplot as plt # 画图
import matplotlib.font_manager as fm # 中文显示
from sklearn.cross_validation import cross_val_predict # 交叉验证
from sklearn.cross_validation import cross_val_score # 交叉验证
# 读取数据
iris = pd.read_csv('e:\data\iris.csv')

# 保存描述统计
b = iris.describe()
print(b)

# 拆分训练集和测试集
Train,Test = train_test_split(iris,train_size=0.8, random_state=1234)

# train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=1)

# 一元线性回归

# 训练-ols
fit = smf.ols('SepalLength~SepalWidth',data=Train).fit()
print(fit.summary())
# 训练-LinearRegression()
fit3 = linear_model.LinearRegression()
fit3.fit(Train['SepalWidth'].reshape(-1,1),Train['SepalLength'])
a,b = fit3.coef_,fit3.intercept_
print(a,b)
# 预测-ols
pred = fit.predict(exog = Test)
RMSE = np.sqrt(mean_squared_error(Test.SepalLength,pred))
print('第一个模型的预测效果：RMES=%.4f\n' %RMSE)
# 预测-LinearRegression()
pred3 = fit3.predict(Test['SepalWidth'].reshape(-1,1))
RMSE3 = np.sqrt(mean_squared_error(Test.SepalLength,pred3))
print('第三个模型的预测效果：RMES=%.4f\n' %RMSE3)

# 散点图与回归线
plt.scatter(iris['SepalWidth'],iris['SepalLength'],color = 'blue')
plt.plot(iris['SepalWidth'],fit.predict(exog = iris),color = 'red',linewidth = 4)
plt.plot(iris['SepalWidth'],fit3.predict(iris['SepalWidth'].reshape(-1,1)),color = 'black',linewidth = 4)
plt.show()

# 多元线性回归

# 训练-smf.ols
fit2 = smf.ols('SepalLength~SepalWidth+PetalWidth+PetalLength',data=Train).fit()
print(fit2.summary())
# 训练-sm.ols
x = Train[["SepalWidth","PetalWidth","PetalLength"]]
y = Train["SepalLength"]
x = sm.add_constant(x)
fit5 = sm.OLS(y,x).fit()
print(fit5.summary())
# 训练-sm.ols
x2 = Train.iloc[:,2:5]
y2 = Train.iloc[:,[1]]
x2 = sm.add_constant(x2)
fit6 = sm.OLS(y2,x2).fit()
print(fit6.summary())
# 训练-LinearRegression()
x = Train.iloc[:,2:5]
y = Train.iloc[:,[1]]
fit4 = linear_model.LinearRegression()
fit4.fit(x,y)
c,d = fit4.coef_,fit4.intercept_
print(c,d)
# 测试-smf.ols
pred2 = fit2.predict(exog = Test)
RMSE2 = np.sqrt(mean_squared_error(Test.SepalLength,pred2))
print('第二个模型的预测效果：RMES=%.4f\n' %RMSE2)
# 测试-sm.ols
x_test = Test[["SepalWidth","PetalWidth","PetalLength"]]
y_test = Test["SepalLength"]
x_test = sm.add_constant(x_test)
pred5 = fit5.predict(x_test)
RMSE5 = np.sqrt(mean_squared_error(Test.SepalLength,pred5))
print('第五个模型的预测效果：RMES=%.4f\n' %RMSE5)
# 测试-sm.ols
x2_test = Test.iloc[:,2:5]
y2_test = Test.iloc[:,[1]]
x2_test = sm.add_constant(x2_test)
pred6 = fit6.predict(x2_test)
RMSE6 = np.sqrt(mean_squared_error(Test.SepalLength,pred6))
print('第六个模型的预测效果：RMES=%.4f\n' %RMSE6)
# 测试-LinearRegression()
x3_test = Test.iloc[:,2:5]
pred4 = fit4.predict(x3_test)
RMSE4 = np.sqrt(mean_squared_error(Test.SepalLength,pred4))
print('第四个模型的预测效果：RMES=%.4f\n' %RMSE4)




# 线性回归整理-
X = iris.iloc[:,2:5]
Y = iris.iloc[:,[1]]
# 拆分训练集和测试集
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=1)
# 线性回归模型
linreg = linear_model.LinearRegression()
# 可以通过交叉验证来持续优化模型, 下面采用的是10折交叉验证
Y_pred = cross_val_predict(linreg, train_X, train_Y, cv=10)
print("10折交叉验证MSE:", mean_squared_error(train_Y, Y_pred))
print("10折交叉验证RMSE:", np.sqrt(mean_squared_error(train_Y, Y_pred)))

scores = cross_val_score(linreg,train_X,train_Y,cv=10)
print("交叉验证R方值：",scores)
print("交叉验证R方均值：",np.mean(scores))

linreg.



# 真实值与预测值的关系# 设置绘图风格
plt.style.use('ggplot')
# 设置中文正常显示
font1 = fm.FontProperties(fname='C:\Windows\Fonts\msyh.ttc')
# 散点图
plt.scatter(Test.SepalLength, pred, label = '观测点')
# 回归线
# plt.plot([Test.SepalLength.min(), Test.SepalLength.max()], [pred.min(), pred.max()], 'r--', lw=2, label = '拟合线')
# 添加轴标签和标题
plt.title(u'真实值VS.预测值',fontproperties=font1)
plt.xlabel(u'真实值',fontproperties=font1)
plt.ylabel(u'预测值',fontproperties=font1)
# 去除图边框的顶部刻度和右边刻度
plt.tick_params(top = 'off', right = 'off')
# 添加图例
plt.legend(loc = 'upper left',prop = font1)
# 图形展现
plt.show()