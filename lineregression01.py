# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.cross_validation import train_test_split
from sklearn.metrics import  mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

iris = pd.read_csv('e:\data\iris.csv')

b = iris.describe()
np.savetxt('iris-describe.txt',b)

Train,Test = train_test_split(iris,train_size=0.8, random_state=1234)
fit = smf.ols('SepalLength~SepalWidth+PetalWidth',data=Train).fit()
print(fit.summary())
fit2 = smf.ols('SepalLength~SepalWidth+PetalWidth+PetalLength',data=Train).fit()
print(fit2.summary())

pred = fit.predict(exog = Test)
pred2 = fit2.predict(exog = Test)
RMSE = np.sqrt(mean_squared_error(Test.SepalLength,pred))
RMSE2 = np.sqrt(mean_squared_error(Test.SepalLength,pred2))
print('第一个模型的预测效果：RMES=%.4f\n' %RMSE)
print('第二个模型的预测效果：RMES=%.4f\n' %RMSE2)

# 真实值与预测值的关系# 设置绘图风格
plt.style.use('ggplot')
# 设置中文正常显示
font1 = fm.FontProperties(fname='C:\Windows\Fonts\msyh.ttc')
# 散点图
plt.scatter(Test.SepalLength, pred, label = '观测点')
# 回归线
plt.plot([Test.SepalLength.min(), Test.SepalLength.max()], [pred.min(), pred.max()], 'r--', lw=2, label = '拟合线')
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