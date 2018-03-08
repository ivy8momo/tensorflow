# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 用污染物与肺病门诊人数，测试DNN模型
df = pd.read_csv('C:/Users/hey/Desktop/aecopd.csv', header=0)
LABEL_COLUMN = ['event']
FEATURE_COLUMNS = ['pm25','pm10','o3','so2','time']

x_train, x_test, y_train, y_test = train_test_split(df[FEATURE_COLUMNS], df[LABEL_COLUMN], test_size=0.3)
print(' 训练集：{}\n 测试集：{}'.format(x_train.shape, x_test.shape))

feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURE_COLUMNS]
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                          hidden_units=[64, 128],
                                          model_dir='E:/tensorflow/tmp/test3')

def input_fn(df, label):
    feature_cols = {k: tf.constant(df[k].values) for k in FEATURE_COLUMNS}
    label = tf.constant(label.values)
    return feature_cols, label
def train_input_fn():
    return input_fn(x_train, y_train)
def test_input_fn():
    return input_fn(x_test, y_test)

regressor.fit(input_fn=train_input_fn, steps=5000)
ev = regressor.evaluate(input_fn=test_input_fn, steps=1)
print('ev: {}'.format(ev))
predict = regressor.predict(input_fn=test_input_fn, as_iterable=False)
print('预测值: %s' % predict[:10])
print('实际值: %s'% y_test[:10])

plt.plot(range(len(y_test)),y_test,'b')
plt.plot(range(len(predict)),predict,'r--')
plt.show()
