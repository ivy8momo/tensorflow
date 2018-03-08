# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 用tf自带波斯顿房价数据测试DNN模型
boston=load_boston()
print("data of boston:",boston.data.shape)
print("target of boston:",boston.target.shape)
boston_df = pd.DataFrame(np.c_[boston.data, boston.target], columns=np.append(boston.feature_names, 'MEDV'))
LABEL_COLUMN = ['MEDV']
FEATURE_COLUMNS = [f for f in boston_df if not f in LABEL_COLUMN]

x_train, x_test, y_train, y_test = train_test_split(boston_df[FEATURE_COLUMNS], boston_df[LABEL_COLUMN], test_size=0.3)
print(' 训练集：{}\n 测试集：{}'.format(x_train.shape, x_test.shape))


feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURE_COLUMNS]
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                          hidden_units=[64, 128],
                                          model_dir='E:/tensorflow/tmp/test1')

def input_fn(df, label):
    feature_cols = {k: tf.constant(df[k].values) for k in FEATURE_COLUMNS}
    label = tf.constant(label.values)
    return feature_cols, label

def train_input_fn():
    '''训练阶段使用的 input_fn'''
    return input_fn(x_train, y_train)

def test_input_fn():
    '''测试阶段使用的 input_fn'''
    return input_fn(x_test, y_test)

regressor.fit(input_fn=train_input_fn, steps=5000)
ev = regressor.evaluate(input_fn=test_input_fn, steps=1)
print('ev: {}'.format(ev))
predict = regressor.predict(input_fn=test_input_fn, as_iterable=False)
print('预测值: %s' % predict[:10])
print('实际值: %s'% y_test[:10])


