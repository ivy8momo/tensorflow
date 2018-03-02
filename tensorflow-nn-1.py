# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

test_count = 20         #行：数据集数量
param_count = 5         #列：变量数

x = tf.placeholder(tf.float32, shape=[test_count,param_count])
y = tf.placeholder(tf.float32, shape=[test_count,1])
#之前的 w，因为我们使用神经网络表示了，因此不需要了，用tf.contrib.learn.DNNRegressor 来构造神经网络
feature_columns = [tf.contrib.layers.real_valued_column("x")]           #特征列：定义输入
regressor = tf.contrib.learn.DNNRegressor(feature_columns = feature_columns,
                                          hidden_units = [5,5],         #隐藏层、隐藏单元数
                                          model_dir = "E:\tensorflow\tmp\testtest")     #模型结果存储


def get_train_inputs():
    t_x = np.floor(1000 * np.random.random([test_count, param_count]), dtype=np.float32)
    t_w = np.floor([511, 231, 86, 434, 523], dtype=np.float32).reshape([param_count, 1])
    print(t_w)
    t_y = t_x.dot(t_w)
    # 第一个参数是一个字典，Key 是变量名称，Value 是变量的值转成 Tensor
    feature_cols = {'x': tf.constant(t_x)}
    # 第二个参数就是结果值，也要转成 Tensor
    return feature_cols, tf.constant(t_y)

def get_test_inputs():
    e_x = np.floor(1000 * np.random.random([test_count, param_count]), dtype=np.float32)
    t_w = np.floor([511, 231, 86, 434, 523], dtype=np.float32).reshape([param_count, 1])
    print(t_w)
    e_y = e_x.dot(t_w)
    feature_cols = {'x': tf.constant(e_x)}
    return feature_cols, tf.constant(e_y)

def get_predict_inputs():
    p_x = np.floor(1000 * np.random.random([test_count, param_count]), dtype=np.float32)
    t_w = np.floor([511, 231, 86, 434, 523], dtype=np.float32).reshape([param_count, 1])
    print(t_w)
    feature_cols = {'x': tf.constant(p_x)}
    p_y = p_x.dot(t_w)
    print("预测输入:%s" % p_x)
    print("实际结果:%s" % p_y)
    return feature_cols

LOSS_MIN_VALUE = 100

while True:
        regressor.fit(input_fn=lambda: get_train_inputs(), steps=2000)
        evaluate_result = regressor.evaluate(input_fn=lambda: get_test_inputs(),steps=1)
        print(evaluate_result)

        if evaluate_result['loss'] < LOSS_MIN_VALUE:
                break

result = str(list(regressor.predict(input_fn=lambda: get_predict_inputs())))
print("预测结果:%s" % result)