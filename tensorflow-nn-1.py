# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
# 用随机生成的小样本测试DNN模型
tf.logging.set_verbosity(tf.logging.ERROR)              #日志级别设置成 ERROR，避免干扰
np.set_printoptions(threshold='nan')                    #打印内容不限制长度

param_count = 5         #变量数
test_count = 20         #每次训练的样本数

# 要求的值
t_w = np.floor([511,231,86,434,523],dtype=np.float32).reshape([param_count,1])

#x 是输入量，对应 t_x，用于训练输入，在训练过程中，由外部提供，因此是 placeholder 类型
x = tf.placeholder(tf.float32,shape=[test_count,param_count])
y = tf.placeholder(tf.float32,shape=[test_count,1])

#w 是要求的各个参数的权重，是目标输出，对应 t_w
w = tf.Variable(np.zeros(param_count,dtype=np.float32).reshape((param_count,1)), tf.float32)

feature_columns = [tf.contrib.layers.real_valued_column("")]
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                                                                hidden_units=[5,5],
                                                                                model_dir="/tmp/test2")

LOSS_MIN_VALUE = 20000

while True:
        t_x = np.floor(1000 * np.random.random([test_count,param_count]),dtype=np.float32)
        t_y = t_x.dot(t_w)
        regressor.fit(x=t_x,y=t_y, steps=2000)
        # 测试，测试结果.evaluate使一个字典，包括最终损失和迭代步数
        e_x = np.floor(1000 * np.random.random([test_count,param_count]),dtype=np.float32)
        e_y = e_x.dot(t_w)
        evaluate_result = regressor.evaluate(x=e_x,y=e_y)
        print(evaluate_result)

        if evaluate_result['loss'] < LOSS_MIN_VALUE:
                break

p_x = np.floor(1000 * np.random.random([test_count,param_count]),dtype=np.float32)
p_y = p_x.dot(t_w)
print("预测输入:%s" % p_x)
print("实际结果:%s" % p_y)
print("预测值:" % str(list(regressor.predict(p_x))))
