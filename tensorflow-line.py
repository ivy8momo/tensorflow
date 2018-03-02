# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
import numpy as np

# tf.logging.set_verbosity(tf.logging.ERROR)              #日志级别设置成 ERROR，避免干扰
# np.set_printoptions(threshold='nan')                    #打印内容不限制长度

np.random.seed(123)
t_x = np.floor(10 * np.random.random([5]),dtype=np.float32) #造一组随机输入
t_y = t_x * 3.0 + 8.0                                     # 根据输入计算输出
'''
df = pd.read_csv('C:/Users/hey/Desktop/aecopd.csv', header=0, index_col=0)
t_x = df[['pm25']]
t_y = df[['event']]
'''

x = tf.placeholder(tf.float32)                          # 输入量，在 TensorFlow 中以占位符 placeholder 表示
y = tf.placeholder(tf.float32)
a = tf.Variable(1.0)                                    #输出量，在 TensorFlow 中以Variable  表示
b = tf.Variable(1.0)
curr_y = x * a + b                                      #定义关系
loss = tf.reduce_sum(tf.square(curr_y - y))             #定义损失函数，实际输出数据和训练输出数据的方差
optimizer = tf.train.GradientDescentOptimizer(0.001)    #定义求解最小损失函数方法——梯度下降（Gradient descent），学习率0.001
# optimizer = tf.train.GradientDescentOptimizer(0.00000001)
train = optimizer.minimize(loss)                        #训练的结果是使得损失函数最小


sess = tf.Session()                                     #创建 Session
sess.run(tf.global_variables_initializer())             #变量初始化
for i in range(3000):
        sess.run(train, {x:t_x, y:t_y})
print(sess.run([a,b,loss],{x:t_x, y:t_y}))

''' 用误差精度控制迭代次数
LOSS_MIN_VALUE = tf.constant(1e-2)               #达到此精度的时候结束训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())
run_count = 0
last_loss = 0
while True:
        run_count  = run_count+1
        sess.run(train, {x:t_x, y:t_y})
        curr_loss,is_ok = sess.run([loss,loss < LOSS_MIN_VALUE],{x:t_x, y:t_y})
        if last_loss == curr_loss:
                break
        last_loss = curr_loss
        if is_ok:
                break
print("运行%d 次,loss=%s" % (run_count,curr_loss))
print(sess.run([a,b,loss],{x:t_x, y:t_y}))
'''

exit(0)