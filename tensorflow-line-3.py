# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 一元
x_train = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y_train = [[10], [11.5], [12], [13], [14.5], [15.5], [16.8], [17.3], [18], [18.7]]

X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(X, w) + b
cost = tf.reduce_mean(tf.square(Y - y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
feed = {X: x_train, Y: y_train}
cost_history = []
for i in range(10000):
    sess.run(train_step, feed_dict=feed)
    cost_history.append(sess.run(cost, feed_dict=feed))
print("w:%f" % sess.run(w))
print("b:%f" % sess.run(b))
print("cost:%f" % sess.run(cost, feed))

# 多元线性回归
x_train = [[1, 2], [2, 1], [2, 3], [3, 5], [1, 3], [4, 2], [7, 3], [4, 5], [11, 3], [8, 7]]
y_train = [[7], [8], [10], [14], [8], [13], [20], [16], [28], [26]]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(X, w) + b
cost = tf.reduce_mean(tf.square(Y - y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
feed = {X: x_train, Y: y_train}
cost_history = []
for i in range(10000):
    sess.run(train_step, feed_dict=feed)
    cost_history.append(sess.run(cost, feed_dict=feed))

print("w0:%f" % sess.run(w[0]))
print("w1:%f" % sess.run(w[1]))
print("b:%f" % sess.run(b))
print("cost:%f" % sess.run(cost, feed_dict=feed))

plt.plot(range(len(cost_history)), cost_history)
plt.axis([0, 10000, 0, np.max(cost_history)])
plt.xlabel('training epochs')
plt.ylabel('cost')
plt.title('cost history')
plt.show()

''' 测试集
pred_y = sess.run(y, feed_dict={x: X_test}) 
mse = tf.reduce_mean(tf.square(pred_y - y_test)) 
print("MSE: %.4f" % sess.run(mse))
'''

# 读取csv数据：一元
df = pd.read_csv('C:/Users/hey/Desktop/aecopd.csv', header=0)
x_train = df[['pm25']]
y_train = df[['event']]
X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(X, w) + b
cost = tf.reduce_mean(tf.square(Y - y))
train_step = tf.train.GradientDescentOptimizer(0.00000001).minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
feed = {X: x_train, Y: y_train}
cost_history = []
for i in range(10000):
    sess.run(train_step, feed_dict=feed)
    cost_history.append(sess.run(cost, feed_dict=feed))
print("w:%f" % sess.run(w))
print("b:%f" % sess.run(b))
print("cost:%f" % sess.run(cost, feed))

# 读取csv数据：多元
x_train = df[['pm25', 'pm10']]
y_train = df[['event']]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(X, w) + b
cost = tf.reduce_mean(tf.square(Y - y))
train_step = tf.train.GradientDescentOptimizer(0.00000001).minimize(cost)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
feed = {X: x_train, Y: y_train}
cost_history = []
for i in range(10000):
    sess.run(train_step, feed_dict=feed)
    cost_history.append(sess.run(cost, feed_dict=feed))

print("w0:%f" % sess.run(w[0]))
print("w1:%f" % sess.run(w[1]))
print("b:%f" % sess.run(b))
print("cost:%f" % sess.run(cost, feed_dict=feed))

plt.plot(range(len(cost_history)), cost_history)
plt.axis([0, 10000, 0, np.max(cost_history)])
plt.xlabel('training epochs')
plt.ylabel('cost')
plt.title('cost history')
plt.show()

# mult
x_train = df.iloc[:, 5:12]  # pm10  so2  no2    co  o3  temp  rh
y_train = df[['event']]
param_count = x_train.shape[1]  # 变量数

X = tf.placeholder(tf.float32, [None, param_count])
Y = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.zeros([param_count, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(X, w) + b
cost = tf.reduce_mean(tf.square(Y - y))
train_step = tf.train.GradientDescentOptimizer(0.00000001).minimize(cost)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
feed = {X: x_train, Y: y_train}
cost_history = []
for i in range(10000):
    sess.run(train_step, feed_dict=feed)
    cost_history.append(sess.run(cost, feed_dict=feed))

for i in range(param_count):
    print("w%d" % i, "is:%f" % sess.run(w[i]))
print("b:%f" % sess.run(b))
print("cost:%f" % sess.run(cost, feed_dict=feed))

plt.plot(range(len(cost_history)), cost_history)
plt.axis([0, 10000, 0, np.max(cost_history)])
plt.xlabel('training epochs')
plt.ylabel('cost')
plt.title('cost history')
plt.show()

