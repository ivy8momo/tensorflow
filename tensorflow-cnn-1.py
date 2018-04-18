# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 第一部分读取图片数据、创建session

mnist_data = "E:\gitworkspace\gitworkspace"
MNIST_DATASET = input_data.read_data_sets(mnist_data, one_hot=True)
print(MNIST_DATASET.train.images.shape,MNIST_DATASET.train.labels.shape)
sess=tf.InteractiveSession()

# 第二部分，函数声明：初始权重和偏置、卷积大小、池化
# 权重：正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0，截断正态分布标准差为0.1的噪声
# 偏置：为偏置加0.1，防止0值出现-死亡节点
# 卷积：tf.nn.conv2d是一个2维卷积函数，strides是步长，各个方向均为1，SAME：边缘外自动补0，遍历相乘
# 池化层：采用kernel大小为2 * 2，步数也为2，周围补0，取最大值，数据量缩小了4倍

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# 第三部分，定义输入输出结构
# x：声明一个占位符，None表示输入图片的数量不定，28*28图片分辨率
# x_image：按照 conv2d中input的格式来reshape处理xs reshape成28*28*1，因为是灰色图片，所以通道是1，-1代表图片数量不定

x = tf.placeholder(tf.float32, [None, 28*28])
x_image = tf.reshape(x, [-1,28,28,1])


# 第四部分，搭建网络，定义算法公式

# 第一层
# 权重：是在每个5*5的patch中算出32个特征，1是指输入通道，黑白为1，彩色RGB为3
# 偏置：32个偏置
# 卷积：图片乘以卷积核，并加上偏执量，卷积结果28x28x32
# 池化结果：14x14x32

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

# 第二层
# 卷积核5*5，输入通道为32，输出通道为64
# 偏置：64个
# 卷积前图像的尺寸为 ?*14*14*32， 卷积后为14*14*64，h_pool1是第一层池化后结果
# c池化：第二个池化层池化后，输出的图像尺寸为7*7*64

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

# 第三层 是个全连接层,输入维数7*7*64, 输出维数为1024
# 权重：二维张量，第一个参数7*7*64的patch，也可以认为是只有一行7*7*64个数据的卷积，第二个参数代表卷积个数共1024
# 偏置：1024个
# 改变池化结果形状：将第二层卷积池化结果reshape成只有一行7*7*64个数据
# 卷积：结果是1*1*1024，单行乘以单列等于1*1矩阵，matmul实现最基本的矩阵相乘
# drop：这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第四层，输入1024维，输出10维，也就是具体的0~9分类
# 最后的分类，结果为1*1*10 softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   #softmax层

# 类别是0-9总共10个类别，对应输出分类结果
y_actual = tf.placeholder(tf.float32, shape=[None, 10])

# 第五部分，定义loss，选定优化方法，交叉熵、梯度下降法
cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

# 第六部分，训练数据及测试
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                 #精确度计算
sess.run(tf.global_variables_initializer())
for i in range(2000):
    batch = MNIST_DATASET.train.next_batch(50)
    if i%100 == 0:
        train_acc = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_acc))
    train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})

test_acc=accuracy.eval(feed_dict={x: MNIST_DATASET.test.images, y_actual: MNIST_DATASET.test.labels, keep_prob: 1.0})
print("test accuracy %g"%test_acc)
