# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep

# xavier初始化器，把权重初始化在low和high范围内(满足N(0,2/Nin+Nout))
def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low ,maxval=high ,dtype=tf.float32)
# 外层管理类
class  AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input,n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weight = self.initalize_weight()
        self.weight = network_weight
        # 噪声：scale*tf.random_normal((n_input,))，（x+scale）*w1+b1
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)),
                                                     self.weight['w1']),
                                           self.weight['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weight['w2']),self.weight['b2'])
        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # 隐藏层有激活函数，用Xavier设置一个合适的w1,b1和输出层w2,b2设置为0
    def _initialize_weights(self):
        all_weight = dict()
        all_weight['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weight['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weight['w1'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input], dtype=tf.float32))
        all_weight['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weight

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    # 运行cost节点
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # 返回隐藏层输出结果
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})
    def generate(self,hidden = None):
        if hidden is None:hidden = np.random.normal(size=self.weight['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})
    def getWeight(self):
        return self.sess.run(self.weight['w1'])
    def getBiases(self):
        return self.sess.run(self.weight['b1'])

# 数据零均值，特征方差归一化处理
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test
# 获取批量文本的策略,
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

mnist = input_data.read_data_sets('MNIST',one_hot=True)
X_train,X_test = standard_scale(mnist.train.images,mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                               n_hidden= 200,
                                               transfer_function=tf.nn.softplus,
                                               optimizer= tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train,batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples*batch_size

    if epoch % display_step == 0:
        print('Epoch:','%04d' % (epoch+1),'cost=','{:.9f}'.format(avg_cost))
print('total cost : '+ str(autoencoder.calc_total_cost(X_test)))