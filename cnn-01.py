# -*- coding: utf-8 -*-
import tensorflow as tf

# test-1
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print(result)        # 输出“Tensor("add:0", shape=(2,), dtype=float32) ”
sess = tf.Session()
print(sess.run(result))    # 输出“[ 3.  5.]”
sess.close()

# test-2 mnist
import os
import gzip
import numpy as np
import urllib
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# download data-mnist
url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
filename = 'mnist.pkl.gz'
if not os.path.exists(filename):
    print("Downloading MNIST dataset...")
    urllib.request.urlretrieve(url, filename)
with gzip.open(filename, 'rb') as f:
    data = data = pickle.load(f, encoding = 'iso-8859-1')

# cut data-mnist
X_train, y_train = data[0]
X_val, y_val = data[1]
X_test, y_test = data[2]
X_train = X_train.reshape((-1, 1, 28, 28))
X_val = X_val.reshape((-1, 1, 28, 28))
X_test = X_test.reshape((-1, 1, 28, 28))
y_train = y_train.astype(np.uint8)
y_val = y_val.astype(np.uint8)
y_test = y_test.astype(np.uint8)

# plt image
plt.imshow(X_train[0][0], cmap=cm.binary)
print(y_train[0])
plt.imshow(X_train[1][0], cmap=cm.binary)
print(y_train[1])

import tensorflow.examples.tutorials.mnist.input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


