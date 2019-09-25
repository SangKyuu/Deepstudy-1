import os
from glob import glob
import tensorflow as tf
from PIL import Image
import numpy as np
import random

class Ramen_Dataloader():

    def __init__(self, image_size, batch_size=32, train=True, drop_last=True):
        self.image_size = image_size
        self.batch_size = batch_size
        self.dl = drop_last
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.files = glob(os.path.join(self.root_dir, 'dataset', 'train', '*\\*.png')) if train else \
            glob(os.path.join(self.root_dir, 'dataset', 'val', '*\\*.png'))

    def make_batch(self):
        random.shuffle(self.files)
        total_batch = len(self.files)//self.batch_size
        for i in range(0, total_batch):
            self.batch_file = self.files[i*self.batch_size:(i+1)*self.batch_size]
            yield self.file2Tensor()

    def file2Tensor(self):
        batch_labels = np.zeros((self.batch_size, 4))
        batch_images = np.zeros((self.batch_size, 600, 600, 3))
        for n, file in enumerate(self.batch_file):
            image, label= self._parse_data(file)
            batch_images[n,:,:,:] = image
            batch_labels[n,:] = label
        return batch_images, batch_labels

    def _parse_data(self, file):
        image_content = Image.open(file)
        label_content = int(file.split('\\')[-2])

        label = np.asarray(np.eye(4)[label_content])  # one hot encoding
        image = np.asarray(image_content.resize((600, 600)))
        return image, label

    def __len__(self):
        return len(self.files)

class Ramen_model():
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 600, 600, 3])
            self.Y = tf.placeholder(tf.float32, [None, 4])

            # L1 ImgIn shape=(?, 600, 600, 3)
            W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
            #    Conv     -> (?, 600, 600, 32)
            #    Pool     -> (?, 300, 300, 32)
            L1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            # L2 ImgIn shape=(?, 300, 300, 32)
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            #    Conv      ->(?, 300, 300, 64)
            #    Pool      ->(?, 150, 150, 64)
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

            # L3 ImgIn shape=(?, 150, 150, 64)
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            #    Conv      ->(?, 150, 150, 128)
            #    Pool      ->(?, 75, 75, 128)
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding='SAME')
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)

            # L4 ImgIn shape=(?, 75, 75, 128)
            W4 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01))
            #    Conv      ->(?, 75, 75, 128)
            #    Pool      ->(?, 38, 38, 128)
            L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
            L4 = tf.nn.relu(L4)
            L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding='SAME')
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            #    Reshape   ->(?, 75 * 75 * 128) # Flatten them for FC
            L4_flat = tf.reshape(L4, [-1, 128 * 38 * 38])

            # L4 FC 4x4x128 inputs -> 625 outputs
            W5 = tf.get_variable("W5", shape=[128 * 38 * 38, 6125],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([6125]))
            L5 = tf.nn.relu(tf.matmul(L4_flat, W5) + b5)
            L5 = tf.nn.dropout(L5, keep_prob=self.keep_prob)

            # L5 Final FC 625 inputs -> 10 outputs
            W6 = tf.get_variable("W6", shape=[6125, 4],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b6 = tf.Variable(tf.random_normal([4]))
            self.logits = tf.matmul(L5, W6) + b6

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

batch_size = 32
a = Ramen_Dataloader(600, batch_size)

with tf.Session() as sess:
    m1 = Ramen_model(sess, "m1")
    sess.run(tf.global_variables_initializer())

    for epoch in range(0,10):
        avg_cost = 0
        total_batch = int(a.__len__() / batch_size)

        for batch in a.make_batch():
            c, _ = m1.train(batch[0], batch[1])
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
