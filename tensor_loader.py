from glob import glob
import os
from PIL import Image
import numpy as np
import tensorflow as tf

root_dir = os.path.dirname(os.path.realpath(__file__))

train_files = glob(os.path.join(root_dir, 'dataset', 'train', '*/*.png'))
test_files = glob(os.path.join(root_dir, 'dataset', 'val', '*/*.png'))

Epoch = 7

def make_dataset(files):
    file_len = len(files)

    images = np.zeros((file_len,256,256,3))
    labels = np.zeros((file_len,4))

    for i, file in enumerate(files):
        image = Image.open(file)
        label = int(file.split('/')[-2])

        label = np.asarray(np.eye(4)[label])  # one hot encoding
        image = np.asarray(image.resize((256, 256)))

        images[i,:,:,:] = image
        labels[i,:] = label
    return images, labels


with tf.variable_scope('model_cnn'):
    # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
    # for testing
    keep_prob = tf.placeholder(tf.float32)

    batch_size = tf.placeholder(tf.int64)

    X = tf.placeholder(tf.float32, [None, 256, 256, 3])
    Y = tf.placeholder(tf.float32, [None, 4])

    dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().batch(batch_size)

    iter = dataset.make_initializable_iterator()
    image, label = iter.get_next()

    # L1 ImgIn shape=(?, 256, 256, 3)
    W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
    #    Conv     -> (?, 256, 256, 32)
    #    Pool     -> (?, 128, 128, 32)
    L1 = tf.nn.conv2d(image, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    # L2 ImgIn shape=(?, 128, 128, 32)
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    #    Conv      ->(?, 128, 128, 64)
    #    Pool      ->(?, 64, 64, 64)
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

    # L3 ImgIn shape=(?, 64, 64, 64)
    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    #    Conv      ->(?, 64, 64, 128)
    #    Pool      ->(?, 32, 32, 128)
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME')
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

    # L4 ImgIn shape=(?, 32, 32, 128)
    W4 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
    #    Conv      ->(?, 32, 32, 256)
    #    Pool      ->(?, 16, 16, 256)
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME')
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

    #    Reshape   ->(?, 16 * 16 * 256) # Flatten them for FC
    L4_flat = tf.reshape(L4, [-1, 256 * 16 * 16])

    # L4 FC 16x16x256 inputs -> 1025 outputs
    W5 = tf.get_variable("W5", shape=[256 * 16 * 16, 1025],
                         initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([1025]))
    L5 = tf.nn.relu(tf.matmul(L4_flat, W5) + b5)
    L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

    # L5 Final FC 1025 inputs -> 4 outputs
    W6 = tf.get_variable("W6", shape=[1025, 4],
                         initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.Variable(tf.random_normal([4]))
    logits = tf.matmul(L5, W6) + b6

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=label))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.001).minimize(cost)

    correct_prediction = tf.equal(
        tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.device('/gpu:0'):

    train_img, train_lb = make_dataset(train_files)
    test_img, test_lb = make_dataset(test_files)


    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(iter.initializer, feed_dict={
                            X: train_img, Y: train_lb, batch_size:32})
        for i in range(Epoch):
            avg_cost = 0
            total_batch = int(len(train_files) / 32)

            for j in range(total_batch):
                c, _ = sess.run([cost, optimizer], feed_dict={keep_prob: 0.7})
                avg_cost += c / total_batch

            print('Epoch:', '%04d' % (i + 1), 'cost =', '{:.9f}'.format(avg_cost))

        sess.run(iter.initializer, feed_dict={
            X: test_img, Y: test_lb, batch_size: len(test_img)})
        acc = sess.run(accuracy,feed_dict={keep_prob:1})
        print(acc)

















