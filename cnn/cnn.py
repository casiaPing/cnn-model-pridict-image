# -*- coding:UTF-8 -*-
import input_data
import Image
import tensorflow as tf
import numpy as np
import input_myselfdata

#
mnist = input_data.read_data_sets("./image/pkl/", one_hot=True)
#batch_xs, batch_ys = mnist.train.next_batch(10)


# print("batch_ys",batch_ys)

# 将数字（代表类别）转换成多维矩阵
def data2vector(train_y_temp):
    result = []
    for idx in train_y_temp:
        lst = [0] * 10
        lst[10 - idx] = 1
        result.append(lst)
    result = np.reshape(result, (len(train_y_temp), 10) )
    return result


def next_batch(train_x, train_y, index_in_epoch, batch_size):
    start = index_in_epoch
    end = index_in_epoch + batch_size

    if index_in_epoch > train_x.shape[0]:
        return 0,0,0
    else:
        return train_x[start:end], train_y[start:end], end

# 改用自己的数据集进行训练和测试
train_x,train_y,valid_x,valid_y,test_x,test_y  = input_myselfdata.dataresize()
train_y = data2vector(train_y)
valid_y = data2vector(valid_y)
test_y = data2vector(test_y)

# Parameters
learning_rate = 0.001
training_iters = 5000
batch_size = 5
display_step = 10

# Network Parameters
n_input = 60*60  # MNIST data input (img shape: 28*28) 如果是象棋的话改成19*19
n_classes = 10  # MNIST total classes (0-9 digits)(19x19=361 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create model
# 下面是一个卷积层 定义 relu(wx+b)  下面是tensorflow来表示relu(wx+b)的公式
# 其中要注意参数 strides 是卷积滑动的步长 你可以配置更多的系数，
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


# 下面我们讲一讲池化层pool 池化层 不会减少输出，只会把MXNXO ，把卷积提取的特征 做进一步卷积 取MAX AVG MIN等 用局部代替整理进一步缩小特征值大小

# 下面是一个用kxk 的核做maxpooling的定义
def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(_X, _weights, _biases, _dropout):
    # Reshape input picture，如果是象棋28要改成19
    _X = tf.reshape(_X, shape=[-1, 60, 60, 1])

    # Convolution Layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k=2)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)

    # Convolution Layer
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(conv2,
                        [-1, _weights['wd1'].get_shape().as_list()[0]])  # Reshape conv2 output to fit dense layer input
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))  # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out


# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),  # 5x5 conv, 1 input, 32 outputs 卷基层
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),  # 5x5 conv, 32 inputs, 64 outputs
    # 注意输入是7*7*64
    'wd1': tf.Variable(tf.random_normal([35 * 35 * 64, 1024])),  # fully connected, 7*7*64 inputs, 1024 outputs
    'out': tf.Variable(tf.random_normal([1024, n_classes]))  # 1024 inputs, 10 outputs (class prediction)
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
# print("pred:",pred)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    index_in_epoch = 0
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        batch_xs, batch_ys,index_in_epoch = next_batch(train_x, train_y, index_in_epoch,batch_size)
        #batch_xs, batch_ys,index_in_epoch = next_batch(batch_xs, batch_ys, index_in_epoch,10)
        if index_in_epoch:
        
            print("batch_xs", batch_xs.shape)
            print("batch_ys", batch_ys.shape)

            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                # print("pred_temp:",pred_temp)
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print "Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
                loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
            
            step += 1
    saver.save(sess, './alexnet.tfmodel');
    print
    "Optimization Finished!"
    # Calculate accuracy for 256 mnist test images
    print
    "Testing Accuracy:", sess.run(accuracy,
                                  feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})

img = Image.open("input.png")
img = img.convert('L').resize((28, 28))
width, hight = img.size
img = np.asarray(img, dtype='float64') / 256.
tmp = img.reshape(1, 60 * 60)
print("tmp", mnist.test.images[:1].shape)
print("tmplbale", mnist.test.labels[:1])

with tf.Session() as sess:
    # sess.run(init)
    saver.restore(sess, './alexnet.tfmodel')
    predictions = sess.run(pred, feed_dict={x: mnist.test.images[:1], y: mnist.test.labels[:1], keep_prob: 1.})
    accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images[:1], y: mnist.test.labels[:1], keep_prob: 1.})
    print("pred", predictions, accuracy)
