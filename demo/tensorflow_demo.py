# coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def hello_world():
    """

    """
    # create data
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.1 + 0.3
    #    create tensor flow structure start
    weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    biases = tf.Variable(tf.zeros([1]))
    y = weights * x_data + biases
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    # 初始化
    init = tf.initialize_all_variables()
    #     create tensor flow structure end
    # 激活会话
    sess = tf.Session()
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(weights), sess.run(biases))
    sess.close()


def session_demo():
    """

    """
    matrix1 = tf.constant([[3, 3]])
    matrix2 = tf.constant([[2], [2]])
    product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1,m2)

    # method1
    sess = tf.Session()
    result = sess.run(product)
    print(result)
    sess.close()

    # method2
    with tf.Session() as sess:
        result2 = sess.run(product)
        print(result2)


def variable_demo():
    """

    """
    state = tf.Variable(0, name='counter')
    print(state.name)
    one = tf.constant(1)

    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))


def place_holder_demo():
    """

    """
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)

    output = tf.multiply(input1, input2)
    with tf.Session() as sess:
        print(sess.run(output, feed_dict={
            input1: [7.],
            input2: [2.]
        }))


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None, keep_prob=0.6):
    """

    """
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/wieghts', weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(inputs, weights) + biases
            wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob=keep_prob)
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


def my_network():
    """
        # tensorboard --logdir='logs/'
    """
    # make up some real data
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    # define placeholder for inputs to network
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
    # add hidden layer
    l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
    # add output layer
    prediction = add_layer(l1, 10, 1, n_layer=2)
    # the error between prediction and real data
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    merge = tf.summary.merge_all()
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('logs/', sess.graph)
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.scatter(x_data, y_data)
        # plt.ion()
        # plt.show()
        for i in range(1000):
            sess.run(train_step, feed_dict={
                xs: x_data,
                ys: y_data
            })
            if i % 50 == 0:
                print(sess.run(loss, feed_dict={
                    xs: x_data,
                    ys: y_data
                }))
                # prediction_value = sess.run(prediction, feed_dict={
                #     xs: x_data
                # })
                # try:
                #     ax.lines.remove(lines[0])
                # except Exception:
                #     pass
                # lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
                # plt.pause(0.1)
                result = sess.run(merge, feed_dict={xs: x_data, ys: y_data})
                writer.add_summary(result, i)


def network_plot():
    """

    """
    # make up some real data
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    # define placeholder for inputs to network
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
    # add hidden layer
    l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
    # add output layer
    prediction = add_layer(l1, 10, 1, n_layer=2)
    # the error between prediction and real data
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x_data, y_data)
        plt.ion()
        plt.show()
        for i in range(1000):
            sess.run(train_step, feed_dict={
                xs: x_data,
                ys: y_data
            })
            if i % 50 == 0:
                print(sess.run(loss, feed_dict={
                    xs: x_data,
                    ys: y_data
                }))
                prediction_value = sess.run(prediction, feed_dict={
                    xs: x_data
                })
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
                plt.pause(0.1)


def compute_accuracy(v_xs, v_ys, xs, ys, prediction, sess):
    """

    :param sess:
    :param prediction:
    :param ys:
    :param xs:
    :param v_xs:
    :param v_ys:
    :return:
    """
    y_pre = sess.run(prediction, feed_dict={
        xs: v_xs
    })
    correct_prediction = tf.equal(tf.math.argmax(y_pre, 1), tf.math.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={
        xs: v_xs,
        ys: v_ys
    })
    return result
    pass


def classification_demo():
    """

    """
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784])  # 28*28
    ys = tf.placeholder(tf.float32, [None, 10])

    # add output layer
    prediction = add_layer(xs, 784, 10, n_layer=1, activation_function=tf.nn.softmax)

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={
            xs: batch_xs,
            ys: batch_ys
        })
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels, xs, ys, prediction, sess))
    sess.close()


def overfitting_demo():
    digits = load_digits()
    X = digits.data
    y = digits.target
    y = LabelBinarizer().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # define placeholder for inputs to network
    keep_prob = tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, [None, 64])  # 8*8
    ys = tf.placeholder(tf.float32, [None, 10])

    # add output layer
    l1 = add_layer(xs, 64, 100, n_layer=1, activation_function=tf.nn.tanh, keep_prob=keep_prob)
    prediction = add_layer(l1, 100, 10, n_layer=2, activation_function=tf.nn.softmax, keep_prob=keep_prob)

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)
    merge = tf.summary.merge_all()
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    for i in range(1000):
        sess.run(train_step, feed_dict={
            xs: X_train,
            ys: y_train,
            keep_prob: 0.6
        })
        if i % 50 == 0:
            train_result = sess.run(merge, feed_dict={
                xs: X_train,
                ys: y_train,
                keep_prob: 1
            })
            test_result = sess.run(merge, feed_dict={
                xs: X_test,
                ys: y_test,
                keep_prob: 1
            })
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)
    sess.close()


def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn_demo():
    # define placeholder for inputs to network
    keep_prob = tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, [None, 784])  # 28*28
    ys = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(xs, [-1, 28, 28, 1])  # channel 1
    # print(x_image.shape) #[n_samples,28,28,1]
    # conv1 layer
    W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5*5, in size 1, out size 32
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28*28*32
    h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14*32
    # conv2 layer
    W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5*5, in size 32, out size 64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(x_image, W_conv2) + b_conv2)  # output size 14*14*64
    h_pool2 = max_pool_2x2(h_conv2)  # output size 7*7*64
    # func1 layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # [n_samples,7,7,64]->[n_samples, 7*7*64]
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
    # func2 layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # classification

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    merge = tf.summary.merge_all()
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={
            xs: batch_xs,
            ys: batch_ys
        })
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels, xs, ys, prediction, sess))
    sess.close()


def save_file_demo():
    # remember to define the same dtype and shape when restore
    W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
    b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, 'my_net/save_net.ckpt')
        print("Save to path:" + save_path)


def restore_variables():
    # redefine the same shape and same type for your variables
    W = tf.Variable(np.arrange(6).reshape((2, 3)), dtype=tf.float32, name='weights')
    b = tf.Variable(np.arrange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

    # no need init step
    saver = tf.train.Saver()
    saver.restore()
    with tf.Session() as sess:
        saver.restore(sess, 'my_net/save_net.ckpt')
        print('weights:', sess.run(W))
        print('biases:', sess.run(b))


def RNN(X, weights, biases):
    # hidden layer for input to cell

    # X(128 batch, 28 steps, 28 inputs)
    # ->(128*28, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # ->(128 batch*28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # ->(128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_unit])

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unit, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    results = tf.matmul(states[1], weights['out']) + biases['out']

    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))  # states is the last output
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results


pass

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28  # image 28*28
n_steps = 28
n_hidden_unit = 128  # neurons in hidden layer
n_classes = 10  # classes (0-9 digits)


def rnn_classify_demo():
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])

    weights = {
        # (28,128)
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_unit])),
        # (128,10)
        'out': tf.Variable(tf.random_normal([n_hidden_unit, n_classes]))
    }

    biases = {
        # (128,)
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unit, ])),
        # (10,)
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    }

    pred = RNN(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step * batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys
            })
            if step % 20 == 0:
                print(sess.run(accuracy, feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }))


BATCH_START = 0
TIME_STEP = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006
BATCH_START_TEST = 0


def get_batch():
    global BATCH_START, TIME_STEP
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEP * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEP))
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEP
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step,in_size)
        # Ws(in_size,cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs(cell_size,)
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y=(batch*nsteps,cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y -> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
        pass

    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell, self.l_in_y,
                                                                     initial_state=self.cell_init_state,
                                                                     time_major=False)
        pass

    def add_output_layer(self):
        # shape=[batch*steps,cell_size]
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')  # (batch*n_step,in_size)
        # Ws(cell_size,output_size)
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        # bs(output_size,)
        bs_out = self._bias_variable([self.output_size, ])
        # l_in_y=(batch*nsteps,cell_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out
        pass

    def compute_cost(self):
        self.cost = tf.losses.mean_squared_error(labels=tf.reshape(self.ys, [-1]),
                                                 predictions=tf.reshape(self.pred, [-1]))
        tf.summary.scalar('loss', self.cost)
        pass

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


def rnn_regress_demo():
    """

    """
    model = LSTMRNN(TIME_STEP, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs', sess.graph)

    sess.run(tf.initialize_all_variables())

    for i in range(200):
        seq, res, xs = get_batch()
        if i == 0:
            feed_dict = {
                model.xs: seq,
                model.ys: res
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state
            }
        _, cost, state, pred = sess.run([model.train_op, model.cost, model.cell_final_state, model.pred],
                                        feed_dict=feed_dict)
        if i % 20 == 0:
            print('cost', round(cost, 4))
            result = sess.run(merged, feed_dict=feed_dict)
            writer.add_summary(result, i)
    pass


# visualize decoder setting
# parameter
learning_rate = 0.01
training_epochs = 5
batch_size = 256
display_step = 1
example_to_show = 10

# network parameters
n_input = 784

# hidden layer settings
n_hidden_1 = 256
n_hidden_2 = 128

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
}


def auto_encoder_demo():
    # Hyper Parameters
    BATCH_SIZE = 64
    LR = 0.002  # learning rate
    N_TEST_IMG = 5

    # Mnist digits
    mnist = input_data.read_data_sets('./mnist', one_hot=False)  # use not one-hotted target data
    test_x = mnist.test.images[:200]
    test_y = mnist.test.labels[:200]

    # plot one example
    print(mnist.train.images.shape)  # (55000, 28 * 28)
    print(mnist.train.labels.shape)  # (55000, 10)
    plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
    plt.title('%i' % np.argmax(mnist.train.labels[0]))
    plt.show()

    # tf placeholder
    tf_x = tf.placeholder(tf.float32, [None, 28 * 28])  # value in the range of (0, 1)

    # encoder
    en0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
    en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
    en2 = tf.layers.dense(en1, 12, tf.nn.tanh)
    encoded = tf.layers.dense(en2, 3)

    # decoder
    de0 = tf.layers.dense(encoded, 12, tf.nn.tanh)
    de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
    de2 = tf.layers.dense(de1, 128, tf.nn.tanh)
    decoded = tf.layers.dense(de2, 28 * 28, tf.nn.sigmoid)

    loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
    train = tf.train.AdamOptimizer(LR).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # initialize figure
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
    plt.ion()  # continuously plot

    # original data (first row) for viewing
    view_data = mnist.test.images[:N_TEST_IMG]
    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(());
        a[0][i].set_yticks(())

    for step in range(8000):
        b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
        _, encoded_, decoded_, loss_ = sess.run([train, encoded, decoded, loss], {tf_x: b_x})

        if step % 100 == 0:  # plotting
            print('train loss: %.4f' % loss_)
            # plotting decoded image (second row)
            decoded_data = sess.run(decoded, {tf_x: view_data})
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(());
                a[1][i].set_yticks(())
            plt.draw();
            plt.pause(0.01)
    plt.ioff()

    # visualize in 3D plot
    view_data = test_x[:200]
    encoded_data = sess.run(encoded, {tf_x: view_data})
    fig = plt.figure(2);
    ax = Axes3D(fig)
    X, Y, Z = encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2]
    for x, y, z, s in zip(X, Y, Z, test_y):
        c = cm.rainbow(int(255 * s / 9));
        ax.text(x, y, z, s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max());
    ax.set_ylim(Y.min(), Y.max());
    ax.set_zlim(Z.min(), Z.max())
    plt.show()
    pass


if __name__ == '__main__':
    hello_world()
    # session_demo()
    # variable_demo()
    # place_holder_demo()
    # my_network()
    # classification_demo()
    # overfitting_demo()
    # cnn_demo()
    # save_file_demo()
    # rnn_classify_demo()
    rnn_regress_demo()
    # auto_encoder_demo()
