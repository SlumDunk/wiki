import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, SimpleRNN, LSTM, TimeDistributed, \
    Input
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam
from keras.models import Model, load_model, Sequential
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def regression_demo():
    # create some data
    X = np.linspace(-1, 1, 200)
    np.random.shuffle(X)  # randomize the data
    Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))
    # plot data
    plt.scatter(X, Y)
    plt.show()
    X_train, Y_train = X[:160], Y[:160]  # first 160 data points
    X_test, Y_test = X[160:], Y[160:]  # last 40 data points
    # build a neural network from the 1st layer to the last layer
    model = Sequential()
    model.add(Dense(units=1, input_dim=1))
    # choose loss function and optimizing method
    model.compile(loss='mse', optimizer='sgd')
    # training
    print('Training -----------')
    for step in range(301):
        cost = model.train_on_batch(X_train, Y_train)
        if step % 100 == 0:
            print('train cost: ', cost)
    # test
    print('\nTesting ------------')
    cost = model.evaluate(X_test, Y_test, batch_size=40)
    print('test cost:', cost)
    W, b = model.layers[0].get_weights()
    print('Weights=', W, '\nbiases=', b)
    # plotting the prediction
    Y_pred = model.predict(X_test)
    plt.scatter(X_test, Y_test)
    plt.plot(X_test, Y_pred)
    plt.show()


def classify_demo():
    # download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
    # X shape (60,000 28x28), y shape (10,000, )
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # data pre-processing
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
    y_train = np_utils.to_categorical(y_train, num_classes=10)  # one hot 编码
    y_test = np_utils.to_categorical(y_test, num_classes=10)
    # Another way to build your neural net
    model = Sequential([
        Dense(32, input_dim=784),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])
    # Another way to define your optimizer
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # We add metrics to get more results you want to see
    model.compile(optimizer=rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('Training ------------')
    # Another way to train the model
    model.fit(X_train, y_train, epochs=2, batch_size=32)
    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(X_test, y_test)
    print('test loss: ', loss)
    print('test accuracy: ', accuracy)


def cnn_demo():
    # download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
    # training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # data pre-processing
    X_train = X_train.reshape(-1, 1, 28, 28) / 255.
    X_test = X_test.reshape(-1, 1, 28, 28) / 255.
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    # Another way to build your CNN
    model = Sequential()

    # Conv layer 1 output shape (32, 28, 28)
    model.add(Convolution2D(
        batch_input_shape=(None, 1, 28, 28),
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',  # Padding method
        data_format='channels_first',
    ))
    model.add(Activation('relu'))

    # Pooling layer 1 (max pooling) output shape (32, 14, 14)
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',  # Padding method
        data_format='channels_first',
    ))

    # Conv layer 2 output shape (64, 14, 14)
    model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
    model.add(Activation('relu'))

    # Pooling layer 2 (max pooling) output shape (64, 7, 7)
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

    # Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))

    # Fully connected layer 2 to shape (10) for 10 classes
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # Another way to define your optimizer
    adam = Adam(lr=1e-4)

    # We add metrics to get more results you want to see
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('Training ------------')
    # Another way to train the model
    model.fit(X_train, y_train, epochs=1, batch_size=64, )

    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(X_test, y_test)

    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)


def rnn_classifier_demo():
    TIME_STEPS = 28  # same as the height of the image
    INPUT_SIZE = 28  # same as the width of the image
    BATCH_SIZE = 50
    BATCH_INDEX = 0
    OUTPUT_SIZE = 10
    CELL_SIZE = 50
    LR = 0.001

    # download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
    # X shape (60,000 28x28), y shape (10,000, )
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # data pre-processing
    X_train = X_train.reshape(-1, 28, 28) / 255.  # normalize
    X_test = X_test.reshape(-1, 28, 28) / 255.  # normalize
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    # build RNN model
    model = Sequential()

    # RNN cell
    model.add(SimpleRNN(
        # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
        # Otherwise, model.evaluate() will get error.
        batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),  # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
        output_dim=CELL_SIZE,
        unroll=True,
    ))

    # output layer
    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('softmax'))

    # optimizer
    adam = Adam(LR)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # training
    for step in range(4001):
        # data shape = (batch_num, steps, inputs/outputs)
        X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
        Y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]
        cost = model.train_on_batch(X_batch, Y_batch)
        BATCH_INDEX += BATCH_SIZE
        BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

        if step % 500 == 0:
            cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
            print('test cost: ', cost, 'test accuracy: ', accuracy)


BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


def get_batch():
    global BATCH_START, BATCH_SIZE
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (
            10 * np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


def rnn_regress_demo():
    model = Sequential()
    # build a LSTM RNN
    model.add(LSTM(
        batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),  # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
        output_dim=CELL_SIZE,
        return_sequences=True,  # True: output at all steps. False: output as last step.
        stateful=True,  # True: the final state of batch1 is feed into the initial state of batch2
    ))
    # add output layer
    model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
    adam = Adam(LR)
    model.compile(optimizer=adam,
                  loss='mse', )

    print('Training ------------')
    for step in range(501):
        # data shape = (batch_num, steps, inputs/outputs)
        X_batch, Y_batch, xs = get_batch()
        cost = model.train_on_batch(X_batch, Y_batch)
        pred = model.predict(X_batch, BATCH_SIZE)
        plt.plot(xs[10, :], Y_batch[10].flatten(), 'r', xs[10, :], pred[10].flatten(), 'b--')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.1)
        if step % 10 == 0:
            print('train cost: ', cost)


def auto_encoder_demo():
    # download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
    # X shape (60,000 28x28), y shape (10,000, )
    (x_train, _), (x_test, y_test) = mnist.load_data()

    # data pre-processing
    x_train = x_train.astype('float32') / 255. - 0.5  # minmax_normalized
    x_test = x_test.astype('float32') / 255. - 0.5  # minmax_normalized
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    print(x_train.shape)
    print(x_test.shape)

    # in order to plot in a 2D figure
    encoding_dim = 2

    # this is our input placeholder (, 784)
    input_img = Input(shape=(784,))

    # encoder layers
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(10, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # decoder layers
    decoded = Dense(10, activation='relu')(encoder_output)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='tanh')(decoded)

    # construct the autoencoder model
    autoencoder = Model(input=input_img, output=decoded)

    # construct the encoder model for plotting
    encoder = Model(input=input_img, output=encoder_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    autoencoder.fit(x_train, x_train,
                    epochs=20,
                    batch_size=256,
                    shuffle=True)

    # plotting
    encoded_imgs = encoder.predict(x_test)
    plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
    plt.colorbar()
    plt.show()


def save_load_demo():
    # create some data
    X = np.linspace(-1, 1, 200)
    np.random.shuffle(X)  # randomize the data
    Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))
    X_train, Y_train = X[:160], Y[:160]  # first 160 data points
    X_test, Y_test = X[160:], Y[160:]  # last 40 data points
    model = Sequential()
    model.add(Dense(output_dim=1, input_dim=1))
    model.compile(loss='mse', optimizer='sgd')
    for step in range(301):
        cost = model.train_on_batch(X_train, Y_train)

    # save
    print('test before save: ', model.predict(X_test[0:2]))
    model.save('my_model.h5')  # HDF5 file, you have to pip3 install h5py if don't have it
    del model  # deletes the existing model

    # load
    model = load_model('my_model.h5')
    print('test after load: ', model.predict(X_test[0:2]))
    """
    # save and load weights
    model.save_weights('my_model_weights.h5')
    model.load_weights('my_model_weights.h5')
    
    # save and load fresh network without trained weights
    from keras.models import model_from_json
    json_string = model.to_json()
    model = model_from_json(json_string)
    """


if __name__ == '__main__':
    # regression_demo()
    # classify_demo()
    # cnn_demo()
    # rnn_classifier_demo()
    # rnn_regress_demo()
    auto_encoder_demo()
