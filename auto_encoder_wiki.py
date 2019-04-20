import numpy as np
import networkx as nx
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import normalize
import pickle as pkl
import scipy.sparse as sp

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

with open('./data/wiki_edit_x_y.pkl', 'rb') as f:
    A = pkl.load(f)
    y = pkl.load(f)
G = nx.from_scipy_sparse_matrix(A)


def dae(X_train, X_test, y_train, y_test, input_dim):
    """

    :param X_train: 训练特征数据
    :param X_test: 测试特征数据
    :param y_train: 训练标签数据
    :param y_test: 测试标签数据
    :param input_dim: 输入维度
    :return:
    """
    input_img = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    #     encoded = Dense(32, activation='tanh')(encoded)

    decoded = Dense(128, activation='relu')(encoded)
    #     decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='tanh')(decoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')  # mse
    autoencoder.fit(X_train, X_train,
                    epochs=20,
                    batch_size=256,
                    verbose=0,
                    shuffle=True,
                    validation_data=(X_test, X_test))

    output = Dense(1, activation='sigmoid', activity_regularizer=regularizers.l2(10e-5))(encoded)
    predict = Model(input_img, output)
    predict.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    predict.fit(X_train, y_train, epochs=20, batch_size=100, validation_split=0.05, verbose=0)
    scores = predict.evaluate(X_test, y_test, verbose=0)
    predict_y = predict.predict(X_test)
    predict_y = (predict_y > 0.5).astype('int32')
    #     print("%s: %.2f%%" % (predict.metrics_names[1], scores[1]*100))
    #     print(classification_report(y_test, predict_y, digits=4))
    f1 = f1_score(y_test, predict_y, pos_label=0)
    return scores[1] * 100, f1


def construct_dae_input(X):
    """
    construct matrix for input of neural network
    :param X:
    :return:
    """
    pos_n = []
    neg_n = []
    for nid in G.nodes():
        pos_vec = []
        neg_vec = []
        for nid2 in G[nid]:
            if G[nid][nid2]['weight'] == 1:
                pos_vec.append(X[nid2])
            else:
                neg_vec.append(X[nid2])
        if len(pos_vec) == 0:
            pos_vec = np.zeros((1, X.shape[1]))
        if len(neg_vec) == 0:
            neg_vec = np.zeros((1, X.shape[1]))
        avg_p_v = np.mean(np.asarray(pos_vec), axis=0)
        avg_n_v = np.mean(np.asarray(neg_vec), axis=0)
        pos_n.append(avg_p_v)
        neg_n.append(avg_n_v)
    pos_n = np.asarray(pos_n)
    neg_n = np.asarray(neg_n)
    assert pos_n.shape == X.shape
    assert neg_n.shape == X.shape

    input_img = np.hstack((X, pos_n))
    input_img = np.hstack((input_img, neg_n))
    return input_img


def dimension_range():
    """
    test the performance of the model in different number of  principle dimensions
    """
    # 不同维度
    for v in np.arange(10, 60, 10):
        _, vecs = sp.linalg.eigs(A.asfptype(), k=v, which='LR')
        X = normalize(vecs.real, norm='l2')
        input_img = construct_dae_input(X)
        accu = []
        f1 = []
        for i in range(5):
            X_train, X_test, y_train, y_test = train_test_split(input_img, y, test_size=0.8)
            a, f = dae(X_train, X_test, y_train, y_test, 3 * v)
            accu.append(a)
            f1.append(f)
        print(np.mean(accu))
        print(np.mean(f1))


def test_range():
    for r in np.arange(0.5, 0.8, 0.1):
        accu = []
        f1 = []
        k = 30;
        _, vecs = sp.linalg.eigs(A.asfptype(), k=k, which='LR')
        X = normalize(vecs.real, norm='l2')
        input_img = construct_dae_input(X)
        for i in range(10):
            #         print(i)
            X_train, X_test, y_train, y_test = train_test_split(input_img, y, test_size=r)
            a, f = dae(X_train, X_test, y_train, y_test, 3 * k)
            accu.append(a)
            f1.append(f)
        print(np.mean(accu))
        print(np.mean(f1))


if __name__ == '__main__':
    # dimension_range()
    # test_range()
    print('helloworld')