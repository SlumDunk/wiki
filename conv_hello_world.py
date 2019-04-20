import numpy as np
import networkx as nx
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras.layers.pooling import AveragePooling1D, MaxPooling1D
from keras.models import Model
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import scipy.sparse as sp
import pickle as pkl

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

with open('data/wiki_edit_x_y.pkl', 'rb') as f:
    A = pkl.load(f)
    y = pkl.load(f)
G = nx.from_scipy_sparse_matrix(A)


# [rows, cols] = vecs.shape
# print(rows, cols)
# for i in range(rows):
#     for j in range(cols):
#         virtual = np.imag(vecs[i, j])
#         if virtual > 0:
#             print(vecs[i, j])


def conv(X_train, X_test, y_train, y_test, input_shape):
    """

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param input_shape:
    :return:
    """
    input_img = Input(shape=input_shape)
    conv_blocks = []
    for sz, pl in zip([1, 2, 3], [3, 2, 1]):
        conv1 = Conv1D(filters=150, kernel_size=sz, activation='relu')(input_img)
        pool = AveragePooling1D(pool_size=pl)(conv1)
        flatten = Flatten()(pool)
        conv_blocks.append(flatten)
    out = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    out = Dropout(0.5)(out)
    out = Dense(100, activation="relu")(out)
    model_output = Dense(1, activation="sigmoid")(out)
    model = Model(input_img, model_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=100, validation_split=0.05, verbose=0)
    scores = model.evaluate(X_test, y_test, verbose=2)
    predict_y = model.predict(X_test)
    predict_y = (predict_y > 0.5).astype('int32')
    f1 = f1_score(y_test, predict_y, pos_label=0)
    return scores[1] * 100, f1


pass


def construct_input_for_CNN(X):
    """
    construct input matrix for cnn model
    :param X:
    :return:
    """
    pos_n = []
    neg_n = []
    for nid in G.nodes():
        pos_vec = []
        neg_vec = []
        vec = []
        for nid2 in G[nid]:
            vec.append(X[nid2])
            if G[nid][nid2]['weight'] == 1:
                pos_vec.append(X[nid2])
            else:
                neg_vec.append(X[nid2])
        avg_vec = np.mean(np.asarray(vec), axis=0).reshape(1, X.shape[1])
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

    input_img = np.concatenate((X[:, np.newaxis], pos_n[:, np.newaxis], neg_n[:, np.newaxis]), axis=1)
    print(input_img.shape)
    return input_img


pass


def test_range():
    _, vecs = sp.linalg.eigs(A.asfptype(), k=30, which='LR')
    X = normalize(vecs.real, norm='l2')
    input_img = construct_input_for_CNN(X)
    for r in np.arange(0.5, 0.8, 0.1):
        accu = []
        f1 = []
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(input_img, y, test_size=r)
            a, f = conv(X_train, X_test, y_train, y_test, input_shape=(3, 30))
            accu.append(a)
            f1.append(f)
        print(np.mean(accu))
        print(np.mean(f1))


def dimension_range():
    for r in np.arange(10, 60, 10):
        print(r)
        accu = []
        f1 = []
        _, vecs = sp.linalg.eigs(A.asfptype(), k=r, which='LR')
        X = normalize(vecs.real, norm='l2')
        X = X[:, np.newaxis, :]
        for i in range(5):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
            a, f = conv(X_train, X_test, y_train, y_test, input_shape=(1, r))
            accu.append(a)
            f1.append(f)
        print(np.mean(accu))
        print(np.mean(f1))


pass

if __name__ == '__main__':
    # test_range()
    dimension_range()