# coding=utf-8
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import pickle
from sklearn.externals import joblib


def knn_classifier():
    """

    """
    iris = datasets.load_iris()
    iris_x = iris.data
    iris_y = iris.target
    # print(iris_x[:2, :])
    # print(iris_y)
    x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.3)
    # print(y_train)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    print(knn.predict(x_test))
    print(y_test)
    print(knn.score(x_test, y_test))

    # 保存模型
    with open('save/clf.pickle', 'wb') as f:
        pickle.dump(knn, f)

    with open('save/knn.pickle', 'rb') as f:
        knn2 = pickle.load(f)
        print(knn2.predict(x_test))

    #     joblib
    joblib.dump(knn, 'save/knn2.pkl')
    knn3 = joblib.load('save/knn2.pkl')
    print(knn3.predict(x_test))


def linear_regression():
    """

    """
    loaded_data = datasets.load_boston()
    data_x = loaded_data.data
    data_y = loaded_data.target

    model = LinearRegression()
    model.fit(data_x, data_y)
    # 斜率
    print(model.coef_)
    # 截距
    print(model.intercept_)
    print(model.get_params())
    print(model.predict(data_x[:4, :]))
    print(data_x[:4, :])

    x, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=1)
    plt.scatter(x, y)
    plt.show()


def normalize_data():
    """

    """
    a = np.array([[10, 2.7, 3.6], [-100, 5, -2], [120, 20, 40]], dtype=np.float64)
    print(a)
    print(preprocessing.scale(a))
    x, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2, random_state=22,
                               n_clusters_per_class=1, scale=100)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()

    x = preprocessing.scale(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf = SVC()
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))


def cross_validation():
    """

    """
    iris = datasets.load_iris()
    iris_x = iris.data
    iris_y = iris.target

    k_range = range(1, 31)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, iris_x, iris_y, cv=5, scoring='accuracy')  # for classification
        loss = cross_val_score(knn, iris_x, iris_y, cv=10, scoring='neg_mean_squared_error')  # for regression
        k_scores.append(scores.mean())

    plt.plot(k_range, k_scores)
    plt.xlabel('value of k for knn')
    plt.ylabel('cross validated accuracy')
    plt.show()


def over_fitting():
    """

    """
    digits = datasets.load_digits()
    x = digits.data
    y = digits.target
    train_sizes, train_loss, test_loss = learning_curve(SVC(gamma=0.001), x, y, cv=10, scoring='neg_mean_squared_error',
                                                        train_sizes=[0.1, 0.25, 0.75, 1])
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)

    plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label='Training')
    plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label='Cross-validation')

    plt.xlabel('Training examples')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()


def validate_curve():
    """

    """
    digits = datasets.load_digits()
    x = digits.data
    y = digits.target
    param_range = np.logspace(-6, -2.3, 5)
    train_loss, test_loss = validation_curve(SVC(), x, y, param_name='gamma', param_range=param_range, cv=10,
                                             scoring='neg_mean_squared_error')
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)

    plt.plot(param_range, train_loss_mean, 'o-', color='r', label='Training')
    plt.plot(param_range, test_loss_mean, 'o-', color='g', label='Cross-validation')

    plt.xlabel('gamma')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    # knn_classifier()
    # linear_regression()
    # normalize_data()
    # over_fitting()
    validate_curve()
