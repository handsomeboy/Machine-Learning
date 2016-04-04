import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import plot_utils as pu
from sklearn import preprocessing
import svm as svm

def mapFeatures(x, degree):
    poly = preprocessing.PolynomialFeatures(degree)
    return poly.fit_transform(x)

def func(x):
    return 1 * (np.dot(x, [2,-1])>0)

def func2(x):
    val = np.dot(x, [-1,-2,-1])
    if (abs(val - 0) < 0.5):
        if(random.random() < 0.5):
            return 1 * (not(val>0))
    return 1 * (val>0)

def main():
    X = np.empty([500,2])
    X[:,0] = np.matrix((random.sample(range(-10000, 10000), 500))) / float(1000)
    X[:,1] = np.matrix((random.sample(range(-10000, 10000), 500))) / float(1000)

    #linearly separable
    y = func(X)

    #plot data and decision surface
    ax = plt.gca()
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.scatter(X[:,0], X[:,1], c=(y == 1), cmap=cm_bright)
    plt.xlabel("X1")
    plt.ylabel("X2")
    pu.plot_surface(X,y, X[:, 0], X[:,1], disc_func=func, ax=ax)
    plt.show()

    #train svm
    w,w0 = svm.train(X,y,c=999999999999999)
    print(w,w0)

    #plot result
    predicted_labels = svm.classify_all(X,w,w0)
    print(svm.getAccuracy(y,predicted_labels))
    for i in range(X.shape[0]):
        if(y[i] != predicted_labels[i]):
            print(X[i,:])
    ax = plt.gca()
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.scatter(X[:, 0], X[:, 1], c=(y == 1), cmap=cm_bright)
    plt.xlabel("X1")
    plt.ylabel("X2")
    pu.plot_surface2(X[:,0], X[:,1], w,w0)
    plt.show()


    #not separable
    # y = np.empty([100,1])
    # for i in range(X.shape[0]):
    #     y[i] = func2(X[i,:])
    #
    # # plot data and decision surface
    # ax = plt.gca()
    # cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # ax.scatter(X[:,1], X[:,2], c=(y == 1), cmap=cm_bright)
    # plt.xlabel("X1")
    # plt.ylabel("X2")
    # pu.plot_surface(X,y, X[:, 1], X[:, 2], disc_func=func, ax=ax)
    # plt.show()

if __name__ == "__main__":
    main()