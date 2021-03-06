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
    if (np.dot(x, [-2,-1]) > 0):
        return 1
    else :
        return -1


def main():
    m=100
    X = np.empty([m,2])
    X[:,0] = np.matrix((random.sample(range(-10000, 10000), m))) / float(1000)
    X[:,1] = np.matrix((random.sample(range(-10000, 10000), m))) / float(1000)

    # preprocessing.scale(X)

    #linearly separable
    y = np.empty([m,1])
    for i in range(m):
        y[i] = func(X[i,])

    #plot data and decision surface
    ax = pu.plot_data(X,y)
    pu.plot_surface(X,y, X[:, 0], X[:,1], disc_func=func, ax=ax)
    plt.show()

    #train svm

    w,w0, support_vectors_idx = svm.train(X,y,c=999999999999999, eps=10, type='gaussian')
    # w, w0, support_vectors_idx = svm.train(X, y, c=999999999999999, eps=10, type='polynomial')
    #plot result
    predicted_labels = svm.classify_all(X,w,w0)
    print("Accuracy: {}".format(svm.getAccuracy(y,predicted_labels)))


    ax = pu.plot_data(X,y, support_vectors_idx)
    pu.plot_surfaceSVM(X[:,0], X[:,1], w,w0, ax=ax)
    plt.show()



if __name__ == "__main__":
    main()