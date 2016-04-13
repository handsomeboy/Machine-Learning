import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import plot_utils as pu
from sklearn import preprocessing
import svm as svm


def trans_val(val):
    if(val > 0):
        return 1
    else:
        return -1

def func(x):
    if (np.dot(x, [-2,-1]) > 0) :
        return 1
    else :
        return -1

def func2(x):
    val = np.dot(x, [-2,-1])
    if (abs(val - 0) < 5):
        if(random.random() < 0.5):
            return trans_val(-1*val)
    return trans_val(val)

def main():
    m=350
    random.seed(2)
    X = np.empty([m,2])
    X[:,0] = np.matrix((random.sample(range(-10000, 10000), m))) / float(1000)
    X[:,1] = np.matrix((random.sample(range(-10000, 10000), m))) / float(1000)

    #not separable
    y = np.empty([m,1])
    for i in range(X.shape[0]):
        y[i] = func2(X[i,:])


    #plot data and decision surface
    ax = pu.plot_data(X,y)
    pu.plot_surface(X,y, X[:, 0], X[:,1], disc_func=func, ax=ax)
    plt.show()

    #train svm
    #change c to hard/soft margins
    w,w0, support_vectors_idx = svm.train(X,y,c=99999,eps=0.1)

    #plot result
    predicted_labels = svm.classify_all(X,w,w0)
    print("Accuracy: {}".format(svm.getAccuracy(y,predicted_labels)))


    ax = pu.plot_data(X,y, support_vectors_idx)
    pu.plot_surfaceSVM(X[:,0], X[:,1], w,w0, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()