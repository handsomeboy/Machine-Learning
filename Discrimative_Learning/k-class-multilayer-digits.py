#!/usr/bin/python

import numpy as np
from multilayer import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from metrics import *
from get_digits_data import *
import learnmultilayer

def main():
    #read data
    X,y = getDigitsData()

    #get only first 3 classes
    ind = [ k for k in range(len(y)) if y[k] in [0,1,2] ]

    X = X[ind]
    y = y[ind]

    #shuffle
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]

    #use first 500 examples
    X = X[:500]
    y = y[:500]

    # perfrom 10 fold cross validation
    validation = kfoldCrossValidation(X, y, 5, h=784, maxIterations=100, learning_rate=0.005, learning_rate_v=0.00001)

    #plot effect of learning rate
    classes, y = np.unique(y, return_inverse=True)
    handles = np.empty([0,1])
    h=784
    n = X.shape[1]
    k = len(y)
    iv = np.random.rand(k,h)/10
    iw = np.random.rand(h,n)/10
    for lr in [0.05,0.1, 0.15, 0.2]:
        #perform gradient descent
        v,w, iterations, errors = learnmultilayer.gradient_descent(X,y, classes, h=h, maxIterations=50,learning_rate=lr, learning_rate_v=0.00001,iv=iv,iw=iw)
        handle = plt.plot(errors[1:50,0],errors[1:50,1], linewidth = 4, label="Learning Rate = {:.3f}".format(lr))
        handles = np.append(handles,handle)
    plt.legend(handles=[handles[0],handles[1],handles[2],handles[3]])
    plt.ticklabel_format(useOffset=False)
    plt.xlabel("Iteration")
    plt.ylabel("- log likelihood")
    plt.show()




if __name__ == "__main__":
    main()