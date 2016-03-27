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

    ind = [ k for k in range(len(y)) if y[k] in [0,1,2] ]

    X = X[ind]
    y = y[ind]

    #shuffle
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]

    X = X[:500]
    y = y[:500]


    # classes, y = np.unique(y, return_inverse=True)
    # v,w,iterations, all_likelihoods = train(X,y, h=784, maxIterations=500,learning_rate=0.00001)
    # print(iterations)
    # print("Training accuracy: {}".format(getAccuracy(y,classify_all(X,X,y,maxIterations=50,learning_rate=0.000001,v=v,w=w,h=500),1)))

    # classes, y = np.unique(y, return_inverse=True)
    # handles = np.empty([0,1])
    # h=784
    # n = X.shape[1]
    # k = len(y)
    # iv = np.random.rand(k,h)/10
    # iw = np.random.rand(h,n)/10
    # for lr in [0.05,0.1, 0.15, 0.2]:
    #     #perform gradient descent
    #     v,w, iterations, errors = learnmultilayer.gradient_descent(X,y, classes, h=h, maxIterations=50,learning_rate=lr,iv=iv,iw=iw)
    #     # print("Itearative Method Coefficients: {},Iterations:{}".format(thetas,iterations))
    #     handle = plt.plot(errors[1:50,0],errors[1:50,1], linewidth = 4, label="Learning Rate = {:.3f}".format(lr))
    #     handles = np.append(handles,handle)
    # plt.legend(handles=[handles[0],handles[1],handles[2],handles[3]])
    # plt.ticklabel_format(useOffset=False)
    # plt.xlabel("Iteration")
    # plt.ylabel("- log likelihood")
    # plt.show()

    # classes, y = np.unique(y, return_inverse=True)
    # handles = np.empty([0, 1])
    # lr=0.25
    # n = X.shape[1]
    # k = len(y)
    # the_errors = list()
    # for h in [500, 600, 700, 800, 900]:
    #     iv = np.random.rand(k, h) / 10
    #     iw = np.random.rand(h, n) / 10
    #     # perform gradient descent
    #     v, w, iterations, errors = learnmultilayer.gradient_descent(X, y, classes, h=h, maxIterations=100,
    #                                                                 learning_rate=lr, iv=iv, iw=iw, threshold=0.1)
    #     # print("Itearative Method Coefficients: {},Iterations:{}".format(thetas,iterations))
    #     the_errors.append(errors[iterations-1])
    # print(the_errors)

    accuracies = list()
    fmeasures = list()
    for h in [100,200]:
        validation = kfoldCrossValidation(X, y, 5, 3, h=h, maxIterations=100,learning_rate=1.5)
        fmeasures.append(validation[4])
    # plt.plot(range(500,1000, 100),fmeasures)
    # plt.xlabel("h")
    # plt.ylabel("F-Measure")
    # plt.show()
    print(fmeasures)

if __name__ == "__main__":
    main()