#!/usr/bin/python

from data_utils import *
import numpy as np
from multilayer import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from metrics import *
import learnmultilayer
def main():
    #read data
    x, labels = readData("Data/iris.data",",",scale=False)

    #shuffle
    p = np.random.permutation(len(x))
    x = x[p]
    labels = labels[p]
    classes, y = np.unique(labels, return_inverse=True)

    # #plot h vs f-measure
    accuracies = list()
    fmeasures = list()
    for h in range(1,8):
        validation = kfoldCrossValidation(x,labels, 10, h=h, maxIterations=500,learning_rate=0.003, learning_rate_v=0.003)
        fmeasures.append(validation[4])
    plt.bar(range(1,8),fmeasures)
    plt.xlabel("h")
    plt.ylabel("F-Measure")
    plt.show()

    #plot effect of learning_rate
    handles = np.empty([0,1])
    k = len(labels)
    h=4
    n = x.shape[1]
    iv = np.random.rand(k,h)/10
    iw = np.random.rand(h,n)/10
    for lr in [0.001,0.002, 0.003, 0.004, 0.005]:
        #perform gradient descent
        v,w, iterations, errors = learnmultilayer.gradient_descent(x,y, classes, h=4, maxIterations=100,learning_rate=lr, learning_rate_v=lr,iv=iv,iw=iw)
        # print("Itearative Method Coefficients: {},Iterations:{}".format(thetas,iterations))
        handle = plt.plot(errors[1:100,0],errors[1:100,1], linewidth = 4, label="Learning Rate = {:.3f}".format(lr))
        handles = np.append(handles,handle)
    plt.legend(handles=[handles[0],handles[1],handles[2],handles[3],handles[4]])
    plt.ticklabel_format(useOffset=False)
    plt.xlabel("Iteration")
    plt.ylabel("- log likelihood")
    plt.show()


if __name__ == "__main__":
    main()