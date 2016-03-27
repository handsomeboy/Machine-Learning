#!/usr/bin/python

from data_utils import *
import numpy as np
from multilayer import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from metrics import *

def main():
    #read data
    x, labels = readData("Data/iris.data",",",scale=False)

    #shuffle
    p = np.random.permutation(len(x))
    x = x[p]
    labels = labels[p]

    classes, y = np.unique(labels, return_inverse=True)
    # accuracies = list()
    # fmeasures = list()
    # for h in range(1,8):
    #     #v,w = train(x,y, h=h, maxIterations=800,learning_rate=0.0005)
    #     #accuracy = getAccuracy(labels,classify_all(x,x,labels,v,w, h=h),1)
    #     validation = kfoldCrossValidation(x,labels, 10, 3, h=h, maxIterations=2000,learning_rate=0.0005)
    #     #accuracies.append(accuracy)
    #     fmeasures.append(validation[4])
    #     #print("Training accuracy: {}".format(accuracy))
    # plt.bar(range(1,8),fmeasures)
    # plt.xlabel("h")
    # plt.ylabel("F-Measure")
    # plt.show()
    # print(fmeasures)

    handles = np.empty([0,1])
    k = len(labels)
    h=4
    n = x.shape[1]
    iv = np.random.rand(k,h)/10
    iw = np.random.rand(h,n)/10
    for lr in [0.001,0.002, 0.003, 0.004, 0.005]:
        #perform gradient descent
        v,w, iterations, errors = gradient_descent(x,y, classes, h=4, maxIterations=100,learning_rate=lr,iv=iv,iw=iw)
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