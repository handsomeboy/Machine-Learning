#!/usr/bin/python

import numpy as np
from multilayer import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from metrics import *
from get_digits_data import *

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


    classes, y = np.unique(y, return_inverse=True)
    v,w,iterations, all_likelihoods = train(X,y, h=500, maxIterations=500,learning_rate=0.00002)
    print(iterations)
    print("Training accuracy: {}".format(getAccuracy(y,classify_all(X,X,y,maxIterations=50,learning_rate=0.000001,v=v,w=w,h=500),1)))


    # classes, y = np.unique(y, return_inverse=True)
    # v,w = train(data,y)
    # print("Training accuracy: {}".format(getAccuracy(labels,classify_all(x,x,labels),1)))

if __name__ == "__main__":
    main()