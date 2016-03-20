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
    X_train, X_test = X[:40000], X[40000:]
    y_train, y_test = y[:40000], y[40000:]

    classes, y = np.unique(y_train, return_inverse=True)
    v,w = train(X_train,y_train, maxIterations=10,learning_rate=0.0000001)
    print("Training accuracy: {}".format(getAccuracy(y_train,classify_all(X_train,X_train,y_train,v,w),1)))


    # classes, y = np.unique(y, return_inverse=True)
    # v,w = train(data,y)
    # print("Training accuracy: {}".format(getAccuracy(labels,classify_all(x,x,labels),1)))

if __name__ == "__main__":
    main()