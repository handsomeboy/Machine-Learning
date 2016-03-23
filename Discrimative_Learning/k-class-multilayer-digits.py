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

    classes, y = np.unique(y, return_inverse=True)
    v,w = train(X,y, maxIterations=10,learning_rate=0.0000001)
    print("Training accuracy: {}".format(getAccuracy(y,classify_all(X,X,y,v,w),1)))


    # classes, y = np.unique(y, return_inverse=True)
    # v,w = train(data,y)
    # print("Training accuracy: {}".format(getAccuracy(labels,classify_all(x,x,labels),1)))

if __name__ == "__main__":
    main()