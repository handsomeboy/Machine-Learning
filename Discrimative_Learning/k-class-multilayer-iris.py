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
    x, labels = readData("C:/Users/marro/Repo/CS584/Generative_Learning/Data/iris.data",",",scale=False)

    classes, y = np.unique(labels, return_inverse=True)
    v,w = train(x,y, maxIterations=800,learning_rate=0.0005)
    print("Training accuracy: {}".format(getAccuracy(labels,classify_all(x,x,labels,v,w),1)))


    # classes, y = np.unique(y, return_inverse=True)
    # v,w = train(data,y)
    # print("Training accuracy: {}".format(getAccuracy(labels,classify_all(x,x,labels),1)))

if __name__ == "__main__":
    main()