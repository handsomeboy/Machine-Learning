#!/usr/bin/python

from data_utils import *
import numpy as np
from logistic_regression_kclass import *
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

    #use 0.001, and 900 iterations
    # thetas,all_likelihoods = train(x,labels,0.01,900,0.0005)
    # print("Training accuracy: {}".format(getAccuracy(labels,classify_all(x,x,labels, thetas),1)))
    print("Kfold Accuracy, recall, precission,tp,tn,fp,fn: {}".format(kfoldCrossValidation3Classes(x,labels, 10, 3)))

if __name__ == "__main__":
    main()