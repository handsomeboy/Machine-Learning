#!/usr/bin/python

from data_utils import *
import numpy as np
from logistic_regression_kclass import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from metrics import *

#map matrix to a higher polynomial degree
def mapFeatures(x, degree):
    poly = preprocessing.PolynomialFeatures(degree,interaction_only=True)
    return poly.fit_transform(x)

def main():
    #read data
    x, labels = readData("Data/iris.data",",",scale=False)

    #shuffle
    p = np.random.permutation(len(x))
    x = x[p]
    labels = labels[p]

    classes, y = np.unique(labels, return_inverse=True)
    x = mapFeatures(x,1)
    #evaluate using 10 fold cross validation
    print("Kfold Accuracy, fmeasure, confusion matrix: {}".format(kfoldCrossValidation(x,y, 10, maxIterations=900, learning_rate=0.001)))

if __name__ == "__main__":
    main()