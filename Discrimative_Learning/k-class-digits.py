#!/usr/bin/python

from data_utils import *
import numpy as np
from logistic_regression_kclass import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from get_digits_data import *
from metrics import *

#map matrix to a higher polynomial degree
def mapFeatures(x, degree):
    poly = preprocessing.PolynomialFeatures(degree,interaction_only=True)
    return poly.fit_transform(x)

def main():
    #read data
    X,y = getDigitsData()
    X_train, X_test = X[:70000], X[70000:]
    y_train, y_test = y[:70000], y[70000:]

    #shuffle
    p = np.random.permutation(len(X_train))
    X_train = X_train[p]
    y_train = y_train[p]

    X_train = mapFeatures(X_train,1)

    #evaluate using 10 fold cross validation
    all_metrics, all_n = kfoldCrossValidation(X_train,y_train, 10, maxIterations=20, learning_rate=0.000005)
    #save confusion matrix in csv file
    np.savetxt("confusion.csv", all_n.reshape(10,10), delimiter=",", fmt='%7.2f')
    print("Kfold Accuracy, fmeasure, confussion matrix: {}".format(all_metrics))


if __name__ == "__main__":
    main()