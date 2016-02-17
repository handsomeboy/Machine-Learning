#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from data_utils import *
from regression import *

def mulitple_features_dataset():

    data_X, data_Y = readData("data/mvar-set1.dat", " ")

    #compare results of regression methods
    z = mapFeatures(data_X,1)
    thetas = fit_model(z, data_Y)
    print("Method 1 Coefficients: {}\n".format(thetas))
    regr = linear_model.LinearRegression()
    regr.fit(z, data_Y)
    print("Ready Made method Coefficients: {} Intercept: {}\n".format(regr.coef_, regr.intercept_))

    z = mapFeatures(data_X,1)

    thetas = gradient_descent(z,data_Y, learning_weight=0.00001)
    print("Itearative Method Coefficients: {}\n".format(thetas))
    #normalize data?

    #compute learning weight

if __name__ == "__main__":
    mulitple_features_dataset()