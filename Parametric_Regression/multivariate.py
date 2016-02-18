#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from data_utils import *
from regression import *

def main():

    data_X, data_Y = readData("data/mvar-set1.dat", " ")

    #compare results of regression methods
    z = mapFeatures(data_X,1)
    thetas = fit_model(z, data_Y)
    print("Coefficients: {}\n".format(thetas))

    regr = linear_model.LinearRegression()
    regr.fit(z, data_Y)
    print("Ready Made method Coefficients: {} Intercept: {}\n".format(regr.coef_, regr.intercept_))

    #evaluate with 10 fold cross validation
    print("10-fold cross validation mean squared error is {}\n".format(kfold_validation(z,data_Y,10, fit_model)))

    thetas,iterations,errors = gradient_descent(z,data_Y, learning_weight=0.00001)
    print("Iterative Method Coefficients: {}".format(thetas))
    print("10-fold cross validation mean squared error is {}\n".format(kfold_validation_gradient_descent(z,data_Y,10, lw=0.00001)))

if __name__ == "__main__":
    main()