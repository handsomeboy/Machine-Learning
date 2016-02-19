#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from sklearn.cross_validation import _PartitionIterator

from data_utils import *
from regression import *
from scipy.spatial.distance import pdist, squareform
import scipy
import time
def gaussian_kernel():

    data_X, data_Y = readData("data/CCPP/Folds.csv", delim=",", skipHeader=True, scale=False)
    #compare results of regression methods
    z = mapFeatures(data_X,1)
    z = z[:,:]
    data_Y = data_Y[:]
    start = time.time()
    thetas = fit_model(z, data_Y)
    end = time.time()
    print("Method 1 Coefficients: {}".format(thetas))
    print("Time for explicit solution: {}".format(end - start))
    # print("10-fold performance: {}\n".format(kfold_validation(z,data_Y,10)))


    regr = linear_model.LinearRegression()
    regr.fit(z, data_Y)
    print("Ready Made method Coefficients: {} Intercept: {}\n".format(regr.coef_, regr.intercept_))

    #iterative solution
    start = time.time()
    thetas, iterations, errors = gradient_descent(z,data_Y, threshold=0.00001, learning_weight=0.00000001)
    end = time.time()
    print("Itearative Method Coefficients: {},Iterations:{}".format(thetas,iterations))
    print("Time for iterative solution: {}\n".format(end - start))

    #solve dual
    start = time.time()
    thetas = solveDual(getGaussianGramMatrix(z,1), z, data_Y)
    end = time.time()
    print("Dual Problem - Gaussian Kernel Coefficients: {}".format(thetas))
    print("Time for explicit solution: {}".format(end - start))
    # print("10-fold performance: {}\n".format(kfold_validation_gaussian(z,data_Y,10,1)))

    #predict with dual
    example = z[1]
    print("prediction: {}".format(predict(thetas, z[1])))
    print(data_Y[1])

if __name__ == "__main__":
    gaussian_kernel()