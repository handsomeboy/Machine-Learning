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

def gaussian_kernel():

    data_X, data_Y = readData("data/mvar-set4.dat", " ")

    #compare results of regression methods
    z = mapFeatures(data_X,1)
    z = z[1:1500,:]
    data_Y = data_Y[1:1500]
    thetas = fit_model1(z, data_Y)
    print("Method 1 Coefficients: {}\n".format(thetas))
    print("10-fold performance: {}".format(kfold_validation(z,data_Y,10)))

    regr = linear_model.LinearRegression()
    regr.fit(z, data_Y)
    print("Ready Made method Coefficients: {} Intercept: {}\n".format(regr.coef_, regr.intercept_))

    #solve dual
    thetas = solveDual(getGaussianGramMatrix(z,4), z, data_Y)
    print("Dual Problem - Gaussian Kernel Coefficients: {}\n".format(thetas))
    print("10-fold performance: {}".format(kfold_validation_gaussian(z,data_Y,10)))

    #predict with dual
    example = z[1]
    print("prediction: {}".format(predict(thetas, z[1])))
    print(data_Y[1])

if __name__ == "__main__":
    gaussian_kernel()