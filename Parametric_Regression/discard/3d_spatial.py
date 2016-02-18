#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from data_utils import *
from regression import *
import  scipy.stats as stats

def main():

    X, data_Y = readData("data/0032/ethylene_CO.txt", skipHeader=True,scale=False)
    print(X.shape)
    #compare results of regression methods
    z = mapFeatures(X[:,10:],1)
    thetas = fit_model(z, data_Y)
    print("Method 1 Coefficients: {}".format(thetas))
    print("10-fold performance: {}".format(kfold_validation(z,data_Y,10)))
    print("10-fold performance: {}\n".format(kfold_validation_gaussian(z,data_Y,10,3)))



    # regr = linear_model.LinearRegression()
    # regr.fit(z, data_Y)
    # print("Ready Made method Coefficients: {} Intercept: {}".format(regr.coef_, regr.intercept_))
    # print("Mean error: {}".format(getMeanErro



if __name__ == "__main__":
    main()