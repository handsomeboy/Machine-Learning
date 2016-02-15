#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from data_utils import *
from regression import *

def housing():

    X, data_Y = readData("data/housing/housing.data", scale=False)

    # #plot RM attribute
    # data_X = X[:,np.newaxis,5]
    # plt.scatter(data_X, data_Y,  color='black')
    # plt.xlabel('Avg number of rooms per dwelling')
    # plt.ylabel("Median value of owner-occupied homes in $1000's")
    # plt.show()
    #
    # #plot regression model for different degrees
    # for degree in range(1,4):
    #     #map features
    #     z=mapFeatures(data_X,degree)
    #     # Split the data into training/testing sets
    #     data_X_train, data_Y_train, data_X_test, data_Y_test = splitDataSet(0.9,z,data_Y)
    #
    #     #calculate thetas method 1
    #     thetas = fit_model1(data_X_train, data_Y_train)
    #     print("Method 1 Coefficients: {}\n".format(thetas))
    #
    #
    #     plt.scatter(data_X_train[:,1], predict(thetas,data_X_train), color='green',
    #              linewidth=3)
    #
    #     plt.scatter(data_X_test[:,1], data_Y_test,  color='black')
    #     plt.scatter(data_X_test[:,1], predict(thetas,data_X_test), color='blue',
    #              linewidth=3)
    #     plt.xlabel('Avg number of rooms per dwelling')
    #     plt.ylabel("Value of owner-occupied homes in $1000's")
    #     plt.show()
    #
    # values = np.empty([10,2])
    # for degree in range(1,10):
    #     #map features
    #     newx = mapFeatures(data_X,degree)
    #     performance = kfold_validation(newx,data_Y,10, fit_model1)
    #     values[degree-1] = [degree,performance]
    #     print("degree: {}, 10-fold cross valdiation: {}".format(degree,kfold_validation(newx,data_Y,10, fit_model1)))
    # plt.plot(values[:,np.newaxis,0],values[:,np.newaxis,1])
    # plt.show()



if __name__ == "__main__":
    housing()