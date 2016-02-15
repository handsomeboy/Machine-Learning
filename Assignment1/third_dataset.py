#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from data_utils import *
from regression import *

def third_dataset():

    data_X, data_Y = readData("data/svar-set3.dat", " ")

    #plot original data
    plt.scatter(data_X, data_Y,  color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    for degree in range(1,4):
        #map features
        z=mapFeatures(data_X,degree)
        # Split the data into training/testing sets
        data_X_train, data_Y_train, data_X_test, data_Y_test = splitDataSet(0.9,z,data_Y)

        #calculate thetas method 1
        thetas = fit_model1(data_X_train, data_Y_train)
        print("Method 1 Coefficients: {}\n".format(thetas))


        plt.scatter(data_X_train[:,1], predict(thetas,data_X_train), color='green',
                 linewidth=3)

        plt.scatter(data_X_test[:,1], data_Y_test,  color='black')
        plt.scatter(data_X_test[:,1], predict(thetas,data_X_test), color='blue',
                 linewidth=3)
        plt.xlabel('x')
        plt.ylabel("y")
        plt.show()

#--------------------------------------------------------------------#

    for degree in range(1,50):
        poly = preprocessing.PolynomialFeatures(degree)
        newx = poly.fit_transform(data_X)
        print("degree: {}, 10-fold cross valdiation: {}".format(degree,kfold_validation(newx,data_Y,10)))


    z = mapFeatures(data_X,3)


    for k in range (2,10,1):
        # data_X_train, data_Y_train, data_X_test, data_Y_test = splitDataSet(training_size,z,data_Y)

        # thetas = fit_model1(z, data_Y)
        error = kfold_validation(z,data_Y,k)
        print("k={}, error={}".format(k,error))


if __name__ == "__main__":
    third_dataset()