#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from data_utils import *
from regression import *

def main():

    data_X, data_Y = readData("data/svar-set1.dat", " ")

    #plot original data
    plt.scatter(data_X, data_Y,  color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    #map features to polynomial degree (if degree=1 only add a 1's column)
    z=mapFeatures(data_X,1)

    # Split the data into training/testing sets
    data_X_train, data_Y_train, data_X_test, data_Y_test = splitDataSet(0.9,z,data_Y)

    #calculate thetas
    thetas = fit_model(data_X_train, data_Y_train)
    print("Coefficients: {}\n".format(thetas))

    #compare to ready made method
    regr = linear_model.LinearRegression()
    X_train, Y_train, X_test, Y_test = splitDataSet(0.9,data_X,data_Y)
    regr.fit(X_train, Y_train)
    print("Ready Made method Coefficients: {} Intercept: {}\n".format(regr.coef_, regr.intercept_))

    #get training and testing error
    print("Training Mean Squared Error: {}".format(getMeanError(thetas,data_X_train,data_Y_train)))
    print("Ready Made Training Mean Squared Error: {}\n".format(np.mean((regr.predict(data_X[:-20]) - data_Y_train) ** 2)))

    print("Testing Mean Squared Error: {}".format(getMeanError(thetas,data_X_test,data_Y_test)))
    print("Ready Made Testing Mean Squared Error: {}\n".format(np.mean((regr.predict(data_X[-20:]) - data_Y_test) ** 2)))

    #evaluate with 10 fold cross validation
    print("10-fold cross validation mean squared error is {}\n".format(kfold_validation(z,data_Y,10, fit_model)))

    #Use Gradient Descent algorithm
    thetas,iterations,errors = gradient_descent(data_X_train,data_Y_train,learning_weight=0.001)
    print("Itearative Method Coefficients: {}".format(thetas))
    print("Iterative Method Testing Mean Squared Error: {}".format(getMeanError(thetas,data_X_test,data_Y_test)))
    print("10-fold cross validation mean squared error is {}\n".format(kfold_validation_gradient_descent(z,data_Y,10, lw=0.001)))

if __name__ == "__main__":
    main()