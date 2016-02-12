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
    #plt.scatter(data_X, data_Y,  color='black')
    #plt.show()

    #add ones to first column in X
    poly = preprocessing.PolynomialFeatures(1)
    z = poly.fit_transform(data_X)

    # Split the data into training/testing sets
    data_X_train, data_Y_train, data_X_test, data_Y_test = splitDataSet(0.9,z,data_Y)

    #calculate thetas method 1
    thetas = fit_model1(data_X_train, data_Y_train)
    print("Method 1 Coefficients: {}\n".format(thetas))

    #calculate thetas method 2
    thetas = fit_model2(data_X_train, data_Y_train)
    print("Method 2 Coefficients: {}\n".format(thetas))

    thetas = gradient_descent(data_X_train,data_Y_train,learning_weight=0.001)
    print("Itearative Method Coefficients: {}\n".format(thetas))

    #calculate thetas method 3
    regr = linear_model.LinearRegression()
    X_train, Y_train, X_test, Y_test = splitDataSet(0.9,data_X,data_Y)
    regr.fit(X_train, Y_train)
    print("Ready Made method Coefficients: {} Intercept: {}\n".format(regr.coef_, regr.intercept_))

    print("Training Mean Squared Error: {}\n".format(getMeanError(thetas,data_X_train,data_Y_train)))
    print("Ready Made Training Mean Squared Error: {}\n".format(np.mean((regr.predict(data_X[:-20]) - data_Y_train) ** 2)))

    print("Testing Mean Squared Error: {}\n".format(getMeanError(thetas,data_X_test,data_Y_test)))
    print("Ready Made Testing Mean Squared Error: {}\n".format(np.mean((regr.predict(data_X[-20:]) - data_Y_test) ** 2)))

    plt.scatter(data_X_test[:,1], data_Y_test,  color='black')
    plt.plot(data_X_test[:,1], predict(thetas,data_X_test), color='blue',
             linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

if __name__ == "__main__":
    main()