#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from data_utils import *
from regression import *

def main():

    data_X, data_Y = readData("data/svar-set1.dat", " ")

    #plot original data
    #plt.scatter(data_X, data_Y,  color='black')
    #plt.show()

    #add ones to first column in X
    z = createZMatrix(data_X)

    # Split the data into training/testing sets
    data_X_train, data_Y_train, data_X_test, data_Y_test = splitDataSet(0.9,z,data_Y)

    #calculate thetas method 1
    thetas = fit_model1(data_X_train, data_Y_train)
    print("Method 1 Coefficients: {}\n".format(thetas))

    #calculate thetas method 2
    thetas = fit_model2(data_X_train, data_Y_train)
    print("Method 2 Coefficients: {}\n".format(thetas))

    #calculate thetas method 3
    regr = linear_model.LinearRegression()
    regr.fit(data_X[:-20], data_Y_train)
    print("Ready Made method Coefficients: {} Intercept: {}\n".format(regr.coef_, regr.intercept_))

    print("Training Mean Squared Error: {}\n".format(getMeanError(thetas,data_X_train,data_Y_train)))
    print("Ready Made Training Mean Squared Error: {}\n".format(np.mean((regr.predict(data_X[:-20]) - data_Y_train) ** 2)))

    print("Testing Mean Squared Error: {}\n".format(getMeanError(thetas,data_X_test,data_Y_test)))
    print("Ready Made Testing Mean Squared Error: {}\n".format(np.mean((regr.predict(data_X[-20:]) - data_Y_test) ** 2)))



if __name__ == "__main__":
    main()