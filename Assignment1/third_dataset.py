#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from data_utils import *
from regression import *

def second_dataset():

    data_X, data_Y = readData("data/svar-set3.dat", " ")

    #plot original data
    #plt.scatter(data_X, data_Y,  color='black')
    #plt.show()

    for degree in range(1,5):
        poly = preprocessing.PolynomialFeatures(degree)
        newx = poly.fit_transform(data_X)
        print("degree: {}, 10-fold cross valdiation: {}".format(degree,kfold_validation(newx,data_Y,10)))

    poly = preprocessing.PolynomialFeatures(2)
    z = poly.fit_transform(data_X)
    for training_size in np.arange(0.9,0.0,-0.1):
        data_X_train, data_Y_train, data_X_test, data_Y_test = splitDataSet(training_size,z,data_Y)
        thetas = fit_model1(data_X_train, data_Y_train)
        print("Training Size: {0:.1f}, Training Mean Squared Error: {1}".format(training_size, getMeanError(thetas,data_X_train,data_Y_train)))
        print("Testing Size: {0:.1f}, Testing Mean Squared Error: {1}\n".format(1-training_size, getMeanError(thetas,data_X_test,data_Y_test)))

        plt.clf()
        plt.close()
        plt.scatter(data_X_train[:,1], data_Y_train,  color='black')
        plt.scatter(data_X_train[:,1], predict(thetas,data_X_train), color='blue', linewidth=3)
        plt.show()
        plt.clf()
        plt.close()
        plt.scatter(data_X_test[:,1], data_Y_test,  color='black')
        plt.scatter(data_X_test[:,1], predict(thetas,data_X_test), color='blue', linewidth=3)
        plt.show()


if __name__ == "__main__":
    second_dataset()