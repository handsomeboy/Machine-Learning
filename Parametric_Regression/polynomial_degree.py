#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from data_utils import *
from regression import *

def main():

    data_X, data_Y = readData("data/svar-set2.dat", " ")

    #plot original data
    plt.scatter(data_X, data_Y,  color='black')
    plt.xlabel('x')
    plt.ylabel('y')

    color = ["red","blue","green"]
    handles = np.empty([0,1])
    for degree in range(1,4):
        poly = preprocessing.PolynomialFeatures(degree)
        newx = poly.fit_transform(data_X)
        thetas = fit_model(newx, data_Y)
        handle = plt.scatter(data_X,predict(thetas,newx),color=color[degree-1], label = "Polynomial degree: {0:d}".format(degree))
        handles = np.append(handles,handle)
        print("degree: {}, 10-fold cross valdiation: {}".format(degree,kfold_validation(newx,data_Y,10, fit_model)))
    plt.legend(handles=[handles[0],handles[1],handles[2]])
    plt.show()

if __name__ == "__main__":
    main()