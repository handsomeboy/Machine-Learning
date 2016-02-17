#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from data_utils import *
from regression import *
import  scipy.stats as stats

def housing():

    data = np.genfromtxt("data/MPG/mpg.csv", delimiter=",")
    data_Y = data[:,0]
    X_ = data[:, 1:len(data[0])-1]


    col_mean = stats.nanmean(X_,axis=0)
    inds = np.where(np.isnan(X_))
    X_[inds]=np.take(col_mean,inds[1])

    X_scaled = preprocessing.scale(X_);

    # #degree
    # X = X[:,np.newaxis,3]
    # values = np.empty([10,2])
    # for degree in range(1,10,1):
    #     #map features
    #     newx = mapFeatures(X,degree)
    #     performance = kfold_validation(newx,data_Y,10, fit_model1)
    #     values[degree-1] = [int(degree),performance]
    #     print("degree: {}, 10-fold cross valdiation: {}".format(degree,kfold_validation(newx,data_Y,10, fit_model1)))
    # plt.clf()
    # plt.close()
    # plt.bar(values[:,np.newaxis,0],values[:,np.newaxis,1])
    # plt.xlabel('Polynomial Degree')
    # plt.ylabel('10 fold cross validation mean testing error')
    # plt.show()
    #
    # #training size
    # values = np.empty([9,2])
    # newx = mapFeatures(X,2)
    # for k in range(2,10,1):
    #     #map features
    #     performance = kfold_validation(newx,data_Y,k, fit_model1)
    #     values[k-2] = [int(k),performance]
    #     print("degree: {}, 10-fold cross valdiation: {}".format(degree,kfold_validation(newx,data_Y,10, fit_model1)))
    # plt.clf()
    # plt.close()
    # plt.bar(values[:,np.newaxis,0],values[:,np.newaxis,1])
    # plt.xlabel('K Folds')
    # plt.ylabel('Mean testing error')
    # plt.show()
    #
    # plt.scatter(X, data_Y,  color='black')
    # plt.xlabel('horse power')
    # plt.ylabel('mpg')
    # plt.show()

    #multivariate
    values = np.empty([4,2])
    for degree in range(1,5):
        #map features
        newx = mapFeatures(X_scaled,degree)
        print(newx.shape)
        performance = kfold_validation(newx,data_Y,10, fit_model)
        values[degree-1] = [degree,performance]
        print("degree: {}, 10-fold cross valdiation: {}".format(degree,performance))
    plt.plot(values[:,np.newaxis,0],values[:,np.newaxis,1])
    plt.xlabel('Polynomial Degree')
    plt.ylabel('10 fold cross validation mean testing error')
    plt.show()

    #training # #training size
    values = np.empty([9,2])
    newx = mapFeatures(X_scaled,2)
    for k in range(2,10,1):
        #map features
        performance = kfold_validation(newx,data_Y,k, fit_model)
        values[k-2] = [int(k),performance]
        print("degree: {}, 10-fold cross valdiation: {}".format(degree,kfold_validation(newx,data_Y,10, fit_model)))
    plt.xlabel('K Folds')
    plt.ylabel('Mean testing error')
    plt.bar(values[:,np.newaxis,0],values[:,np.newaxis,1])
    plt.show()

if __name__ == "__main__":
    housing()