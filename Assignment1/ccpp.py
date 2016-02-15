#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from data_utils import *
from regression import *
import  scipy.stats as stats

def housing():

    X, data_Y = readData("data/CCPP/Folds.csv", delim=",", skipHeader=True, scale=False)
    col_mean = stats.nanmean(X,axis=0)
    inds = np.where(np.isnan(X))
    print (inds)
    X[inds]=np.take(col_mean,inds[1])
    #feature 5: overfitting
    for i in range(0,4,1):
        data_X = X[:,np.newaxis,i]
        z = mapFeatures(data_X,2)
        thetas = fit_model1(z, data_Y)
        print("Method 1 Coefficients: {}\n".format(thetas))
        plt.scatter(data_X, data_Y,  color='black')
        plt.scatter(data_X, predict(thetas,z), color='blue',
                 linewidth=3)

        plt.show()



if __name__ == "__main__":
    housing()