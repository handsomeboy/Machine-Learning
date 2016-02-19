#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from sklearn.cross_validation import _PartitionIterator

from data_utils import *
from regression import *
from scipy.spatial.distance import pdist, squareform
import scipy
import math
def main():

    #read data
    data_X, data_Y = readData("data/housing/housing.data", scale=True)

    #add 1's in first column
    z = mapFeatures(data_X,1)

    #compute alphas
    alphas = solveDual(z,data_Y,1)

    #obtain prediction for an example of the training set
    example = z[3]
    print("predicted: {}".format(predictDual(z,alphas,example,1)))
    print("real: {}".format(data_Y[3]))

    #evaluate with 10 fold cross validation
    values = np.empty([7,2])
    for sigma in range(1,8):
        performance = kfold_validation_dual(z,data_Y,10,sigma)
        print("10-fold cross validation mean squared error for sigma = {} is {}\n".format(sigma, performance))
        values[sigma-1] = [int(sigma),performance]
    plt.xlabel('Sigma')
    plt.ylabel('Mean Squared Error (10 Fold Cross Validation')
    plt.plot(values[:,np.newaxis,0],values[:,np.newaxis,1])
    plt.show()
if __name__ == "__main__":
    main()