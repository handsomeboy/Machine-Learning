#!/usr/bin/python

from data_utils import *
import numpy as np
from logistic_regression_2class import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from metrics import *

#map matrix to a higher polynomial degree
def mapFeatures(x, degree):
    poly = preprocessing.PolynomialFeatures(degree,interaction_only=True)
    return poly.fit_transform(x)

def main():
     #read data
    x, labels = readData("Data/banknote/data_banknote_authentication.txt",",",scale=False)

    #shuffle
    p = np.random.permutation(len(x))
    x = x[p]
    labels = labels[p]

    #train with all data
    thetas,all_likelihoods = train(x,labels, maxIterations=1000,learning_rate=0.0003)
    print("Training accuracy: {}".format(getAccuracy(labels,classify_all(x,x,labels,thetas),1)))

    #evaluate with 10-fold cross validation
    print("Kfold Accuracy, recall, precission,tp,tn,fp,fn: {}".format(kfoldCrossValidation(x,labels, 10, 1, 1000, 0.0003)))

    # map matrix to a higher polynomial degree and evaluate with kfold cross validation
    x = mapFeatures(x,2)
    print("Kfold Accuracy, recall, precission,tp,tn,fp,fn: {}".format(kfoldCrossValidation(x,labels, 10, 1, 1000, 0.00003)))

if __name__ == "__main__":
    main()