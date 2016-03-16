#!/usr/bin/python

from scipy.spatial.distance import pdist
import numpy as np
import math
from math import exp
import numpy.linalg as linalg
from sklearn import metrics
from sklearn.cross_validation import KFold
import random

def classify(thetas, x):
    return sigmoid(thetas,x) > 0.5

def loglikelihood(thetas,x,y):
    sum = 0
    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]
        prediction = sigmoid(thetas,xi)
        if(yi == 1):
            sum += math.log(prediction)
        else:
            sum += math.log(1-prediction)
    return sum

def train(x,labels, threshold=0.01):
    classes, y = np.unique(labels, return_inverse=True)
    return gradient_descent(x,labels, threshold);

def sigmoid(thetas, x):
    return 1 / (1 + exp(-np.dot(np.transpose(thetas),x)))

#gradient descent algorithm
def gradient_descent(x,y, threshold=0.01, maxIterations=10000, delta=9999, learning_rate=0.0003 ):
    #iterative solution
    iterations = 0
    thetas = []

    #random start around 0
    for i in range(len(x[0])):
        thetas.append(random.randrange(1,10)/1000)

    all_likelihoods = np.empty([0,2])
    all_likelihoods = np.append(all_likelihoods,[[0, loglikelihood(thetas,x,y)]],axis=0)
    while (delta > threshold and iterations < maxIterations):
        sum=0
        for i in range(x.shape[0]):
            sum += (sigmoid(thetas,x[i]) - y[i]) * x[i]
        new_thetas = thetas - (learning_rate*sum)
        delta = loglikelihood(new_thetas,x,y) - loglikelihood(thetas,x,y)
        # print(likelihood(new_thetas,x,y))
        iterations += 1
        all_likelihoods = np.append(all_likelihoods, [[iterations,loglikelihood(new_thetas,x,y)]],axis=0 )
        #all_errors = np.append(all_errors,[[iterations,getMeanError(new_thetas,x,y)]],axis=0)
        thetas = new_thetas
    return thetas,all_likelihoods

def classify_all(x,data,y):
    thetas,all_likelihoods = train(data,y)
    predictedLabels = list()
    for i in range(0,x.shape[0]):
        predictedLabels.append(classify(thetas,x[i]))
    return predictedLabels