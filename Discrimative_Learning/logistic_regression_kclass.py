#!/usr/bin/python

from scipy.spatial.distance import pdist
import numpy as np
import math
from math import exp
import numpy.linalg as linalg
from sklearn import metrics
from sklearn.cross_validation import KFold
import random

def indicator(condition):
    if(condition):
        return 1
    else:
        return 0

def classify(thetas, x, labels):
    kmax = None
    max = -9999999999
    for k in range(len(labels)):
        value = softmax(thetas,x, k, labels, getSoftmaxDen(thetas,x,labels))
        if(value > max):
            kmax = k
            max = value
    return labels[kmax]

def loglikelihood(thetas,x,y, labels, all_softmax):
    sum = 0
    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]
        sum2 = 0
        for k in range(len(labels)):
            ind = indicator(k == yi)
            sum2 += (ind * math.log(all_softmax[i,k]))
        sum += sum2
    return sum

def train(x,labels, threshold=0.01):
    classes, y = np.unique(labels, return_inverse=True)
    return gradient_descent(x, y, classes,  threshold);

def getSoftmaxDen(thetas, x, labels):
    den = 0
    for k in range(len(labels)):
        den += exp(np.dot(np.transpose(thetas[k]),x))
    return den

def softmax(thetas, x, j, labels, softmaxDen):
    numerator = exp(np.dot(np.transpose(thetas[j]),x))
    # denominator = np.sum(exp(np.dot(np.transpose(thetas),x)))

    return numerator / softmaxDen

#gradient descent algorithm
def gradient_descent(x,y, labels, threshold=0.00001, maxIterations=30, delta=9999, learning_rate=0.0005 ):
    #iterative solution
    iterations = 0
    thetas = np.empty([len(labels),len(x[0])])
    new_thetas = np.empty([len(labels),len(x[0])])

    #random start around 0
    for j in range(len(labels)):
        for i in range(len(x[0])):
            thetas[j,i]=random.randrange(1,10)/1000

    all_likelihoods = np.empty([0,2])
    # all_likelihoods = np.append(all_likelihoods,[[0, loglikelihood(thetas,x,y,labels)]],axis=0)
    # while (delta > threshold and iterations < maxIterations):
    all_softmax = getAllSoftmax(thetas,x,labels)
    while (iterations < maxIterations):

        for j in range(len(labels)):
            # current_softmax_den = getSoftmaxDen(thetas, x[i], j, labels)
            #update class j thetas
            sum=0
            for i in range(x.shape[0]):
                sum += ( (all_softmax[i,j] - indicator(y[i] == j)) * x[i])
            new_thetas[j] = (thetas[j] - (learning_rate*sum))

        all_softmax_new = getAllSoftmax(new_thetas,x,labels)

        # print(thetas,new_thetas)
        delta = loglikelihood(new_thetas,x,y, labels, all_softmax_new) - loglikelihood(thetas,x,y, labels, all_softmax)
        print(loglikelihood(new_thetas,x,y,labels, all_softmax),delta)
        iterations += 1
        # all_likelihoods = np.append(all_likelihoods, [[iterations,loglikelihood(new_thetas,x,y,labels, all_softmax)]],axis=0 )
        #all_errors = np.append(all_errors,[[iterations,getMeanError(new_thetas,x,y)]],axis=0)
        for p in range(len(thetas)):
            thetas[p] = new_thetas[p]
        all_softmax = all_softmax_new
    return thetas,all_likelihoods

def getAllSoftmax(thetas, x, labels):
    all_softmax = np.empty([x.shape[0],len(labels)])
    for i in range(x.shape[0]):
        softmaxDen = getSoftmaxDen(thetas,x[i],labels)
        for j in range(len(labels)):
            all_softmax[i,j] = softmax(thetas, x[i], j, labels, softmaxDen)
    return all_softmax

def classify_all(x,data,y):
    classes, y = np.unique(y, return_inverse=True)
    thetas,all_likelihoods = train(data,y)
    predictedLabels = list()
    for i in range(0,x.shape[0]):
        predictedLabels.append(classify(thetas,x[i], classes))
    return predictedLabels