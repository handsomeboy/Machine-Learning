#!/usr/bin/python

from scipy.spatial.distance import pdist
import numpy as np
from copy import deepcopy
import math
from math import exp
import numpy.linalg as linalg
from sklearn import metrics
from sklearn.cross_validation import KFold
import random
from metrics import *
import pyximport;
import learnmultilayer

def indicator(condition):
    if(condition):
        return 1
    else:
        return 0

def classify(v,w, x, labels, h):
    kmax = None
    max = -9999999999

    z = np.empty([h])
    k = len(labels)
    ybar = np.empty([k])

    #compute z
    for j in range(h):
        z[j] = sigmoid(w[j],x)
    #computer ybar
    for j in range(k):
        ybar[j] = softmax(v,z,j,labels,getSoftmaxDen(v,z,labels))

    for j in range(k):
        if(ybar[j] > max):
            kmax = j
            max = ybar[j]
    return labels[kmax]

def loglikelihood(x,y, ybar, labels):
    sum = 0
    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]
        sum2 = 0
        for k in range(len(labels)):
            ind = indicator(k == yi)
            sum2 += (ind * math.log(ybar[i,k]))
        sum += sum2
    return -sum

def train(x,labels, h, threshold=0.01, maxIterations = 900, learning_rate=0.001):
    classes, y = np.unique(labels, return_inverse=True)
    return learnmultilayer.gradient_descent(x, y, classes,  threshold=threshold, h=h, maxIterations=maxIterations, learning_rate=learning_rate);

def getSoftmaxDen(thetas, x, labels):
    den = 0
    for k in range(len(labels)):
        den += exp(np.dot(np.transpose(thetas[k]),x))
    return den

def softmax(thetas, x, j, labels, softmaxDen):
    numerator = exp(np.dot(np.transpose(thetas[j]),x))
    # denominator = np.sum(exp(np.dot(np.transpose(thetas),x)))

    return numerator / softmaxDen

def sigmoid(thetas, x):
    return 1 / (1 + np.exp(-np.dot(np.transpose(thetas),x)))

#gradient descent algorithm
def gradient_descent(x,y, labels, h, threshold=0.001, maxIterations=900, delta=9999, learning_rate=0.001, iv=None,iw=None):
    #iterative solution
    iterations = 0
    k = len(labels)
    n = x.shape[1]
    m = x.shape[0]
    if(iv==None):
        v = np.random.rand(k,h)/10
        w = np.random.rand(h,n)/10
    else:
        v = deepcopy(iv)
        w = deepcopy(iw)
    z = np.empty([m,h])
    ybar = np.empty([m,k])
    delta = 9999
    logl = 0
    all_likelihoods = np.empty([0,2])
    z = np.transpose(sigmoid(np.transpose(w),np.transpose(x)))
    #computer ybar
    for i in range(m):
        for j in range(k):
            ybar[i,j] = softmax(v,z[i],j,labels,getSoftmaxDen(v,z[i],labels))
    newloglikelihood = loglikelihood(x,y,ybar,labels)
    print(newloglikelihood)
    delta = newloglikelihood - logl
    all_likelihoods = np.append(all_likelihoods,[[0,loglikelihood(x,y,ybar,labels)]],axis=0)
    while (iterations < maxIterations and delta > threshold):
        #compute z
        # for i in range(m):
        #     for j in range(h):
        #         z[i,j] = sigmoid(w[j],x[i])
        z = np.transpose(sigmoid(np.transpose(w),np.transpose(x)))
        #computer ybar
        for i in range(m):
            for j in range(k):
                ybar[i,j] = softmax(v,z[i],j,labels,getSoftmaxDen(v,z[i],labels))
        newloglikelihood = loglikelihood(x,y,ybar,labels)
        print(iterations,newloglikelihood)
        delta = newloglikelihood - logl
        #update v
        for j in range(k):
            sum = 0
            for i in range(m):
                sum += ((ybar[i,j] - indicator(y[i]==j)) * z[i])
            v[j] -= learning_rate * sum
        #update w
        for j in range(h):
            sum = 0
            for i in range(m):
                sum2 = 0
                for l in range(k):
                    sum2 += ((ybar[i,l] - indicator(y[i]==l)) * v[l,j])
                sum += sum2 * z[i,j] * (1-z[i,j]) * x[i]
            w[j] -= learning_rate * sum
        iterations += 1
        all_likelihoods = np.append(all_likelihoods,[[iterations,loglikelihood(x,y,ybar,labels)]],axis=0)
    return v,w, iterations, all_likelihoods

def getAllSoftmax(thetas, x, labels):
    all_softmax = np.empty([x.shape[0],len(labels)])
    for i in range(x.shape[0]):
        softmaxDen = getSoftmaxDen(thetas,x[i],labels)
        for j in range(len(labels)):
            all_softmax[i,j] = softmax(thetas, x[i], j, labels, softmaxDen)
    return all_softmax

def classify_all(x,data,y,maxIterations, learning_rate, v = None, w = None, h=None, ):
    classes, y = np.unique(y, return_inverse=True)
    if (v == None):
        v,w,it,al = train(data,y,  h=h, maxIterations=maxIterations,learning_rate=learning_rate)
    predictedLabels = list()
    for i in range(0,x.shape[0]):
        predictedLabels.append(classify(v,w,x[i], classes, h=h))
    return predictedLabels

def kfoldCrossValidation(x,labels,k, positive_class, h, maxIterations,learning_rate):
    kf = KFold(len(x), n_folds=k)
    all_metrics = list()
    all_n = list()
    for train_index, test_index in kf:
        x_train = x[train_index]
        labels_train = labels[train_index]
        x_test = x[test_index]
        labels_test = labels[test_index]
        predictedLabels = classify_all(x_test,x_train,labels_train, maxIterations=maxIterations,learning_rate=learning_rate, h=h)
        accuracy = getAccuracy(labels_test,predictedLabels, positive_class)
        recall = getRecall(labels_test,predictedLabels, positive_class)
        precision = getPrecision(labels_test,predictedLabels, positive_class)

        builtinaccuracy = metrics.accuracy_score(labels_test, predictedLabels)
        fmeasure = getFMeasure(labels_test,predictedLabels,positive_class)
        all_metrics.append([builtinaccuracy, accuracy,recall,precision,fmeasure])
    return np.mean(all_metrics,axis=0)