#!/usr/bin/python

from scipy.spatial.distance import pdist
import numpy as np
import math
from math import exp
import numpy.linalg as linalg
from sklearn import metrics
from sklearn.cross_validation import KFold
import random
from metrics import *
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

def train(x,labels, threshold=0.0001, maxIterations=400, learning_rate=0.00001):
    classes, y = np.unique(labels, return_inverse=True)
    return gradient_descent(x,labels, threshold, maxIterations=maxIterations, learning_rate=learning_rate);

def sigmoid(thetas, x):
    return 1 / (1 + exp(-np.dot(np.transpose(thetas),x)))

#gradient descent algorithm
def gradient_descent(x,y, threshold=0.01, maxIterations=1000, delta=9999, learning_rate=0.0003 ):

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
        iterations += 1
        all_likelihoods = np.append(all_likelihoods, [[iterations,loglikelihood(new_thetas,x,y)]],axis=0 )
        thetas = new_thetas
    return thetas,all_likelihoods

def classify_all(x,data,y, thetas=None, maxIterations=400, learning_rate=0.00001):
    if(thetas == None):
        thetas,all_likelihoods = train(data,y, maxIterations=maxIterations, learning_rate=learning_rate)
    predictedLabels = list()
    for i in range(0,x.shape[0]):
        predictedLabels.append(classify(thetas,x[i]))
    return predictedLabels

def kfoldCrossValidation(x,labels,k, positive_class, maxIterations, learning_rate):
    kf = KFold(len(x), n_folds=k)
    all_metrics = list()
    for train_index, test_index in kf:
        x_train = x[train_index]
        labels_train = labels[train_index]
        x_test = x[test_index]
        labels_test = labels[test_index]
        predictedLabels = classify_all(x_test,x_train,labels_train, maxIterations=maxIterations, learning_rate=learning_rate)
        accuracy = getAccuracy(labels_test,predictedLabels, positive_class)
        recall = getRecall(labels_test,predictedLabels, positive_class)
        precision = getPrecision(labels_test,predictedLabels, positive_class)
        tp = getTP(labels_test,predictedLabels,positive_class)
        tn = getTN(labels_test,predictedLabels,positive_class)
        fp = getFP(labels_test,predictedLabels,positive_class)
        fn = getFN(labels_test,predictedLabels,positive_class)
        fmeasure = getFMeasure(labels_test,predictedLabels,positive_class)
        print(accuracy)
        all_metrics.append([accuracy,recall,precision,tp,tn,fp,fn,fmeasure])
    return np.mean(all_metrics,axis=0)