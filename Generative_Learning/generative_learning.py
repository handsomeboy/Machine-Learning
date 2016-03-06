#!/usr/bin/python

from scipy.spatial.distance import pdist
import numpy as np
import math
import numpy.linalg as linalg
from metrics import *
from sklearn.cross_validation import KFold

def getMembership(x, data, labels, label):
    class_examples = data[np.ix_(labels == label)]
    mean = np.mean(class_examples,axis=0)
    cov = np.cov(class_examples.T)
    prior = class_examples.shape[0] / data.shape[0]
    if(cov.size > 1):
        mahalaobis_distance = - 1/2 * np.dot(np.dot((x-mean).T, linalg.inv(cov)), (x-mean))
    else:
        mahalaobis_distance = - 1/2 * np.dot(np.dot((x-mean).T, 1/cov), (x-mean))
    if(cov.size >1):
        membership = (- math.log(linalg.det(cov))) + mahalaobis_distance + prior
    else:
        membership = (- math.log(math.sqrt(cov))) + mahalaobis_distance + prior
    return membership

def discriminative(x,data,labels):
    return getMembership(x,data,labels,1) - getMembership(x,data,labels,0)

def classify(x,data, labels):
    maxMembership = None
    maxLabel = None
    for label in np.unique(labels):
        membership = getMembership(x,data,labels,label)
        if maxMembership is None:
            maxMembership = membership
            maxLabel = label
        else:
            if membership>maxMembership:
                maxMembership = membership
                maxLabel = label
    return maxLabel

def classifyAll(x,data, labels):
    predictedLabels = list()
    for i in range(0,x.shape[0]):
        maxMembership = None
        maxLabel = None
        for label in np.unique(labels):
            membership = getMembership(x[i,:],data,labels,label)
            if maxMembership is None:
                maxMembership = membership
                maxLabel = label
            else:
                if membership>maxMembership:
                    maxMembership = membership
                    maxLabel = label
        predictedLabels.append(maxLabel)
    return predictedLabels

def kfoldCrossValidation(x,labels,k, positive_class):
    kf = KFold(len(x), n_folds=k)
    all_metrics = list()
    for train_index, test_index in kf:
        x_train = x[train_index]
        labels_train = labels[train_index]
        x_test = x[test_index]
        labels_test = labels[test_index]
        predictedLabels = classifyAll(x_test,x_train,labels_train)
        accuracy = getAccuracy(labels_test,predictedLabels, positive_class)
        recall = getRecall(labels_test,predictedLabels, positive_class)
        precision = getPrecision(labels_test,predictedLabels, positive_class)
        tp = getTP(labels_test,predictedLabels,positive_class)
        tn = getTN(labels_test,predictedLabels,positive_class)
        fp = getFP(labels_test,predictedLabels,positive_class)
        fn = getFN(labels_test,predictedLabels,positive_class)
        fmeasure = getFMeasure(labels_test,predictedLabels,positive_class)
        all_metrics.append([accuracy,recall,precision,tp,tn,fp,fn,fmeasure])
    return np.mean(all_metrics,axis=0)