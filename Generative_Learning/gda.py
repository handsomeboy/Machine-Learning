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

def kfoldCrossValidation3Classes(x,labels,k, positive_class):
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
        c00 = getN(labels_test,predictedLabels,0,0)
        c01 = getN(labels_test,predictedLabels,0,1)
        c02 = getN(labels_test,predictedLabels,0,2)
        c10 = getN(labels_test,predictedLabels,1,0)
        c11 = getN(labels_test,predictedLabels,1,1)
        c12 = getN(labels_test,predictedLabels,1,2)
        c20 = getN(labels_test,predictedLabels,2,0)
        c21 = getN(labels_test,predictedLabels,2,1)
        c22 = getN(labels_test,predictedLabels,2,2)

        fmeasure = getFMeasure(labels_test,predictedLabels,positive_class)
        all_metrics.append([accuracy,recall,precision,tp,tn,fp,fn,fmeasure,c00,c01,c02,c10,c11,c12,c20,c21,c22])
    return np.mean(all_metrics,axis=0)