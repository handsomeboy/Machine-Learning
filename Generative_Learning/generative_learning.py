#!/usr/bin/python

from scipy.spatial.distance import pdist
import numpy as np
import math
import numpy.linalg as linalg
from sklearn import metrics
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

def getTP(labels, predictedLabels, class_label):
    sum = 0
    for i in range(0,labels.shape[0]):
        if(labels[i] == class_label and predictedLabels[i] ==  class_label):
            sum += 1
    return sum

def getTN(labels, predictedLabels, class_label):
    sum = 0
    for i in range(0,labels.shape[0]):
        if(labels[i] != class_label and predictedLabels[i] != class_label):
            sum += 1
    return sum

def getFP(labels, predictedLabels, class_label):
    sum = 0
    for i in range(0,labels.shape[0]):
        if(predictedLabels[i] == class_label and labels[i] != class_label):
            sum += 1
    return sum

def getFN(labels, predictedLabels, class_label):
    sum = 0
    for i in range(0,labels.shape[0]):
        if(predictedLabels[i] != class_label and labels[i] == class_label):
            sum += 1
    return sum

def getAccuracy(labels,predictedLabels, positive_label):
    #predictedLabels = classifyAll(x,x,labels)
    totalExamples = labels.shape[0]
    accuracy = (getTP(labels,predictedLabels,positive_label) + getTN(labels,predictedLabels,positive_label))/ totalExamples
    print ("Built-in accuracy = {}".format(metrics.accuracy_score(labels, predictedLabels)))
    print ("Accuracy = {}".format(accuracy))
    return accuracy

def getRecall(labels,predictedLabels, positive_label):
    if (getTP(labels,predictedLabels,positive_label) == 0):
        recall = 0
    else:
        recall = getTP(labels,predictedLabels,positive_label)/(getTP(labels,predictedLabels,positive_label)+getFN(labels,predictedLabels,positive_label))
    print ("Built-in recall = {}".format(metrics.recall_score(labels, predictedLabels)))
    print ("Recall = {}".format(recall))
    return recall

def getPrecision(labels,predictedLabels, positive_label):
    if(getTP(labels,predictedLabels,positive_label) == 0):
        precision = 0
    else:
        precision = getTP(labels,predictedLabels,positive_label)/(getTP(labels,predictedLabels,positive_label)+getFP(labels,predictedLabels,positive_label))
    print ("Built-in precision = {}".format(metrics.precision_score(labels, predictedLabels)))
    print ("Precision = {}".format(precision))
    return precision

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
        all_metrics.append([accuracy,recall,precision,tp,tn,fp,fn])
    return np.mean(all_metrics,axis=0)