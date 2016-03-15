#!/usr/bin/python

from scipy.spatial.distance import pdist
import numpy as np
import math
import numpy.linalg as linalg
from scipy.stats._continuous_distns import maxwell_gen
from sklearn import metrics
from sklearn.cross_validation import KFold
from math import factorial
from metrics import *

def getMembership(x,class_mean, prior):
    sum = 0
    for j in range(0,len(class_mean)):
        if(class_mean[j] == 0):
            aux = 0
        else:
            aux = math.log(class_mean[j])
        if( class_mean[j] == 1):
            aux2 = 0
        else:
            aux2 = math.log(1 - class_mean[j])
        sum += x[j]*aux + (1-x[j]) * aux2
    sum += math.log(prior)
    return sum

def combination(n,k):
    numerator=factorial(n)
    denominator=(factorial(k)*factorial(n-k))
    answer=numerator/denominator
    return answer

def getMembershipBinomial(x,alfas, prior, class_counts, total_class_counts):
    sum = 0
    for j in range(0,len(alfas)):
        sum += (math.log(combination(class_counts[j],x[j]))) +  (x[j]*math.log(alfas[j])) + ((class_counts[j]-x[j]) * math.log(1 - alfas[j]))
    sum += math.log(prior)
    return sum

def classify(x, data, y):
    classes, y = np.unique(y, return_inverse=True)
    max_label = None
    max = None
    for class_label in np.nditer(classes):
        examples = data[np.ix_(y == class_label)]
        class_mean = np.mean(examples,axis=0)
        prior = len(examples) / len(data)
        membership = getMembership(x,class_mean, prior)
        if(max_label is None):
            max_label = class_label
            max = membership
        else:
            if membership>max:
                max = membership
                max_label = class_label
    return max_label


def classify_binomial(x, data, counts, y):
    classes, y = np.unique(y, return_inverse=True)
    max_label = None
    max = None
    for class_label in np.nditer(classes):
        class_examples = data[np.ix_(y == class_label)]
        class_counts = counts[np.ix_(y == class_label)]
        total_class_counts = sum(class_counts)
        alfas = (class_examples.sum(axis=0) + 0.01)/(total_class_counts + 0.01)

        prior = len(class_examples) / len(data)
        membership = getMembershipBinomial(x,alfas, prior, class_counts, total_class_counts)
        if(max_label is None):
            max_label = class_label
            max = membership
        else:
            if(class_label == 0):
                if membership>max:
                    max = membership
                    max_label = class_label
            else:
                if membership>(max+8.5):
                    max = membership
                    max_label = class_label
    return max_label


def classifyAll(x,data, labels):
    predictedLabels = list()
    for i in range(0,x.shape[0]):
        predictedLabels.append(classify(x[i,:],data,labels))
    return predictedLabels

def classifyAllBinomial(x,data, counts,labels):
    predictedLabels = list()
    for i in range(0,x.shape[0]):
        predictedLabels.append(classify_binomial(x[i,:],data,counts,labels))
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

def kfoldCrossValidationBinomial(x,counts,labels,k, positive_class):
    kf = KFold(len(x), n_folds=k)
    all_metrics = list()
    for train_index, test_index in kf:
        x_train = x[train_index]
        labels_train = labels[train_index]
        x_test = x[test_index]
        labels_test = labels[test_index]
        predictedLabels = classifyAllBinomial(x_test,x_train,counts, labels_train)
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