#!/usr/bin/python
from sklearn import metrics
import numpy as np

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

def getCM2(labels,predictedLabels,l1,l2):
    sum = 0
    for i in range(0,labels.shape[0]):
        if(predictedLabels[i] == l2 and labels[i] == l1):
            sum += 1
    return sum

def getCM(labels,predictedLabels):
    cm = np.empty([np.unique(labels).size,np.unique(labels).size])
    for l1 in range(0, np.unique(labels).size):
        row = np.empty(np.unique(labels).size)
        for l2 in range(0, np.unique(labels).size):
            row[l2] = getCM2(labels,predictedLabels,l1,l2)
        cm[l1] = row
    return cm

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

def getFMeasure(labels,predictedLabels, positive_label):
    precision = getPrecision(labels,predictedLabels, positive_label)
    recall = getRecall(labels,predictedLabels, positive_label)
    fmeasure = (2*(precision*recall))/(precision+recall)
    print ("Built-in fmeasure = {}".format(metrics.f1_score(labels, predictedLabels)))
    print ("fmeasure = {}".format(fmeasure))
    return fmeasure

