#!/usr/bin/python

from data_utils import *
import numpy as np
from naive_bayes import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from sklearn import metrics


def main():
    #read data
    x, labels = readData("C:/Users/marro/Repo/CS584/Generative_Learning/Data/spambase/spambase.data",",",scale=False)
    #get first 48 attributes
    p = np.random.permutation(len(x))
    x = x[p]
    labels = labels[p]

    x = x[:,range(0,48)]
    binarizer = preprocessing.Binarizer().fit(x)

    x = binarizer.transform(x)

    predicted_labels = classifyAll(x,x,labels)

    print("Kfold Accuracy, recall, precission,tp,tn,fp,fn: {}".format(kfoldCrossValidation(x,labels, 10, 1)))
    print(metrics.accuracy_score(labels,predicted_labels))
    print(predicted_labels)

if __name__ == "__main__":
    main()