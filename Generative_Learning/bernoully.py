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
    x = x[:,range(0,48)]
    binarizer = preprocessing.Binarizer().fit(x)

    x = binarizer.transform(x)

    # c0_examples = x[np.ix_(labels == 0)]
    # c1_examples = x[np.ix_(labels == 1)]
    #
    # mean_c0 = np.mean(c0_examples,axis=0)
    # mean_c1 = np.mean(c1_examples,axis=0)

    # classify(x[0],x,labels)
    predicted_labels = classifyAll(x,x,labels)
    # for i in range(0,2):
    #     getMembership(,i)
    #
    print(metrics.accuracy_score(labels,predicted_labels))
    print(predicted_labels)

if __name__ == "__main__":
    main()