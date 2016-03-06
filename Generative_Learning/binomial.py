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
    x = np.around(x)

    counts = np.empty([len(x), 1])
    counts.fill(100)

    x

    # classify(x[0],x,labels)
    # predicted_labels = classify_binomial(x[0,:],x,counts,labels)
    predicted_labels = classifyAllBinomial(x,x,counts,labels)
    # for i in range(0,2):
    #     getMembership(,i)
    #
    print(metrics.accuracy_score(labels,predicted_labels))
    print(predicted_labels)

if __name__ == "__main__":
    main()