#!/usr/bin/python

from data_utils import *
import numpy as np
from naive_bayes import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from sklearn import metrics
import random
def main():
    #read data
    x, labels = readData("C:/Users/marro/Repo/CS584/Generative_Learning/Data/spambase/spambase.data",",",scale=False)

    p = np.random.permutation(len(x))
    x = x[p]
    labels = labels[p]

    #get first 48 attributes
    x = x[:,range(0,48)]
    x = np.around(x)

    counts = np.empty([len(x), 1])
    for w in range(0,len(x)):
        counts.fill(random.randint(100,150))


    predicted_labels = classifyAllBinomial(x,x,counts,labels)

    print(metrics.accuracy_score(labels,predicted_labels))
    print(predicted_labels)
    print("Kfold Accuracy, recall, precission,tp,tn,fp,fn: {}".format(kfoldCrossValidationBinomial(x,counts, labels, 10, 1)))
if __name__ == "__main__":
    main()