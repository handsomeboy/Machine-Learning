#!/usr/bin/python

from data_utils import *
import numpy as np
from gda import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from metrics import *

def main():
    #read data
    x, labels = readData("C:/Users/marro/Repo/CS584/Generative_Learning/Data/banknote/data_banknote_authentication.txt",",",scale=False)

    #shuffle
    p = np.random.permutation(len(x))
    x = x[p]
    labels = labels[p]

    #get first feature to make it 1D
    x = x[:,np.newaxis,0]

    print("Training accuracy: {}".format(getAccuracy(labels,classifyAll(x,x,labels),1)))
    print("Kfold Accuracy, recall, precission,tp,tn,fp,fn, fmeasure: {}".format(kfoldCrossValidation(x,labels, 10, 1)))

    #find parameters
    c0_examples = x[np.ix_(labels == 0.0)]
    c1_examples = x[np.ix_(labels == 1)]

    #plot data before
    fig = plt.figure()
    ax=fig.add_subplot(111)
    colors = ['red','green']
    #plot in the same line

    #plot in different lines
    ax.scatter(c0_examples, np.ones(c0_examples.shape[0]), c=labels[np.ix_(labels == 0.0)], lw=0,cmap=matplotlib.colors.ListedColormap(colors))
    colors = ['green','red']

    ax.scatter(c1_examples, np.zeros(c1_examples.shape[0]), c=labels[np.ix_(labels == 1)], lw=0,cmap=matplotlib.colors.ListedColormap(colors))
    ax.yaxis.set_visible(False)
    plt.xlabel('X')
    plt.show()

    #predict and plot
    predictedLabels = list()
    for i in np.nditer(x):
        predictedLabels.append(classify(i,x,labels))
    fig = plt.figure()
    ax=fig.add_subplot(111)
    colors = ['red','green']
    ax.scatter(x, np.ones(x.shape[0]), c=predictedLabels, lw=0,cmap=matplotlib.colors.ListedColormap(colors))
    ax.yaxis.set_visible(False)
    plt.xlabel('X')
    plt.show()

    print("Confussion matrix: {}".format(getCM(labels,classifyAll(x,x,labels))))

if __name__ == "__main__":
    main()