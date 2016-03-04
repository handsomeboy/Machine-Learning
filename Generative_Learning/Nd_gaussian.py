#!/usr/bin/python

from data_utils import *
import numpy as np
from generative_learning import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
import matplotlib.mlab as mlab
import scipy.stats as stats
from matplotlib.colors import ListedColormap
import plot_utils as pu

def main():
    x, y = readData("C:/Users/marro/Repo/CS584/Generative_Learning/Data/banknote/data_banknote_authentication.txt",",",scale=False)

    #shuffle
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]

    # encode class labels
    classes, y = np.unique(y, return_inverse=True)

    print("Training accuracy: {}".format(getAccuracy(y,classifyAll(x,x,y),1)))
    print("Kfold Accuracy, recall, precission,tp,tn,fp,fn: {}".format(kfoldCrossValidation(x,y, 10, 1)))

if __name__ == "__main__":
    main()