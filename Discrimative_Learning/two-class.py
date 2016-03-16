#!/usr/bin/python

from data_utils import *
import numpy as np
from logistic_regression_2class import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from metrics import *

def main():
     #read data
    x, labels = readData("C:/Users/marro/Repo/CS584/Generative_Learning/Data/banknote/data_banknote_authentication.txt",",",scale=False)

    # thetas, all_likelihoods = train(x,labels, threshold=0.1)
    #
    # plt.plot(all_likelihoods[0:50,0],all_likelihoods[0:50,1], linewidth = 4)
    # plt.show()


    predictedLabels = classify_all(x,x,labels)
    print("Training accuracy: {}".format(getAccuracy(labels,classify_all(x,x,labels),1)))


if __name__ == "__main__":
    main()