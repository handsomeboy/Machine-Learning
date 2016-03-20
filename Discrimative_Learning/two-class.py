#!/usr/bin/python

from data_utils import *
import numpy as np
from logistic_regression_2class import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from metrics import *

#map matrix to a higher polynomial degree
def mapFeatures(x, degree):
    poly = preprocessing.PolynomialFeatures(degree,interaction_only=True)
    return poly.fit_transform(x)

def main():
     #read data
    x, labels = readData("C:/Users/marro/Repo/CS584/Generative_Learning/Data/banknote/data_banknote_authentication.txt",",",scale=False)

    #shuffle
    p = np.random.permutation(len(x))
    x = x[p]
    labels = labels[p]
    # thetas, all_likelihoods = train(x,labels, threshold=0.1)
    #
    # plt.plot(all_likelihoods[0:50,0],all_likelihoods[0:50,1], linewidth = 4)
    # plt.show()
    #
    # handles = np.empty([0,1])
    # for lw in [0.0002,0.0003, 0.0004]:
    #     #perform gradient descent
    #     thetas, all_likelihoods = train(x,labels, threshold=0.0001, maxIterations=1000, learning_rate=lw)
    #     # print("Itearative Method Coefficients: {},Iterations:{}".format(thetas,iterations))
    #     handle = plt.plot(all_likelihoods[0:50,0],all_likelihoods[0:50,1], linewidth = 4, label="Learning Weight = {:.6f}".format(lw))
    #     handles = np.append(handles,handle)
    # plt.legend(handles=[handles[0],handles[1],handles[2]])
    # plt.ticklabel_format(useOffset=False)
    # plt.xlabel("Iteration")
    # plt.ylabel("Squared Mean Error")
    # plt.show()

    thetas,all_likelihoods = train(x,labels, maxIterations=1000,learning_rate=0.0003)
    print("Training accuracy: {}".format(getAccuracy(labels,classify_all(x,x,labels,thetas),1)))
    # print("Kfold Accuracy, recall, precission,tp,tn,fp,fn: {}".format(kfoldCrossValidation(x,labels, 10, 1)))

    # x = mapFeatures(x,2)
    # print("Kfold Accuracy, recall, precission,tp,tn,fp,fn: {}".format(kfoldCrossValidation(x,labels, 10, 1)))

if __name__ == "__main__":
    main()