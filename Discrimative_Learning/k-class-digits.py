#!/usr/bin/python

from data_utils import *
import numpy as np
from logistic_regression_kclass import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from get_digits_data import *
from metrics import *

def main():
    #read data
    X,y = getDigitsData()
    X_train, X_test = X[:70000], X[70000:]
    y_train, y_test = y[:70000], y[70000:]

    #shuffle
    p = np.random.permutation(len(X_train))
    X_train = X_train[p]
    y_train = y_train[p]

    all_metrics, all_n = kfoldCrossValidation3Classes(X_train,y_train, 10, 3)
    np.savetxt("foo.csv", all_n.reshape(10,10), delimiter=",", fmt='%7.2f')
    print("Kfold Accuracy, recall, precission,tp,tn,fp,fn: {}".format(all_metrics))
    print("all_n {}".format(all_n))

    #use 0.000005
    thetas, all_likelihoods = train(X_train,y,maxIterations=20, learning_rate=0.000005)
    print("Training accuracy: {}".format(getAccuracy(y_train,classify_all(X_train,X_train,y_train,thetas),1)))

    print(X_train.shape)
    print(y_train.shape)
    size=len(y_train)

    ## extract "3" digits and show their average"
    ind = [ k for k in range(size) if y_train[k]==3 ]
    extracted_images=X_train[ind,:]

    mean_image=extracted_images.mean(axis=0)
    imshow(mean_image.reshape(28,28), cmap=cm.gray)
    show()

if __name__ == "__main__":
    main()