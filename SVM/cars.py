import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import plot_utils as pu
from sklearn import preprocessing
import svm as svm
import math
import pandas

def main():
    data = pandas.read_csv('Data/car.data', sep=',', header=0, index_col=False)
    data = pandas.get_dummies(data)
    arr = data.as_matrix()
    use = [k for k in range(arr.shape[0]) if (arr[k,0] == -1 or arr[k,0] == 1)]
    arr = arr[use]
    X = arr[:,range(1,22)]
    y = arr[:,0]

    #normalize
    # X = preprocessing.scale(X)
    # shuffle
    p = np.random.permutation(len(X))

    X = X[p]
    y = y[p]

    #train svm
    w, w0, support_vectors_idx = svm.train(X, y, c=99, eps=0.00001)

    #get accuracy
    predicted_labels = svm.classify_all(X,w,w0)
    print("Accuracy: {}".format(svm.getAccuracy(y,predicted_labels)))
    #
    # evaluate performance
    kfold = svm.kfoldCrossValidation(X, y, 10, 1, c=99, eps=0.00001)
    print (kfold)

    # evaluate performance with gaussina kernel function
    kfold = svm.kfoldCrossValidation(X, y, 10, 1, c=99, eps=0.00001, type='gaussian')
    print (kfold)

    # evaluate performance with polynomial kernel function
    kfold = svm.kfoldCrossValidation(X, y, 10, 1, c=99, eps=0.00001, type='polynomial')
    print (kfold)
if __name__ == "__main__":
    main()