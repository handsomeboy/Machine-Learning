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
    data = pandas.read_csv('Data/credits.data', sep=',', header=0, index_col=False)
    data = pandas.get_dummies(data)
    arr = data.as_matrix()
    X = arr[:,range(0,6) + range(7,47)]
    y = arr[:,6]

    # shuffle
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]

    #train svm
    # w,w0, support_vectors_idx = svm.train(X[:,[0,1,2,3,4,5,6,7]],y,c=999, eps=0.000001)
    w, w0, support_vectors_idx = svm.train(X, y, c=99999, eps=0.000000001)
    #plot result
    predicted_labels = svm.classify_all(X,w,w0)
    print("Accuracy: {}".format(svm.getAccuracy(y,predicted_labels)))

    kfold = svm.kfoldCrossValidation(X, y, 10, 1, c=99, eps=0.00001)
    print (kfold)

if __name__ == "__main__":
    main()