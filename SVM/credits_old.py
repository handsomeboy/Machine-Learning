import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import plot_utils as pu
from sklearn import preprocessing
import svm as svm
import math

def main():

    dic = {'a':0, 'b':1, '?':-1}
    data = np.genfromtxt('Data/credits.data', skip_header=True, delimiter=',',
                         usecols=[0,1,2,7,10,15],converters={0: lambda s: dic[s]})

    use = [k for k in range(len(data)) if data[k][0] != -1 and (not math.isnan(data[k][1]))]
    data = data[use]

    X = np.empty([len(data),5])
    y = np.empty([len(data), 1])
    for i in range(len(data)):

        for j in range(len(data[i])-1):
            X[i,j] = data[i][j]
        y[i] = data[i][5]



    # preprocessing.scale(X[:,1])
    #train svm
    w,w0, support_vectors_idx = svm.train(X[:,range(1,5)],y,c=10, eps=1)

    #plot result
    predicted_labels = svm.classify_all(X[:,range(1,5)],w,w0)
    print("Accuracy: {}".format(svm.getAccuracy(y,predicted_labels)))


if __name__ == "__main__":
    main()