#!/usr/bin/python

from data_utils import *
import numpy as np
from gda import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
import matplotlib.mlab as mlab
import scipy.stats as stats
from matplotlib.colors import ListedColormap
import plot_utils as pu
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
def main():
    x, y = readData("C:/Users/marro/Repo/CS584/Generative_Learning/Data/banknote/data_banknote_authentication.txt",",",scale=False)

    #shuffle
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]

    x = x[:,(0,1)]
    # encode class labels
    classes, y = np.unique(y, return_inverse=True)

    print("Confussion matrix: {}".format(getCM(y,classifyAll(x,x,y))))
    print("Training accuracy: {}".format(getAccuracy(y,classifyAll(x,x,y),1)))
    print("Kfold Accuracy, recall, precission,tp,tn,fp,fn: {}".format(kfoldCrossValidation(x,y, 10, 1)))

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(0,1):
        precision[i], recall[i], _ = precision_recall_curve(y,
                                                        classifyAll(x,x,y))
        average_precision[i] = average_precision_score(y, classifyAll(x,x,y))

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall[0], precision[0], label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.show()

    # plot data and decision surface
    ax = plt.gca()
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.scatter(x[:,0], x[:,1], c=(y == 1), cmap=cm_bright)
    plt.xlabel("X1")
    plt.ylabel("X2")
    pu.plot_surface(x,y, x[:, 0], x[:, 1], ax=ax)
    plt.show()



if __name__ == "__main__":
    main()