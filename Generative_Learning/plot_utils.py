#!/usr/bin/python

from data_utils import *
import numpy as np
from generative_learning import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
import matplotlib.mlab as mlab
import scipy.stats as stats
import generative_learning
from matplotlib.colors import ListedColormap

def plot_surface(data_X,data_Y, x_1, x_2, ax=None, threshold=0.0, contourf=False):
    """Plots the decision surface of ``est`` on features ``x1`` and ``x2``. """
    xx1, xx2 = np.meshgrid(np.linspace(x_1.min(), x_1.max(), 100),
                           np.linspace(x_2.min(), x_2.max(), 100))
    # plot the hyperplane by evaluating the parameters on the grid
    X_pred = np.c_[xx1.ravel(), xx2.ravel()]  # convert 2d grid into seq of points

    # pred = est.predict(X_pred)
    pred = np.empty([X_pred.shape[0],1])
    for i in range(0,X_pred.shape[0]):
        pred[i] = discriminative(X_pred[i],data_X,data_Y)

    Z = pred.reshape((100, 100))  # reshape seq to grid
    if ax is None:
        ax = plt.gca()
    # plot line via contour plot

    if contourf:
        ax.contourf(xx1, xx2, Z, levels=np.linspace(0, 1.0, 10), cmap=plt.cm.RdBu, alpha=0.6)
    ax.contour(xx1, xx2, Z, levels=[threshold], colors='black')
    ax.set_xlim((x_1.min(), x_1.max()))
    ax.set_ylim((x_2.min(), x_2.max()))

def main():
    from sklearn.linear_model import LinearRegression

    x, y = readData("C:/Users/marro/Repo/CS584/Generative_Learning/Data/banknote/data_banknote_authentication.txt",",",scale=False)

    x = x[:,(0,1)]
    # encode class labels
    classes, y = np.unique(y, return_inverse=True)
    # y = (y * 2) - 1  # map {0, 1} to {-1, 1}

    # fit OLS regression
    est = LinearRegression(fit_intercept=True, normalize=True)
    est.fit(x, y)

    # plot data and decision surface
    ax = plt.gca()
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.scatter(x[:,0], x[:,1], c=(y == 1), cmap=cm_bright)

    plot_surface(x,y, x[:, 0], x[:, 1], ax=ax)
    plt.show()
