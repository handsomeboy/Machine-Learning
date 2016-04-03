#!/usr/bin/python

import numpy as np
from generate_datasets import mapFeatures
import matplotlib.pyplot as plt
import pylab
import matplotlib
import matplotlib.mlab as mlab
from matplotlib.colors import ListedColormap

def plot_surface(data_X,data_Y, x_1, x_2, disc_func, ax=None, threshold=0.0, contourf=False):
    """Plots the decision surface of ``est`` on features ``x1`` and ``x2``. """
    xx1, xx2 = np.meshgrid(np.linspace(x_1.min(), x_1.max(), 1000),
                           np.linspace(x_2.min(), x_2.max(), 1000))
    # plot the hyperplane by evaluating the parameters on the grid
    X_pred = np.c_[xx1.ravel(), xx2.ravel()]  # convert 2d grid into seq of points

    X_pred = mapFeatures(X_pred,1)
    # pred = est.predict(X_pred)
    pred = np.empty([X_pred.shape[0],1])
    for i in range(0,X_pred.shape[0]):
        pred[i] = disc_func(X_pred[i])

    Z = pred.reshape((1000, 1000))  # reshape seq to grid
    if ax is None:
        ax = plt.gca()
    # plot line via contour plot

    if contourf:
        ax.contourf(xx1, xx2, Z, levels=np.linspace(0, 1.0, 10), cmap=plt.cm.RdBu, alpha=0.6)
    ax.contour(xx1, xx2, Z, levels=[threshold], colors='black')
    ax.set_xlim((x_1.min(), x_1.max()))
    ax.set_ylim((x_2.min(), x_2.max()))
