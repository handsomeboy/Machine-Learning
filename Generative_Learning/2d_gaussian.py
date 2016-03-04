#!/usr/bin/python

from data_utils import *
import numpy as np
from generative_learning import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
import matplotlib.mlab as mlab
import scipy.stats as stats

def a_func(a,b):
    return a-b

def decision_boundary(x_vec, mu_vec1, mu_vec2):
    g1 = (x_vec-mu_vec1).T.dot((x_vec-mu_vec1))
    g2 = 2*( (x_vec-mu_vec2).T.dot((x_vec-mu_vec2)) )
    return g1 - g2

def main():
    #read data
    x, labels = readData("C:/Users/marro/Repo/CS584/Generative_Learning/Data/banknote/data_banknote_authentication.txt",",",scale=False)

    #shuffle
    p = np.random.permutation(len(x))
    x = x[p]
    labels = labels[p]

    #get first feature to make it 1D
    x = x[:,(0,1)]


    mu_vec1 = np.array([0,0])
    mu_vec2 = np.array([1,2])
    f, ax = plt.subplots(figsize=(7, 7))
    c1, c2 = "#3366AA", "#AA3333"
    x_vec = np.linspace(*ax.get_xlim())
    ax.contour(x_vec, x_vec,
           decision_boundary(x_vec, mu_vec1, mu_vec2),
           levels=[0], cmap="Greys_r")



if __name__ == "__main__":
    main()