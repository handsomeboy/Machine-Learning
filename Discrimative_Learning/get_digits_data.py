from sklearn.datasets import fetch_mldata
import numpy as np
import os
from metrics import *
from pylab import *

def getDigitsData():
    mnist = fetch_mldata('MNIST original')
    print(mnist.data.shape)
    print(mnist.target.shape)
    print(np.unique(mnist.target))

    X, y = mnist.data / 255., mnist.target
    return X,y

