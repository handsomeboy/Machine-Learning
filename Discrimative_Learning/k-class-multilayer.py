#!/usr/bin/python

from data_utils import *
import numpy as np
from multilayer import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from metrics import *

def main():
    #read data
    x, labels = readData("C:/Users/marro/Repo/CS584/Generative_Learning/Data/iris.data",",",scale=False)

    #use 0.0005, and 500 iterations

    print("Training accuracy: {}".format(getAccuracy(labels,classify_all(x,x,labels),1)))

if __name__ == "__main__":
    main()