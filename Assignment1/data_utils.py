import numpy as np

def readData(filename, delim):
    data = np.genfromtxt(filename, delimiter=delim)
    x = data[:, np.newaxis, 0]
    y = data[:, np.newaxis, 1]
    return (x,y)