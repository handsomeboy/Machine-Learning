import numpy as np

def readData(filename, delim):
    data = np.genfromtxt(filename, delimiter=delim)
    x = data[:, 0:len(data[0])-1]
    y = data[:, np.newaxis, len(data[0])-1][:,0]
    return (x,y)