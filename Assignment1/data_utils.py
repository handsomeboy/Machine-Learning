import numpy as np
from sklearn import preprocessing
def readData(filename, delim, scale=True):
    data = np.genfromtxt(filename, delimiter=delim)
    x = data[:, 0:len(data[0])-1]
    y = data[:, np.newaxis, len(data[0])-1][:,0]
    if(scale):
        x =  preprocessing.scale(x);
    return (x,y)