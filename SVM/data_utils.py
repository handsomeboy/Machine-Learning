import numpy as np
from sklearn import preprocessing
def readData(filename, delim=None, scale=True,skipHeader=False):
    if(delim is None):
        data = np.genfromtxt(filename, skip_header=skipHeader)
    else:
        data = np.genfromtxt(filename, delimiter=delim, skip_header=skipHeader)
    x = data[:, 0:len(data[0])-1]
    y = data[:, np.newaxis, len(data[0])-1][:,0]
    if(scale):
        x =  preprocessing.scale(x);
    return x,y