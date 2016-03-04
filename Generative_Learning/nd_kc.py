from data_utils import *
import numpy as np
from generative_learning import *
from sklearn import metrics

def main():
    #read data
    x, y = readData("C:/Users/marro/Repo/CS584/Generative_Learning/Data/iris.data",",",scale=False)

    #shuffle
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]

    classes, y = np.unique(y, return_inverse=True)

    print("Confussion matrix: {}".format(getCM(y,classifyAll(x,x,y))))
    print(getRecall(y,classifyAll(x,x,y),0))
    print(getRecall(y,classifyAll(x,x,y),1))
    print(getRecall(y,classifyAll(x,x,y),2))
    cm = metrics.confusion_matrix(y,classifyAll(x,x,y))
    accuracy = metrics.accuracy_score(y,classifyAll(x,x,y))
    precision = metrics.precision_score(y,classifyAll(x,x,y))
    cm.shape
if __name__ == "__main__":
    main()