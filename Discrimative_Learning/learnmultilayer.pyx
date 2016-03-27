import numpy as np
#from scipy.spatial.distance import pdist
#import numpy as np
from copy import deepcopy
#from multilayer import *

from math import exp
#import numpy.linalg as linalg
#from sklearn import metrics
#from sklearn.cross_validation import KFold
#import random
#from metrics import *
from math import log

def indicator(condition):
    if(condition):
        return 1
    else:
        return 0

def loglikelihood(x,y, ybar, labels):
    sum = 0
    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]
        sum2 = 0
        for k in range(len(labels)):
            ind = indicator(k == yi)
            sum2 += (ind * log(ybar[i,k]))
        sum += sum2
    return -sum

def getSoftmaxDen(thetas, x, labels):
    den = 0
    for k in range(len(labels)):
        den += exp(np.dot(np.transpose(thetas[k]),x))
    return den

def sigmoid(thetas, x):
    return 1 / (1 + np.exp(-np.dot(np.transpose(thetas),x)))

def softmax(thetas, x, j, labels, softmaxDen):
    numerator = exp(np.dot(np.transpose(thetas[j]),x))
    # denominator = np.sum(exp(np.dot(np.transpose(thetas),x)))

    return numerator / softmaxDen

def gradient_descent(x,y, labels, h, threshold=0.001, maxIterations=900, delta=9999, learning_rate=0.001, iv=None,iw=None):
    #iterative solution
    cdef iterations = 0
    cdef k = len(labels)
    cdef n = x.shape[1]
    cdef m = x.shape[0]
    if(iv==None):
        v = np.random.rand(k,h)/10
        w = np.random.rand(h,n)/10
    else:
        v = deepcopy(iv)
        w = deepcopy(iw)
    z = np.empty([m,h])
    ybar = np.empty([m,k])
    delta = 9999
    logl = 0
    all_likelihoods = np.empty([0,2])
    z = np.transpose(sigmoid(np.transpose(w),np.transpose(x)))
    #computer ybar
    for i in range(m):
        for j in range(k):
            ybar[i,j] = softmax(v,z[i],j,labels,getSoftmaxDen(v,z[i],labels))
    newloglikelihood = loglikelihood(x,y,ybar,labels)
    print(newloglikelihood)
    delta = newloglikelihood - logl
    all_likelihoods = np.append(all_likelihoods,[[0,loglikelihood(x,y,ybar,labels)]],axis=0)
    while (iterations < maxIterations and delta > threshold):
        #compute z
        # for i in range(m):
        #     for j in range(h):
        #         z[i,j] = sigmoid(w[j],x[i])
        z = np.transpose(sigmoid(np.transpose(w),np.transpose(x)))
        #computer ybar
        for i in range(m):
            for j in range(k):
                ybar[i,j] = softmax(v,z[i],j,labels,getSoftmaxDen(v,z[i],labels))
        newloglikelihood = loglikelihood(x,y,ybar,labels)
        print(iterations,newloglikelihood)
        delta = newloglikelihood - logl
        #update v
        for j in range(k):
            sum = 0
            for i in range(m):
                sum += ((ybar[i,j] - indicator(y[i]==j)) * z[i])
            v[j] -= 0.0001 * sum
        #update w
        for j in range(h):
            sum = 0
            for i in range(m):
                sum2 = 0
                for l in range(k):
                    sum2 += ((ybar[i,l] - indicator(y[i]==l)) * v[l,j])
                sum += sum2 * z[i,j] * (1-z[i,j]) * x[i]
            w[j] -= learning_rate * sum
        iterations += 1
        all_likelihoods = np.append(all_likelihoods,[[iterations,loglikelihood(x,y,ybar,labels)]],axis=0)
    return v,w, iterations, all_likelihoods