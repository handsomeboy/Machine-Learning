import numpy as np
import cvxopt
from cvxopt import solvers, matrix
import operator
from sklearn import metrics

def train(X,y,c=9999, eps=0.1):
    m = X.shape[0]
    n = X.shape[1]
    #p
    P = np.dot(y,y.T) * np.dot(X,X.T)
    P = matrix(P, tc='d')
    #q
    q = np.empty([m, 1])
    q.fill(-1)
    q = matrix(q, tc='d')
    #G
    G = np.zeros([2*m, m])
    np.fill_diagonal(G,-1)
    np.fill_diagonal(G[m:,:],1)
    G = matrix(G)
    #h
    h = np.empty([2*m,1])
    h.fill(0)
    for i in range(m,2*m):
        h[i]=c
    h = matrix(h,tc='d')
    #A
    A = y
    A = matrix(A, tc='d').T
    #b
    b=np.zeros([1,1])
    b=matrix(b, tc='d')

    #solve for alphas
    solvers.options['maxiters'] = 1000
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.matrix(solution['x'])

    #calculat w
    w = np.zeros([1,n])
    for i in range(m):
        w += alphas[i] * y[i] * X[i,:]

    #calculate w0
    support_vectors_idx = [ k for k in range(len(alphas)) if alphas[k] > eps ]

    w0 = 0
    for i in support_vectors_idx:
        print("support vector: {}".format(i,X[i,:]))
        w0 += (y[i] - np.dot(w,X[i,:]))
    w0 = w0/len(support_vectors_idx)

    return w,w0, support_vectors_idx

def classify(x,w,w0):
    if(np.dot(w, x) + w0 > 0):
        return 1
    else:
        return -1

def classify_all(X,w,w0):
    predictedLabels = list()
    for i in range(0,X.shape[0]):
        predictedLabels.append(classify(X[i,:],w,w0))
    return predictedLabels

def getAccuracy(labels,predictedLabels):
    return metrics.accuracy_score(labels, predictedLabels)