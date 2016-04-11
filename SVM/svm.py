import numpy as np
import cvxopt
from cvxopt import solvers, matrix
import operator
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
import scipy
from sklearn import preprocessing, metrics
from sklearn.cross_validation import KFold
import metrics as svmmetrics

def getDistanceGramMatrix(X):
    return squareform(pdist(X, 'euclidean'))

def getGaussianGramMatrix(X,s):
    gram = getDistanceGramMatrix(X)
    return scipy.exp(-(gram**2) / (2*(s**2)))

def getPolynomialGramMatrix(X, degree):
    return metrics.pairwise.polynomial_kernel(X,degree=degree)

def getGramMatrix(X,type=None):
    if(type == None):
        return np.dot(X, X.T)
    elif (type == 'gaussian'):
        return getGaussianGramMatrix(X,2)
    elif (type == 'polynomial'):
        return getPolynomialGramMatrix(X, 4)

def train(X,y,c=9999, eps=0.1, type=None):
    m = X.shape[0]
    n = X.shape[1]
    #p
    gm = getGramMatrix(X,type)

    P = np.dot(y,y.T) * gm
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
    # solvers.options['maxiters'] = 1000
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
    if(w0 != 0):
        w0 = w0/len(support_vectors_idx)

    return w,w0, support_vectors_idx

def classify(x,w,w0, type=None):
    # if(type == 'gaussian'):
    #     sum = 0
    #     for i in range(len(w)):
    #         w[i]
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

def kfoldCrossValidation(x,labels,k, positive_class, c, eps, type=None):
    kf = KFold(len(x), n_folds=k)
    all_metrics = list()
    for train_index, test_index in kf:
        x_train = x[train_index]
        labels_train = labels[train_index]
        x_test = x[test_index]
        labels_test = labels[test_index]
        w,w0,support_vectors_idx = train(x_train,labels_train, type=type)
        predictedLabels = classify_all(x_test,w,w0)
        accuracy = svmmetrics.getAccuracy(labels_test,predictedLabels, positive_class)
        print(accuracy)
        recall = svmmetrics.getRecall(labels_test,predictedLabels, positive_class)
        precision = svmmetrics.getPrecision(labels_test,predictedLabels, positive_class)
        tp = svmmetrics.getTP(labels_test,predictedLabels,positive_class)
        tn = svmmetrics.getTN(labels_test,predictedLabels,positive_class)
        fp = svmmetrics.getFP(labels_test,predictedLabels,positive_class)
        fn = svmmetrics.getFN(labels_test,predictedLabels,positive_class)
        fmeasure = svmmetrics.getFMeasure(labels_test,predictedLabels,positive_class)
        all_metrics.append([accuracy,recall,precision,tp,tn,fp,fn,fmeasure])
    return np.mean(all_metrics,axis=0)