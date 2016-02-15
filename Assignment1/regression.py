#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn.cross_validation import KFold
from sklearn import preprocessing
import random
from scipy.spatial.distance import pdist, squareform
import scipy

def predict(coef,x):
    return np.dot(x,coef)

def getError(coef, z, y):
    aux = np.transpose(np.dot(z,coef) - y)
    return np.dot(aux,np.dot(z,coef) - y)

def getMeanError(coef, z, y):
    return getError(coef,z,y) / len(z)

def createZMatrix(x):
    return  np.append(np.ones((len(x),1)),x,1)

#trainingProp is the proportion of data set to be assigned for training. The remaining is fot testing
def splitDataSet(trainingProp, x, y):
    training_size = len(x)*trainingProp
    m = len(x)
    training_x = x[:-(m - training_size)]
    training_y = y[:-(m - training_size)]
    testing_x = x[-(m - training_size):]
    testing_y = y[-(m - training_size):]
    return (training_x,training_y, testing_x,testing_y)

def fit_model1(data_X_train, data_Y_train):
    return np.dot(np.linalg.pinv(data_X_train),data_Y_train)

def fit_model2(data_X_train, data_Y_train):
    return np.dot(np.dot(la.inv(np.dot(np.transpose(data_X_train),data_X_train)), np.transpose(data_X_train)),data_Y_train)

def kfold_validation(data_x,data_y, k, function=fit_model1):
    kf = KFold(len(data_x), n_folds=k)
    all_errors = list()
    for train_index, test_index in kf:
        x_train = data_x[train_index]
        y_train = data_y[train_index]
        x_test = data_x[test_index]
        y_test = data_y[test_index]
        thetas = function(x_train, y_train)
        all_errors.append(getMeanError(thetas,x_test,y_test))
    return np.mean(all_errors)

def kfold_validation_gradient_descent(data_x,data_y, k, lw):
    kf = KFold(len(data_x), n_folds=k)
    all_errors = list()
    for train_index, test_index in kf:
        x_train = data_x[train_index]
        y_train = data_y[train_index]
        x_test = data_x[test_index]
        y_test = data_y[test_index]
        thetas,iterations = gradient_descent(x_train, y_train,lw)
        all_errors.append(getMeanError(thetas,x_test,y_test))
    return np.mean(all_errors)

def kfold_validation_gaussian(data_x,data_y, k, s):
    kf = KFold(len(data_x), n_folds=k)
    all_errors = list()
    for train_index, test_index in kf:
        x_train = data_x[train_index]
        y_train = data_y[train_index]
        x_test = data_x[test_index]
        y_test = data_y[test_index]
        gram_matrix = getGaussianGramMatrix(x_train,s)
        thetas = solveDual(gram_matrix, x_train, y_train)
        all_errors.append(getMeanError(thetas,x_test,y_test))
    return np.mean(all_errors)

def mapFeatures(x, degree):
    poly = preprocessing.PolynomialFeatures(degree)
    return poly.fit_transform(x)

def gradient_descent(x,y, threshold=0.0000001, maxIterations=100000, delta=9999, learning_weight=0.0000001 ):
    #iterative solution
    iterations = 0
    thetas = []
    #random start
    for i in range(len(x[0])):
        thetas.append(random.random())
    while (delta > threshold and iterations < maxIterations):
        predictions = np.dot(x,thetas)
        errors = predictions - y
        squared_errors = np.dot(np.transpose(x), errors)
        aux1 = np.dot(learning_weight,squared_errors)
        new_thetas = thetas - aux1
        delta = getMeanError(thetas,x,y) - getMeanError(new_thetas,x,y)
        iterations += 1
        thetas = new_thetas
    return (thetas,iterations)

def getGramMatrix(x):
    return squareform(pdist(x, 'euclidean'))

def getGaussianGramMatrix(x,s):
    gram = getGramMatrix(x)
    return scipy.exp(-gram**2 / 2*(s**2))

def solveDual(gram_matrix, x, y):
    alfas = np.dot(la.inv(gram_matrix),y)
    thetas = np.dot(np.transpose(x),alfas)
    return thetas