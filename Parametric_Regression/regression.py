#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn.cross_validation import KFold
from sklearn import preprocessing
import random
from scipy.spatial.distance import pdist, squareform
import scipy

#predict an example x using the thetas coef
def predict(coef,x):
    return np.dot(x,coef)

#get the sum of squared errors
def getError(coef, z, y):
    aux = np.transpose(np.dot(z,coef) - y)
    return np.dot(aux,np.dot(z,coef) - y)

#get the mean squared error
def getMeanError(coef, z, y):
    return getError(coef,z,y) / len(z)

#add ones to first column
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

#fit the model using explicit solution
def fit_model(data_X_train, data_Y_train):
    return np.dot(np.linalg.pinv(data_X_train),data_Y_train)

#evaluate using k-fold cross validation
def kfold_validation(data_x,data_y, k, function=fit_model):
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

#evalutae k-fold cross validation for the gradient descent algorithm
def kfold_validation_gradient_descent(data_x,data_y, k, lw,threshold=0.00001):
    kf = KFold(len(data_x), n_folds=k)
    all_errors = list()
    for train_index, test_index in kf:
        x_train = data_x[train_index]
        y_train = data_y[train_index]
        x_test = data_x[test_index]
        y_test = data_y[test_index]
        thetas,iterations,errors = gradient_descent(x_train, y_train, learning_weight=lw,threshold=threshold)
        all_errors.append(getMeanError(thetas,x_test,y_test))
    return np.mean(all_errors)

#evalutae k-fold cross validation for the dual regression
def kfold_validation_dual(data_x,data_y, k, s):
    kf = KFold(len(data_x), n_folds=k)
    all_errors = list()
    for train_index, test_index in kf:
        x_train = data_x[train_index]
        y_train = data_y[train_index]
        x_test = data_x[test_index]
        y_test = data_y[test_index]
        alphas = solveDual(x_train,y_train,s)
        all_errors.append(getMeanErrorDual(alphas,x_train,x_test,y_test,s))
    return np.mean(all_errors)

#map matrix to a higher polynomial degree
def mapFeatures(x, degree):
    poly = preprocessing.PolynomialFeatures(degree)
    return poly.fit_transform(x)

#gradient descent algorithm
def gradient_descent(x,y, threshold=0.000001, maxIterations=100000, delta=9999, learning_weight=0.0000001 ):
    #iterative solution
    iterations = 0
    thetas = []
    #random start
    for i in range(len(x[0])):
        thetas.append(random.random())
    all_errors = np.empty([0,2])
    all_errors = np.append(all_errors,[[0,getMeanError(thetas,x,y)]],axis=0)
    while (delta > threshold and iterations < maxIterations):
        predictions = np.dot(x,thetas)
        errors = predictions - y
        squared_errors = np.dot(np.transpose(x), errors)
        aux1 = np.dot(learning_weight,squared_errors)
        new_thetas = thetas - aux1
        delta = getMeanError(thetas,x,y) - getMeanError(new_thetas,x,y)
        iterations += 1
        all_errors = np.append(all_errors,[[iterations,getMeanError(new_thetas,x,y)]],axis=0)
        thetas = new_thetas

    return (thetas,iterations,all_errors)

def getGramMatrix(x):
    return squareform(pdist(x, 'euclidean'))

def getGaussianGramMatrix(x,s):
    gram = getGramMatrix(x)
    return scipy.exp(-(gram**2) / (2*(s**2)))

#solve dual regression with gaussian kernel function
def solveDual(z, y, s):
    alfas = alphas = np.dot(la.inv(getGaussianGramMatrix(z,s)),y)
    return alphas

#predict an example for dual regression
def predictDual(z,alphas,example,s):
    sum = 0
    for i in range(0,len(alphas)):
        values = np.array([example,z[i]])
        dist = pdist(values,'euclidean')
        gaussian = scipy.exp(-(dist**2)/(2*(s**2)))
        sum += alphas[i] * gaussian
    return sum


#get the sum of squared errors for dual regression
def getErrorDual(alphas, x, z, y,s):
    sum = 0
    for i in range(0,len(z)):
        aux = predictDual(x,alphas,z[i],s)
        sum += (aux - y[i])**2
    return sum

#get the mean squared error for dual regression
def getMeanErrorDual(alphas, x, z, y,s):
    return getErrorDual(alphas,x, z,y,s) / len(z)