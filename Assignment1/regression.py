#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn.cross_validation import KFold

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

def kfold_validation(data_x,data_y,k):
    kf = KFold(len(data_x), n_folds=k)
    all_errors = list()
    for train_index, test_index in kf:
        x_train = data_x[train_index]
        y_train = data_y[train_index]
        x_test = data_x[test_index]
        y_test = data_y[test_index]
        thetas = fit_model1(x_train, y_train)
        all_errors.append(getMeanError(thetas,x_test,y_test))
    return np.mean(all_errors)
