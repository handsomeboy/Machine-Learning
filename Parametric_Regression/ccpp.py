#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model, preprocessing
import matplotlib.pyplot as plt
from data_utils import *
from regression import *
import  scipy.stats as stats

def main():

    X, data_Y = readData("data/CCPP/Folds.csv", delim=",", skipHeader=True, scale=False)
    col_mean = stats.nanmean(X,axis=0)
    inds = np.where(np.isnan(X))
    X[inds]=np.take(col_mean,inds[1])
    X = preprocessing.scale(X)

    #compare results of regression methods
    z = mapFeatures(X,1)
    thetas = fit_model(z, data_Y)
    print("Method 1 Coefficients: {}".format(thetas))
    regr = linear_model.LinearRegression()
    regr.fit(z, data_Y)
    print("Ready Made method Coefficients: {} Intercept: {}".format(regr.coef_, regr.intercept_))
    print("Mean error: {}".format(getMeanError(thetas,z,data_Y)))

    # values = np.empty([0,3])
    # for lw in [x * 0.00001 for x in range(1, 11)]:
    #     #perform gradient descent
    #     thetas, iterations = gradient_descent(z,data_Y, learning_weight=lw)
    #     print("Itearative Method Coefficients: {},Iterations:{}".format(thetas,iterations))
    #     #evaluate performance
    #     performance = kfold_validation_gradient_descent(z,data_Y,10, lw)
    #     values = np.append(values,[[lw,iterations,performance]],axis=0)
    #     print(values)
    #     np.savetxt("gradient_descent_3.csv", values, fmt='%5.5f',delimiter=",")
    # np.savetxt("gradient_descent_3.csv", values, fmt='%5.5f',delimiter=",")

    plt.clf()
    handles = np.empty([0,1])
    # for lw in [x * 0.00001 for x in range(1, 4, 2)]:
    for lw in [0.000005,0.00001, 0.000015]:
        #perform gradient descent
        thetas, iterations, errors = gradient_descent(z,data_Y, threshold=0.0001, learning_weight=lw)
        print("Itearative Method Coefficients: {},Iterations:{}".format(thetas,iterations))
        handle = plt.plot(errors[0:50,0],errors[0:50,1], linewidth = 4, label="Learning Weight = {:.6f}".format(lw))
        handles = np.append(handles,handle)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(handles=[handles[0],handles[1],handles[2]])
    plt.ticklabel_format(useOffset=False)
    plt.xlabel("Iteration")
    plt.ylabel("Squared Mean Error")
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    #     #evaluate performance
    #     performance = kfold_validation_gradient_descent(z,data_Y,10, lw)
    #     values = np.append(values,[[lw,iterations,performance]],axis=0)
    #     print(values)
    #     np.savetxt("gradient_descent_3.csv", values, fmt='%5.5f',delimiter=",")
    # np.savetxt("gradient_descent_3.csv", values, fmt='%5.5f',delimiter=",")

    #feature 5: overfitting
    # for i in range(0,4,1):
    #     data_X = X[:,np.newaxis,i]
    #     z = mapFeatures(data_X,2)
    #     thetas = fit_model1(z, data_Y)
    #     print("Method 1 Coefficients: {}\n".format(thetas))
    #     plt.scatter(data_X, data_Y,  color='black')
    #     plt.scatter(data_X, predict(thetas,z), color='blue',
    #              linewidth=3)
    #
    #     plt.show()



if __name__ == "__main__":
    main()