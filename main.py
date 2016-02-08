#!/usr/bin/python
import numpy as np
import numpy.linalg as la
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def readData(filename, delim):
    data = np.genfromtxt(filename, delimiter=delim)
    x = data[:, np.newaxis, 0]
    y = data[:, np.newaxis, 1]
    return (x,y)

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

def main():

    data_X, data_Y = readData("data/svar-set1.dat", " ")

    #plot original data
    #plt.scatter(data_X, data_Y,  color='black')
    #plt.show()

    #add ones to first column in X
    z = createZMatrix(data_X)

    # Split the data into training/testing sets
    data_X_train, data_Y_train, data_X_test, data_Y_test = splitDataSet(0.9,z,data_Y)
    '''data_X_train = z[:-20]
    data_X_test = z[-20:]

    # Split the targets into training/testing sets
    data_Y_train = data_Y[:-20]
    data_Y_test = data_Y[-20:]'''

    #calculate thetas method 1
    thetas = np.dot(np.linalg.pinv(data_X_train),data_Y_train)
    print("Method 1 Coefficients: {}\n".format(thetas))

    #calculate thetas method 2
    thetas = np.dot(np.dot(la.inv(np.dot(np.transpose(data_X_train),data_X_train)), np.transpose(data_X_train)),data_Y_train)
    print("Method 2 Coefficients: {}\n".format(thetas))

    #calculate thetas method 3
    regr = linear_model.LinearRegression()
    regr.fit(data_X[:-20], data_Y_train)
    print("Ready Made method Coefficients: {} Intercept: {}\n".format(regr.coef_, regr.intercept_))

    print("Training Mean Squared Error: {}\n".format(getMeanError(thetas,data_X_train,data_Y_train)))
    print("Ready Made Training Mean Squared Error: {}\n".format(np.mean((regr.predict(data_X[:-20]) - data_Y_train) ** 2)))

    print("Testing Mean Squared Error: {}\n".format(getMeanError(thetas,data_X_test,data_Y_test)))
    print("Ready Made Testing Mean Squared Error: {}\n".format(np.mean((regr.predict(data_X[-20:]) - data_Y_test) ** 2)))


######################################################################
    # Create linear regression object
'''    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(mydata_X_train, mydata_y_train)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(mydata_X_test) - mydata_y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(mydata_X_test, mydata_y_test))

    # Plot outputs
    plt.scatter(mydata_X_train, mydata_y_train,  color='black')
    plt.plot(mydata_X_train, regr.predict(mydata_X_train), color='blue',
         linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

'''

if __name__ == "__main__":
    main()