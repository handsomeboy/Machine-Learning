from sklearn.datasets import fetch_mldata
import numpy as np
import os
from metrics import *
from pylab import *
from logistic_regression_kclass import *

mnist = fetch_mldata('MNIST original')
print(mnist.data.shape)
print(mnist.target.shape)
print(np.unique(mnist.target))

X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:40000], X[40000:]
y_train, y_test = y[:40000], y[40000:]

#use 0.000005
print("Training accuracy: {}".format(getAccuracy(y_train,classify_all(X_train,X_train,y_train),1)))

print(X_train.shape)
print(y_train.shape)
size=len(y_train)

## extract "3" digits and show their average"
ind = [ k for k in range(size) if y_train[k]==3 ]
extracted_images=X_train[ind,:]

mean_image=extracted_images.mean(axis=0)
imshow(mean_image.reshape(28,28), cmap=cm.gray)
show()