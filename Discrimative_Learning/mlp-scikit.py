from sklearn.neural_network import MLPClassifier
from data_utils import *
import numpy as np
from multilayer import *
from sklearn import metrics

x, labels = readData("Data/iris.data", ",", scale=False)

# shuffle
p = np.random.permutation(len(x))
x = x[p]
labels = labels[p]
classes, y = np.unique(labels, return_inverse=True)

#fit built-in mlp
clf = MLPClassifier(hidden_layer_sizes=(4,))
fit = clf.fit(x, y)
for coef in clf.coefs_:
    print (coef)

predicted_labels = classify_all(x,x,y,maxIterations=900, h=4,  learning_rate=0.001, learning_rate_v=0.002)


predicted_labels_2 = list()
for i in range(x.shape[0]):
    predicted_labels_2.append(clf.predict(x[i,:]))

print(metrics.accuracy_score(y, predicted_labels))
print(metrics.accuracy_score(y, predicted_labels_2))

