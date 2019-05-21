import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets.samples_generator import make_blobs

from sklearn.svm import SVC


X, Y = make_blobs(n_samples=70, centers=2, random_state=0, cluster_std=0.7)


# print(X)
# print(Y)


plt.scatter(X[:,0], X[:,1], c = Y)
plt.savefig('scatterplot.png')

svm_model = SVC(kernel = 'linear', C = 1E10)
svm_model.fit(X,Y)







