import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets.samples_generator import make_blobs


X, Y = make_blobs(n_samples=70, centers=2, random_state=0, cluster_std=0.7)


print(X)
print(Y)