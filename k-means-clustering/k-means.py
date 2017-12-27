# -*- coding: utf-8 -*-
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt;
import numpy as np;

# number of clusters
k = 2

#We don't need the y because we are using unsupervised learning
X,y = make_blobs(n_samples=100,n_features=2,centers=k,cluster_std=2,random_state=10)

c = X[np.random.randint(X.shape[0],size=k)]
m = X.shape[0]

color_sequence = ['g','r']

X_out = np.zeros((m,k))
y_out = np.empty((m, k), dtype=bool)

plt.title('Initial Centroids')
plt.scatter(X[:,0:1],X[:,1:2])
plt.scatter(c[:,0:1],c[:,1:2], c=color_sequence, s=150)
plt.show();

def cost_function(X,idx,ce):
    cost = 0
    for j in range(0,k):
        cost += np.sum(np.sqrt(np.sum((X[idx[:,j]] - ce[j,:])**2,axis=1)),axis=0)

            
    return cost/m

cost = np.zeros((100,1))
plot = 0;

for i in range(0,1000):
    
    tempc = c
    
    for j in range(0,k):
        X_out[:,j] = np.sqrt(np.sum(((X[:,:]-c[j,:])**2),axis=1))
        
    y_out[:,0] = np.asarray([X_out[:,0]>X_out[:,1]])        
    y_out[:,1] = np.asarray([X_out[:,1]>X_out[:,0]])
    
    for j in range(0,k):
        tempc[j,:] = np.mean(X[y_out[:,j]][:,0:2],axis=0)
    
    c = tempc
    
    # Information for cost graph not important
    if i < 100:
        cost[plot] = cost_function(X,y_out,c)
        plot += 1;

    
# set centroids to average of clusters
    
# If any blue shows up, there are unlabeled points

# Plot result
plt.title('Final Clusters')
plt.scatter(X[:,0:1],X[:,1:2],c='b')

for i in range(0,k):
    plt.scatter(X[y_out[:,i]][:,0:1],X[y_out[:,i]][:,1:2], c=color_sequence[i])

plt.scatter(c[:,0:1],c[:,1:2], c=color_sequence, s=150)

plt.show()

# Plot cost
plt.subplot()
plt.plot(cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()