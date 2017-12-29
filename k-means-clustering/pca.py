#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 18:11:11 2017

@author: samkirkiles
"""

import matplotlib.pyplot as plt
import numpy as np

X = np.array([ 0.69, -1.31,  0.39,  0.09,  1.29,  0.49,  0.19, -0.81, -0.31, -0.71])[:,None]
y = np.array([ 0.49, -1.21,  0.99,  0.29,  1.09,  0.79, -0.31, -0.81, -0.31, -1.01])[:,None]

print("Principle Component Analysis")
print()
print('Plotting original data')
plt.subplot();
plt.scatter(X,y)
plt.show()

# Create a mxd matrix
data = np.hstack((X,y))

c = np.cov(data.T)

# Find eigenvectors/values
w,v = np.linalg.eig(c)

print('Plotting Principal components')  
plt.subplot();
plt.scatter(X,y)
plt.plot([np.mean(X),v[0,0]],[np.mean(y),v[1,0]],'o-',c='r')
plt.plot([np.mean(X),v[0,1]],[np.mean(y),v[1,1]],'o-',c='r')
plt.show()


final = data.dot(v[:,1])

# Show principal component
print('Plotting Principal Component')  
plt.scatter(final,np.zeros(X.shape))

plt.show()

print('Plotting Reconstruction')  
# Plot reconstruction of data
r = final[:,None].dot(v[:,1][:,None].T)

plt.scatter(X,y)
plt.scatter(r[:,0],r[:,1],c='g')
