# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# This will create a matrix with positions and RGB (x1,x2,x3) 
# Features will be x, y, r, g, b
# Each pixel will be a shape
img = mpimg.imread('panda.png')[:,:,:3];
plt.imshow(img)
plt.show()

iterations = 1000
k = 16
x = img.shape[0]
y = img.shape[1]
n = img.shape[2]
m = x * y
img = img.reshape(m,img.shape[2])

centroids = img[np.random.randint(m,size=k)]

X_dist=np.zeros((m,k))
clusters = np.zeros(m)

def cost_function(X,idx,ce):
    cost = 0
    for j in range(0,k):
        cost += np.sum(np.sqrt(np.sum((X[np.ravel(np.argwhere(idx==j)),:] - ce[j,:])**2,axis=1)),axis=0)
    return cost/m

cost = np.zeros((10,1))
plot = 0;

# Training
for i in range(0,iterations):

    for j in range(0,k):
        # Find the distance between the current points and each centroid
        X_dist[:,j] = np.sqrt(np.sum((img[:,:] - centroids[j,:])**2,axis=1))
    
    # Sort the clusters by distance to the centroids
    clusters = np.argmin(X_dist,axis=1)
    
    tempc = centroids

    for j in range(0,k):
       tempc[j,:] = np.mean(img[np.ravel(np.argwhere(clusters==j)),:],axis=0)
    
    centroids = tempc
    
    if i%100 == 0:
        cost[plot] = cost_function(img,clusters,centroids)
        plot += 1;
        

# Assign each pixel to its nearest centroid


img_out = np.zeros(img.shape)

for j in range(0,k):
    # Find the distance between the current points and each centroid
    X_dist[:,j] = np.sqrt(np.sum((img[:,:] - centroids[j,:])**2,axis=1))

# Sort the clusters by distance to the centroids
clusters = np.argmin(X_dist,axis=1)

for j in range(0,k):
   img_out[np.ravel(np.argwhere(clusters==j))] = centroids[j,:]

plt.imshow(img_out.reshape(x,y,n ))
plt.show()


plt.subplot()
iterations = np.array([100,200,300,400,500,600,700,800,900,1000]);
plt.plot(iterations,cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()