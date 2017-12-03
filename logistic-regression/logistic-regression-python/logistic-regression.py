# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize

#load in the data from the textfile
data = np.genfromtxt('data.txt', delimiter=",")

#import the features and 
X = data[:,0:2]

y = data[:,2:3]

#This function will create more features using the ones we already have to create
#a good polynomial that fits the function well
def mapFeature(x1, x2, degree):
    #we want to create more polynomial features up to degree 6
    
    features = np.ones( (x1.shape[0], 1) )
    
    for i in range(1, degree+1):
        for j in range(0, i+1):
            x1_temp = np.power(x1, i - j)
            x2_temp = np.power(x2, j)
            term = (x1_temp * x2_temp).reshape(x1.shape[0],1)
            features = np.append(features, term, axis=1)

    return features
    

def costFunction(theta, X_mapped, y, _lambda):
    m = X_mapped.shape[0]
    h = sigmoid(X_mapped.dot(theta))
    theta = np.reshape(theta, [28,1])    

    term1 = ((np.transpose(-y).dot(np.log(h))))
    term2 = np.transpose((1 - y)).dot(np.log(1-h))
    regterm = ((_lambda/(2 * m)) * np.sum(np.power(theta[1:,:], 2)))
    J = (term1 - term2) * 1/m + regterm # prevents overfitting
    return float(J)    

def sigmoid(z):
    return 1 /( 1 + np.power(math.e, z))

def scatter(data):
    plt.scatter(data[:,1][np.where(data[:,2] == 1.0)], data[:,0][np.where(data[:,2] == 1.0)], alpha=1, color="b", label="y = 1", marker="o")
    plt.scatter(data[:,1][np.where(data[:,2] == 0.0)], data[:,0][np.where(data[:,2] == 0.0)], alpha=1, color="k", label="y = 0", marker="x")
    plt.legend();
    plt.show()


def graphBoundary(data, final_theta):
    plt.scatter(data[:,1][np.where(data[:,2] == 1.0)], data[:,0][np.where(data[:,2] == 1.0)], alpha=1, color="b", label="y = 1", marker="o")
    plt.scatter(data[:,1][np.where(data[:,2] == 0.0)], data[:,0][np.where(data[:,2] == 0.0)], alpha=1, color="k", label="y = 0", marker="x")
    
    x_contour = np.linspace(-1,1.5,50)
    y_contour = np.linspace(-1,1.5,50)
    
    
    z = np.zeros((len(x_contour),len(y_contour)))
    
    for i in range(len(x_contour)):
        for j in range(len(y_contour)):
            test_features = mapFeature(np.array([x_contour[i]]),np.array([y_contour[j]]), 6)
            z[i][j] = np.dot(final_theta.T, test_features.T)
        
    plt.contour(x_contour, y_contour, z,[0])
    
    plt.legend();
    plt.show()

def main():
    print("Starting Logistic Classification")
    
    scatter(data)

    features = mapFeature(X[:,0:1], X[:,1:2],6);

    #initial theta
    theta = np.zeros([features.shape[1],1])
    print("initial: ", costFunction(theta, features, y, 1))
    
    #minimize the cost function using scipy minimize
    res = minimize(costFunction, theta, args=(features,y,1), options={"maxiter":100, "disp":False})
    
    print("The minimized values of theta are:")
    print(res['x']);
    
    final_theta = np.reshape(res['x'], [28,1])
    #calculate the predicted values
    p = sigmoid(np.transpose(final_theta).dot(np.transpose(features))) >= 0.5
    p = np.transpose(p)

    print('Final Accuracy: ', np.mean(p == y) * 100)
    
    graphBoundary(data, final_theta)
    
if __name__ == "__main__": main()
