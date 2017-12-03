#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 22:27:46 2017

@author: samkirkiles
"""

import math;
import numpy as np;
import scipy.io as sio;
import matplotlib.pyplot as plt;
from scipy.optimize import minimize;

contents = sio.loadmat('ex3data1.mat');
X = np.asarray(contents['X']);
y = contents['y'];

def drawImg(input_matrix):
    #visualize the data
    #expect 400 features or perfect square
    
    plt.close();
    plt.set_cmap("gray");

    
    [m, n] = input_matrix.shape
    
    #the number of units in one dimension of the entire output
    units = math.floor(math.sqrt(m));
    #the length of one unit in pixels
    l = math.floor(math.sqrt(n));
    #array of zeroes with the final dimensions which the grayscale features will be loaded into
    output = np.zeros((l * units, l * units));
        
    i = 0;
    
    #iterate over every 20 units of the array and place in the feature from the input matrix
    #reshape and transpose each feature to get it from vector form to a form we can view
    for x in range(0,units):
        for y in range(0,units):
            output[x*l:(x*l)+l, y*l:(y*l)+l] = input_matrix[i,:].reshape(l,l).T;
            i += 1;
    
    #display the plot
    plt.imshow(output);
    plt.show()

def testModel(computed_theta, X_test):
    # take the theta and X and compute predictions for X
    
    
    global predictions
    predictions = predictOneVsAll(computed_theta, X_test)
    predictions = np.argmax(predictions, axis=1) + 1
    predictions = np.reshape(predictions, (predictions.shape[0],1));

    
    for x in range(0, X_test.shape[0]):   
        print("Predicted Value: ", int(predictions[x]))

        plt.close();
        plt.set_cmap("gray");
        plt.imshow(X_test[x,:].reshape(20,20).T);
        plt.show()



    
def sigmoid(z):
    # z should be h(X) = X * theata
    return 1 /( 1 + np.exp(-z))

def costFunction(theta, X_term, y_term, _lambda):
    
    theta = np.reshape(theta, (400,1))

    m = y_term.shape[0]
    h = sigmoid(X_term.dot(theta))
    
    term1 = (y_term * -1).T.dot(np.log(h));
    term2 = (1.-y_term).T.dot(np.log(1.-h));
    cost = (float(term1)-float(term2))/m
    regterm = (_lambda/(2 * m)) * np.sum(theta ** 2)
    J = cost + regterm
    
    #calculate the gradient
    
    grad = (X_term.T.dot((h - y_term))/m);
    
    grad = np.add(grad, ((_lambda/m) * theta));    
    grad = grad.flatten()

    return float(J), grad

    
def oneVsAll(_X,_y,num_labels,_theta,_lambda):
    
    n = _X.shape[1];

    initial_theta = np.zeros([n,1])
    
    initial_theta = np.ndarray.flatten(initial_theta)
    
    all_theta = np.zeros((num_labels,n));

    
    for c in range(1,num_labels + 1):
        
        print()
        print("Optimizing for " , c)
                
        args = (_X, (_y == c), 3) 
        res = minimize(costFunction, x0=initial_theta, jac=True, args=args, method='BFGS', options={'maxiter':1000});
        print(res);
        all_theta[c - 1,:] = res['x'];
        
    return all_theta;
    
def predictOneVsAll(all_theta, X):
    
    predictions = sigmoid(X.dot(all_theta.T))
    
    return predictions


def main():
    print("Starting multi-class classification");
    print();
    print("Drawing random training examples...")
    
    # show the a visualization of the data
    random = np.random.randint(5000, size=100)
    drawImg(X[random,:]);
    
    # find intial cost function
    all_theta = np.zeros((X.shape[1], 1))
    
    cost, grad = costFunction(all_theta, X, y, 3)
    print("Testing cost function with initial cost of: ", cost )
    
    computed_theta = oneVsAll(X, y, 10, all_theta, 3);
        
    predictions = np.argmax(predictOneVsAll(computed_theta, X), axis=1) + 1
    
    predictions = np.reshape(predictions, (5000,1))
    
    
    accuracy = (np.sum(predictions == y)/y.size) * 100
    
    print();
    print("Successfully trained model with ", accuracy, "% accuracy.")
    print();
    print("Testing theta with random training example")
    
    X_test = X[np.random.randint(5000,size = 3),:];
    
    testModel(computed_theta, X_test);
    
    
    
if __name__ == "__main__": main()