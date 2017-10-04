#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# This is the old data that didn't really have a linear correlation
'''
d = pd.read_csv('wine-quality.csv')
d = d[d.columns[:]].iloc[:].sort_values(by='fixed acidity', axis=0, ascending=True, inplace=False)
d['density'] = d['density'].map(lambda x: 100*x);

x_train = d['fixed acidity'].head(300)
y_train = d['density'].head(300)
'''     

d = pd.read_csv('data.csv').values

ones = np.ones((len(d),1))
x_train = d[:,0:1]
x_train = np.concatenate((ones,x_train),1)

y_train = d[:,1:2]

def computeError(theta, x_data, y_data):
    
    h = x_data.dot(theta) - y_data
    return (np.transpose(h).dot(h)) / (2 * len(x_data))
    
def costFunction(theta, y_train, x_train, learning_rate, iterations):
        
    for i in range(0, iterations):
        theta = theta - (learning_rate/iterations) * np.transpose(x_train).dot(x_train.dot(theta) - y_train)
        
    return theta

def graph(formula, x_range):  
    x = np.array(x_range)
    y = formula(x)

    plt.plot(x,y)  

def graphResult(value0, value1, graph_range):
    plt.title('Linear Regression')
    plt.xlabel('X Training Data')
    plt.ylabel('Y Training Data')
    plt.scatter(x_train[:,1:2],y_train)
    graph(lambda x : value0 + value1 * x, graph_range)
    plt.show();
    
def main():
    
    start = time.time()
    
    theta = np.zeros((2,1))
    learning_rate = 0.0001
    iterations = 1000

    print("Starting gradient descent linear regression with theta0: " + str(theta[0,0]) + " theta1: " + str(theta[1,0]))
    print("Working...")
    
    #for each number in iterations run the cost function
    
    theta = costFunction(theta, y_train, x_train, learning_rate, iterations)
    
    graphResult(theta[0,0], theta[1,0],range(int(x_train[:,1:2].min()), int(x_train[:,1:2].max())))
    J = computeError(theta,x_train,y_train)
    
    end = time.time()
    
    print("Finsihed with theta0: " + str(theta[0,0]) + " theta1: " + str(theta[1,0]) + " iterations: " + str(iterations) + " time elapsed: " + str(round(end-start, 3)) + "s" + " error: " + str(J))

if __name__ == "__main__":
    main()
    




