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

d = pd.read_csv('data.csv')
x_train = d[d.columns[0]]
y_train = d[d.columns[1]]


J = None

def computeError(intercept, slope, x_data, y_data):

    y_pred = (intercept + (slope * x_train))
    return ((y_pred - y_data) ** 2).sum()/(len(x_train))

    
def costFunction(intercept, slope, y_train, x_train, learning_rate):
    
    #go through each value 
    sum0 = 0
    sum1 = 0
    count = len(x_train)

        
    y_pred = (intercept + (slope * x_train))
    sum0 += ((y_pred - y_train)/count).sum()
    sum1 += (((y_pred - y_train) * x_train)/count).sum()
        
    temp0 = intercept - (learning_rate * sum0)
    temp1 = slope - (learning_rate * sum1)
        
    return temp0, temp1


def graph(formula, x_range):  
    x = np.array(x_range)
    y = formula(x)

    plt.plot(x,y)  

    
def graphResult(value0, value1, graph_range):
    plt.title('Linear Regression')
    plt.xlabel('X Training Data')
    plt.ylabel('Y Training Data')
    plt.scatter(x_train,y_train)
    graph(lambda x : value0 + value1 * x, graph_range)
    plt.show();


def main():
    
    
    start = time.time()
    
    value0 = 0;
    value1 = 0;
    learning_rate = 0.0001
    iterations = 1000
    
    print("Starting gradient descent linear regression with theta0: " + str(value0) + " theta1: " + str(value1))
    print("Working...")
    
    #for each number in iterations run the cost function
    for i in range(0, iterations):
        value0, value1 = costFunction(value0, value1, y_train, x_train, learning_rate)
    
    graphResult(value0, value1,range(int(x_train.min()), int(x_train.max())))
    J = computeError(value0,value1,x_train,y_train)
    
    end = time.time()
    
    print("Finsihed with theta0: " + str(value0) + " theta1: " + str(value1) + " iterations: " + str(iterations) + " time elapsed: " + str(round(end-start, 3)) + "s" + " error: " + str(J))

if __name__ == "__main__":
    main()
    




