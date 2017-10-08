#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 22:27:46 2017

@author: samkirkiles
"""

import math;
import numpy as np;
import scipy.io as sio
import matplotlib.pyplot as plt

contents = sio.loadmat('ex3data1.mat');
X = np.asarray(contents['X']);
y = contents['y'];


def drawImg(input_matrix):
    #visualize the data
    #expect 400 features
    
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


def main():
    print("Starting multi-class classification");
    random = np.random.randint(5000, size=64)
    drawImg(X[random,:]);
    
    
if __name__ == "__main__": main()
