# -*- coding: utf-8 -*-

import numpy as np;

def sigmoid(z):
    return 1 /( 1 + np.exp(-z))

def sigmoidGradient(z):
    s = sigmoid(z);
    return s * (1 - s);

def randInitializeWeights(size_in, size_out):
    
    # Create random initializatinos for the weights for given dimenson
    # Values are clamped between 1 and -1
    W = 2 * np.random.rand(size_out, 1 + size_in) - 1;
    return W

def costFunction(input_layer_size, hidden_layer_size, X_term, y_term):
    
    J = 0
    rand_seed = 10;
    
    #using simple sum of square error cost function sum((h(x)-y)^2)/2
    
    # X_term shape is (4,3)
    # Theta1 shape is (3(hidden layer),3(input layer))
    
    # Theta1 = (hidden_layer_size,input_layer_size) three hidden units
    np.random.seed(rand_seed);
    theta1 = randInitializeWeights(hidden_layer_size,input_layer_size);

    np.random.seed(rand_seed * 20)
    theta2 = randInitializeWeights(hidden_layer_size,1);
    
    print()
    print("Printing cost every 1000 iterations:")
    
    for i in range(0,10000):
        
        # Forward propagation
        # Add bias unit to input layer
        
        a1 = X_term
        
        #l1
        z1 = a1.dot(theta1); # (4,3)        
        a2 = sigmoid(z1)
        
        # l2
        z2 = a2.dot(theta2.T); # (4,1)
        a3 = sigmoid(z2);
        
        # Simple squared error cost function
        J = sum((a3-y) ** 2)/2
        
        # Every 1000 iterations, print cost 
        # this must be decreasing
        if (i % 1000) == 0:
            print(J)

        # Backpropagation to find the gradient of the cost function with repsect to theta
        # Use chain rule to find the gradient of J w.r.t each theta and subtract from theta2
        # For a proof of the math, read https://www.coursera.org/learn/machine-learning/resources/EcbzQ
        
        error = (a3-y) # 4,1
        
        delta2 = error*sigmoidGradient(a3);
        
        delta1_error = delta2.dot(theta2)
        
        delta1 = delta1_error * sigmoidGradient(a2);

        
        theta2 -= a2.dot(delta2).T;
        theta1 -= a1.T.dot(delta1);
        
    
    return J, a3
    
def main():
    print("Starting NN Classifier")
    
    
    # Create Training set
    # This network will contain 3 features a 3 neuron hidden layer, 1 neuron output layer
    
    # Input features
    global X, y
    X = np.array([[0,1,0],
                 [1,0,1],
                 [1,1,0],
                 [0,0,1]]); # (4,3)
    # Classifications of size 4,1 
    
    y = np.array([[1],[0],[1],[0]]);
    
    input_layer_size = 3;
    hidden_layer_size = 3;

    
    J, out = costFunction(input_layer_size, hidden_layer_size, X, y);  
    
    print()
    print("Predictions:")
    print(np.around(out))
    
    
if __name__ == "__main__": main()