# -*- coding: utf-8 -*-

# This was a pretty garbage nerual network to be honest, I messed up many times but now get it (sort of) I will come back 
# to clean up the code

import numpy as np;

def sigmoid(z):
    return 1 /( 1 + np.exp(-z))

def sigmoidGradient(z):
    s = sigmoid(z);
    return s * (1 - s);

def randInitializeWeights(size_in, size_out):
    
    # Create random initializatinos for the weights for given dimenson
    # Values are clamped between (-epsilon and epsiolon)
    W = 2 * np.random.rand(size_out, 1 + size_in) - 1;
    return W

def costFunction(input_layer_size, hidden_layer_size, X_term, y_term, _lambda):
    J = 0
    rand_seed = 10;
    
    m = X_term.shape[0];
    
    # X_term shape is (4,3)
    # Theta1 shape is (3(hidden layer),3(input layer))
    
    # Theta1 = (hidden_layer_size,input_layer_size) three hidden units
    np.random.seed(rand_seed);
    theta1 = randInitializeWeights(hidden_layer_size,input_layer_size);
    # Add bias unit to theta1
    #theta1 = np.concatenate(( np.ones((theta1.shape[0],1)) , theta1), axis = 1);


    #Theta2 = (output_layer_size,hidden_layer_size) Multiply by a1 to get output
    np.random.seed(rand_seed * 1343)
    theta2 = randInitializeWeights(hidden_layer_size,1);
    # Add bias unit to theta2
    #theta2 = np.concatenate(( np.ones((theta2.shape[0],1)) , theta2), axis = 1);
    
    print()
    print("Printing cost every 1000 iterations:")
    
    for i in range(0,10000):
        
        # Forward propagation
        # Add bias unit to input layer
        
        # l0
        #a1 = np.concatenate(( np.ones((X_term.shape[0],1)) , X_term), axis = 1);
        
        a1 = X_term
        
        #l1
        z1 = a1.dot(theta1); # (4,3)
        #a2 = np.concatenate(( np.ones((sigmoid(z1).shape[0],1)) , sigmoid(z1)), axis = 1);
        
        a2 = sigmoid(z1)
        
        # l2
        z2 = a2.dot(theta2.T); # (4,1)
        a3 = sigmoid(z2);
        # output size is (4 (training examples),1 (number of output classes or output neurons))
        
        # Simple squared error cost function
        J = sum((a3-y) ** 2)/2
        
        if (i % 1000) == 0:
            print(J)

        
        # Backpropagation to find the gradient of the cost function with repsect to theta
        error = (a3-y) # 4,1
        
        # not at all sure if this is correct but it *should* be the correct gradient for the first layer
        delta2 = error*sigmoidGradient(a3);
        
        delta1_error = delta2.dot(theta2)
        
        delta1 = delta1_error * sigmoidGradient(a2);

        
        theta2 -= a2.dot(delta2).T;
        theta1 -= a1.T.dot(delta1);
        
    
    return J, a3
    
def main():
    print("Starting NN Classifier")
    
    input_layer_size = 3;
    hidden_layer_size = 3;
    num_labels = 1;
    
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
    
    J, out = costFunction(3, 3, X, y, 0);  
    
    print()
    print("Predictions:")
    print(np.around(out))
    
    
if __name__ == "__main__": main()