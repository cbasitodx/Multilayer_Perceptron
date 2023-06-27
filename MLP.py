import numpy as np
from Layer import Layer
from Function import Function

'''
Class that models a Multilayer Perceptron (MLP).
Each MLP holds information of (attributes):

* layers : list of layers that make up the MLP

* learning_rate : learning rate for fitting the data to the MLP

* eta           : referring to the greek letter, is a parameter used to calculate the momentum (term that is going to help us achieve faster convergence)
'''

class MultiLayerPerceptron:
    def __init__(self, *layers, learning_rate = 0.1, eta = 0.1):
        self.layers = layers
        self.learning_rate = learning_rate
        self.eta = eta
    
    '''
    forwards an input vector through the MLP, returning an output vector

    @params:
    * input_vector : vector to be forwarded

    @returns:
    * activation : activation of the last layer for the forwarded vector
    '''
    def forward(self, input_vect : np.ndarray):
        activation = self.layers[0].activation(input_vect)
        for i in range(1, len(self.layers)):
            activation = self.layers[i].activation(activation)
        return activation

    '''
    using the back propagation algorithm, we adjust the weights and biases of the net by applying the gradient descent rule on
    each one

    @params:
    * expected   : expected output of the net
    * input_vect : training example 
    '''
    def fitOne(self, expected : np.ndarray, input_vect : np.ndarray):
        
        # First, we define some constants:
        C = len(self.layers) - 1    # Number of layers (for indexing self.layers)

        delta = [None]*(C+1)        # For readability. We define a vector of delta vectors. The position 0 is unused

        a = [None]*(C+1)            # For readability. We define a vector of activations. The position 0 is unused

        da = [None]*(C+1)           # For readability. We define a vector of the derivatives of the activation function evaluated at
                                    # the weighted sum. The position 0 is unused

        # We will be considering:
        # 
        # * current_layer  : layer C-t   (t in {0,...,C-2})
        # * next_layer     : layer C-t-1
        # * previous_layer : layer C-t+1
        # * n_t            : number of neurons of the t layer
        # * y(n)           : output of the net for the n-th training example (in code: y(n) = y) 
        # * y'(n)          : f'(W_(C-1)*a_(C-1)(n) + U_C)
        # * W              : Weights matrix for the !!current layer!! (W is the matrix of weights between the current_layer and the next_layer)
        # * a[C-t]         : Activation of the C-t layer (a_(C-t)(n)) 
        # * da[C-t]        : Derivative of the activation function evaluated at the weighted sum (f'(W_(C-t)*a_(C-t)(n) + U_(C-t+1)))
        # * U              : Bias vector of the !!current layer!! (U is the vector of biases for each neuron. U stands for "umbral")
        # * momentum       : Term for correcting the weights and biases update

        # We now adjust the weights of the connection between the layer C and C-1 and the bias of the layer C

        # 1. We calculate S(n) - Y(n) (expected output for the training example 'n' - obtained output for the training example 'n'):
        output = self.forward(input_vect)
        err = expected - output # This vector should be n_C x 1

        # 2. We obtain y'(n) (and y(n)):
        current_layer = self.layers[C]
        y, dy = current_layer.getActivation() # Should return vectors of size n_C x1

        # 3. We obtain delta_C(n):
        delta[C] = -err * dy # * is multiplication component wise. delta[C] should be n_C x 1
        
        # 4. We update the weights:
        next_layer = self.layers[C-1]
        a[C-1], da[C-1] = next_layer.getActivation()

        momentum = self.eta*(current_layer.getWeights() - current_layer.getPreviousWeights())

        W = current_layer.getWeights() - (self.learning_rate * (np.outer(delta[C], a[C-1]))) + momentum # W should be n_C x n_(C-1)

        current_layer.setWeights(W) # Weight update

        #5. We update the bias:
        U = current_layer.getBias() - self.learning_rate*delta[C] # U should be n_C x 1

        current_layer.setBias(U)

        # We now adjust the rest of the weights...
        for t in range(1, C-2): # This loops go from 1 to C-2

            # 1. We update the reference for the previous, current and next layers:
            previous_layer = self.layers[C-t+1] # C
            current_layer  = self.layers[C-t]   # C-1
            next_layer     = self.layers[C-t-1] # C-2

            # 3. We obtain delta_(C-t)(n): Note that delta goes from delta_2 to delta_C (remember we are indexing starting at 1)
            delta[C-t] = np.dot(np.transpose(previous_layer.getWeights()), delta[C-t+1]) * da[C-t]  # This vector should be n_(C-t) x 1


            # 4. We update the weights:
            a[C-t-1], da[C-t-1] = next_layer.getActivation()

            momentum = self.eta*(current_layer.getWeights() - current_layer.getPreviousWeights())

            W = current_layer.getWeights() - (self.learning_rate * (np.outer(delta[C-t], a[C-t-1]))) + momentum # W should be n_(C-t) x n_(C-t-1)

            current_layer.setWeights(W)

            # 5. We update the bias
            U = current_layer.getBias() - self.learning_rate*delta[C-t] # U should be n_(C-t) x 1

            current_layer.setBias(U)

    '''
    
    '''
    def fit(self, expected_vector : np.ndarray, training_examples : np.ndarray, thresh : int, num_iters : int):
        N = len(training_examples)
        error_vect = [None]*N
        training_error = 9999
        epoch = 1

        while training_error > thresh and epoch <  num_iters:
            for i in range(N):
                input_vect = training_examples[i]
                expected = expected_vector[i]
                self.fitOne(expected, input_vect)
                error_vect[i] = (1/2) * ((np.linalg.norm(self.forward(input_vect)))**2)
            training_error = (1/N) * sum(error_vect) 
            print(f"EPOCH : {epoch} YIELDED ERROR : {training_error}")
            epoch += 1