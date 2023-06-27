import numpy as np
from Function import Function

'''
Class that models a layer of the Multilayer Perceptron (MLP).
Each layers holds information of (attributes):

* current_activation                  : features vector outputted by the layer (also known as activation vector)

* weights                             : weights of the conecction between the current layer (self) and the previous layer (except for the input layer,
                                        where the weights matrix is the identity)
      
* prev_weights                        : weights of the previous iteration (for calculating the momentum term)
                                        
* bias                                : bias vector (if the layer is the input layer, then the bias vector is the zero vector)
              
* act_function                        : activation function. Is an object of the class Function. This object holds a callable for the function and its derivative
              
* input_layer                         : boolean for evaluating if the layer is an input layer
  (SHOULD NOT BE ACCESED BY THE USER)
'''

class Layer:
    '''
    @params:
    * in_features  : Dimension of the features vector received by the layer

    * out_features : Dimension of the features vector outputted by the layer

    * act_function : Activation function

    * input_layer  : If True, then the layer is an input layer
    '''
    def __init__(self, in_features : int, out_features : int, act_function : Function, input_layer : bool):
        
        k = 1/in_features
        
        if input_layer : 
            '''
            If the layer is the input layer, then we set the weights matrix to be the identity and the 
            bias vector to be the zero vector
            '''

            self.weights = np.eye(in_features); self.bias = np.zeros((in_features, 1))
        else : 
            '''
            If the layer is a hidden/output layer, then we set the weights matrix to be out_feautes x in_features dimensional,
            and the bias vector to be out_features-dimensional
            '''
            
            self.weights = np.random.uniform(low = -np.sqrt(k), high = np.sqrt(k), size = (out_features, in_features))
            self.bias = np.random.uniform(low = -np.sqrt(k), high = np.sqrt(k), size = (out_features, 1))

        self.prev_weights = np.empty((np.shape(self.weights)))

        self.current_activation = None

        self.current_activation_derivative = None

        self.act_function = act_function

        self.input_layer = input_layer

    '''
    @params:
    * prev_act : Feature vector feeded to the layer (previous activation vector)
    '''
    def activation(self, prev_act : np.ndarray):
        if(self.input_layer):
            self.current_activation = prev_act
        else:
            '''
            the expression being evaluated is:
            a_c = f(W*a_c-1 + U)
            '''
            self.current_activation            = self.act_function.f(np.dot(self.weights,prev_act) + self.bias)
            self.current_activation_derivative = self.act_function.df(np.dot(self.weights,prev_act) + self.bias)
        return self.current_activation
    
    #------------GETTERS------------#
    def getWeights(self):
        return self.weights

    def getBias(self):
        return self.bias

    def getActivation(self):
        return self.current_activation, self.current_activation_derivative 

    def getActFunction(self):
        return self.act_function
    
    def getPreviousWeights(self):
        return self.prev_weights
    #-------------------------------#

    #------------SETTERS------------#
    def setWeights(self, new_weights : np.ndarray):
        self.prev_weights = self.weights
        self.weights = new_weights
    
    def setBias(self, new_bias : np.ndarray):
        self.bias = new_bias
    #-------------------------------#