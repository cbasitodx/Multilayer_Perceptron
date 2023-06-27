import numpy as np

'''
Auxiliar class that models a function
'''
class Function:
    def __init__(self, f : callable, df : callable):
        
        '''
        @params:
        * f  : **VECTORIAL** expression of the function
        * df : **VECTORIAL** expression of the derivative of the function 
        '''
        self.function = f
        self.derivative_function = df
    
    def f(self, x : np.ndarray):
        '''
        returns the call to f given an input x 
        ''' 
        return self.function(x)
    
    def df(self, x : np.ndarray): 
        '''
        returns the call to df given an input x
        '''
        return self.derivative_function(x)