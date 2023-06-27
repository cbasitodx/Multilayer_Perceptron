import numpy as np

class Function:
    def __init__(self, f : callable, df : callable):
        self.function = f
        self.derivative_function = df
    
    def f(self, x : np.ndarray): 
        return self.function(x)
    
    def df(self, x : np.ndarray): 
        return self.derivative_function(x)