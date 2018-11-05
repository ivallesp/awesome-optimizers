import autograd.numpy as np
from autograd import grad



class Beale:
    def __init__(self):
        self.xmin, self.xmax = -1.5, 5
        self.ymin, self.ymax = -3, 1.5
        self.y_start, self.x_start = -2.8, 2.4  # Start point
        self.x_optimum, self.y_optimum, self.z_optimum = 3, 0.5, 0  # Global optimum 
        self._compute_derivatives()
        
    def _compute_derivatives(self):
        self.df_dx =grad(self.eval, 0) # Partial derivative of the objective function over x
        self.df_dy = grad(self.eval, 1) # Partial derivative of the objective function over y
        
    def eval(self, x, y):
        z = np.log(1+(1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2)/10
        return z