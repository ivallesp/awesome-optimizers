import numpy as np


def run_optimizer(opt, cost_f, iterations, *args, **kwargs):
    errors = [cost_f.eval(cost_f.x_start, cost_f.y_start)]
    xs,ys= [cost_f.x_start],[cost_f.y_start]
    for epochs in range(iterations):
        x, y= opt.step(*args, **kwargs)
        xs.append(x)
        ys.append(y)
        errors.append(cost_f.eval(x,y))
    distance = np.sqrt((np.array(xs)-cost_f.x_optimum)**2 + (np.array(ys)-cost_f.y_optimum)**2)
    return errors, distance, xs, ys

class Optimizer:
    def __init__(self, cost_f, lr, x, y, **kwargs):
        self.lr = lr
        self.cost_f = cost_f
        if x==None or y==None:
            self.x = self.cost_f.x_start
            self.y = self.cost_f.y_start
        else:
            self.x = x
            self.y = y
            
        self.__dict__.update(kwargs)
            
    def step(self, lr):
        raise NotImplementedError()