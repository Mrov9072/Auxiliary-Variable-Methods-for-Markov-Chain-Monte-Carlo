import torch
import numpy as np
from proposal import Proposal
from torch import distributions as dist

class RandomWalk(Proposal):
    """Proposal of the form q(x' | x) = N(x' | x, \sigma^2)"""
    
    def __init__(self, sigma):
        self.sigma = sigma
        self.d = dist.Normal(loc=0., scale=sigma)
        
    def sample(self, x):
        return x + self.d.sample(sample_shape=x.shape)
    
    def log_density(self, x, x_prime):
        return self.d.log_prob(x_prime - x).sum()
    
    def __str__(self):
        return f"Random walk sigma={self.sigma}"
    
class Langevin(Proposal):
    """Proposal given by q(x' | x) = N(x' | x - 0.5 * eps * \nabla \log \pi(x), eps)"""
    def __init__(self, eps, d):
        self.d = dist.Normal(loc=0., scale=np.sqrt(eps))
        self.dist = d
        self.eps = eps
        
    def sample(self, x):
        return x - 0.5 * self.eps * self.dist.grad_log_density(x) + self.d.sample(sample_shape=x.shape)       
    
    def log_density(self, x, x_prime):
        xn = x - 0.5 * self.eps * self.dist.grad_log_density(x)
        return self.d.log_prob(x_prime - xn).sum(dim=-1)
    
    def __str__(self):
        return f"Langevin eps={self.eps}"

