import torch.distributions as dist
from proposal import Proposal

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