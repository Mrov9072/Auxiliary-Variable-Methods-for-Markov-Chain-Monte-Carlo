import numpy as np

class Proposal:
    """Abstract class for proposal"""
    
    def sample(self, x):
        """
            Computes vectorized sample from proposal q(x' | x)
            
            x (torch tensor of shape BxD): current point from which we propose
            returns: (torch tensor of shape BxD) new points
        """
        raise NotImplementedError

    def log_density(self, x, x_prime):
        """
            Computes vectorized log of unnormalized log density
            
            x (torch tensor of shape BxD): B points at which we compute log density
            returns (torch tensor of shape B): \log q(x' | x) 
        """
        raise NotImplementedError