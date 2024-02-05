import torch
from .Distribution import Distribution
import torch.distributions as dist
import numpy as np

class LangevinMixture(Distribution):
    """
        This distribution is defined as posterior on theta_1, theta_2 given
        prior: theta_1 ~ N(0, \sigma_1^2), theta_2 ~ N(0, \sigma_2^2)
        likelihood: x_i ~ 0.5 * N(theta_1, \sigma_x^2) + 0.5 * N(theta_1 + theta_2, \sigma_x^2)
        
        More detailed information can be found at 
        https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf
    """
    
    def __init__(self, theta1=0., theta2=1., 
                       s1=np.sqrt(10.), s2=1., sx=np.sqrt(2.), N=100):
        mu = torch.tensor([theta1, theta1 + theta2])
        pi = np.random.binomial(n=1, p=0.5, size=N).astype(np.int64)

        d = dist.Normal(loc=mu, scale=sx)
        self.X = d.sample((N,))[np.arange(N), pi]
    
        self.d1 = dist.Normal(loc=0., scale=s1)
        self.d2 = dist.Normal(loc=0., scale=s2)
        self.dx = dist.Normal(loc=0., scale=sx)
    
    def log_density(self, theta):
        theta1, theta2 = theta[..., 0], theta[..., 1]
        log_prior = self.d1.log_prob(theta1) + self.d2.log_prob(theta2)
        
        mu = torch.stack([theta1, theta1 + theta2], dim=-1)
        
        log_likelihood = self.dx.log_prob(self.X[None, :, None] - mu[:, None])
        log_likelihood = torch.logsumexp(log_likelihood, dim=-1)
        return log_prior + log_likelihood.sum(dim=-1)