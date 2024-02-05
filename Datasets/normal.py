from .Distribution import Distribution
import torch.distributions as dist
import numpy as np

class Normal(Distribution):
    """Represents normal distribution N(mean, std)"""

    def __init__(self, loc, scale):
        """
            loc (torch tensor of shape D): mean
            scale (torch tensor of shape D) std
        """
        self.dist = dist.Normal(loc=loc, scale=scale)

    def log_density(self, x):
        return self.dist.log_prob(x).sum(dim=-1)

class Funnel(Distribution):
    def __init__(self):
        self.d = dist.Normal(loc=0., scale=np.sqrt(3.))

    def log_density(self, y):
        x, z = y[:, 0], y[:, 1]
        logp_z = self.d.log_prob(z).sum(dim=-1)
        logp_x = dist.Normal(loc=0., scale=z.mul(0.25).exp()).log_prob(x)
        return logp_z + logp_x