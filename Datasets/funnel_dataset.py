from .Distribution import Distribution
import torch.distributions as dist
import numpy as np

class Funnel(Distribution):
    def __init__(self):
        self.d = dist.Normal(loc=0., scale=np.sqrt(3.))

    def log_density(self, y):
        x, z = y[:, 0], y[:, 1]
        logp_z = self.d.log_prob(z).sum(dim=-1)
        logp_x = dist.Normal(loc=0., scale=z.mul(0.25).exp()).log_prob(x)
        return logp_z + logp_x
