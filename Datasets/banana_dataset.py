from .Distribution  import *
import torch.distributions as dist

class Banan(Distribution):
    def __init__(self):
        self.x2 = dist.Normal(loc=0. , scale = 1)

    def log_density(self, y):
        y2, y1 = y[:, 0], y[:, 1]
        logp_x2 = self.x2.log_prob(y1).sum(dim=-1)
        logp_x1 = dist.Normal(loc = 2 * (y2 ** 2 - 5), scale = 1).log_prob(y1)
        return logp_x1 + logp_x2