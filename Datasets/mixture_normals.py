from .Distribution import Distribution
import torch
import torch.distributions as dist

class MixtureOfNormals(Distribution):
    """Represents mixture of normals: \pi(x) = \sum_k \pi_k N(x | mean_k, std_k)"""
    def __init__(self, locs, scales, pi):
        """
            locs (torch tensor of shape NxD): locs[k] = mean_k
            scales (torch tensor of shape NxD): scales[k] = std_k
            pi (torch.tensor of shape N): pi[k] = pi_k
        """
        self.dists = [
            dist.Normal(loc=loc, scale=scale)
            for loc, scale in zip(locs, scales)
        ]
        self.pi = pi
        
    def log_density(self, x):
        log_densities = torch.stack([
            d.log_prob(x).sum(dim=-1)
            for d in self.dists
        ], dim=0)
        return torch.logsumexp(torch.log(self.pi).view(-1, 1) + log_densities, dim=0)
