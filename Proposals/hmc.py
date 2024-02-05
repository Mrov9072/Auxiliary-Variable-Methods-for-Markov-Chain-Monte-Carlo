from proposal import Proposal
from torch import distributions as dist
import torch

class HMC(Proposal):
    """HMC proposal"""
    
    def __init__(self, eps, d, n_steps=5, method='leapfrog'):
        self.d = dist.Normal(loc=0., scale=1.)
        self.dist = d
        self.eps = eps
        self.n_steps = n_steps
        self.method = method
        self._method = {
            'leapfrog': self._leapfrog,
            'euler': self._euler,
            'simple_euler': self._simple_euler
        }[method]
          
    def _energy(self, x, v):
        return -self.dist.log_density(x) + self.d.log_prob(v).sum(dim=-1)
        
    def _leapfrog(self, x, v):
        self.energy = []
        for _ in range(self.n_steps):
            v -= 0.5 * self.eps * self.dist.grad_log_density(x)
            x = x + self.eps * v
            v -= 0.5 * self.eps * self.dist.grad_log_density(x)
            self.energy.append(self._energy(x, v))
        return x, v
    
    def _euler(self, x, v):
        self.energy = []
        for _ in range(self.n_steps):
            v -= self.eps * self.dist.grad_log_density(x)
            x = x + self.eps * v
            self.energy.append(self._energy(x, v))
        return x, v
    
    def _simple_euler(self, x, v):
        self.energy = []
        for _ in range(self.n_steps):
            x, v = x + self.eps * v, v - self.eps * self.dist.grad_log_density(x)
            self.energy.append(self._energy(x, v))
        return x, v
        
    def sample(self, x):
        v = self.d.sample(sample_shape=x.shape)
        self.v0 = v.clone()
        self.x0 = x.clone()
        
        x, v = self._method(x, v)
        self.v = v
        return x
    
    def log_density(self, x, x_prime):
        if torch.norm(x - self.x0).item() < 1e-5:
            return self.d.log_prob(self.v0).sum(dim=-1)
        else:
            return self.d.log_prob(self.v).sum(dim=-1)
    
    def __str__(self):
        return f"HMC eps={self.eps}, n_steps={self.n_steps}, method={self.method}"