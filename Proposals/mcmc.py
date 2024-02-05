import torch
import torch.distributions as dist
import numpy as np

class MCMC:
    def __init__(self, distribution, proposal):
        """
            Constructs MCMC sampler
        
            distribution (Distribution): distribution from which we sample
            proposal (Proposal): MCMC proposal
        """
        self.distribution = distribution
        self.proposal = proposal
    
    def _step(self, x):
        x_prime = self.proposal.sample(x)
        acceptance_prob = self.acceptance_prob(x_prime, x)
        
        mask = torch.rand(x.shape[0]) < acceptance_prob
        x[mask] = x_prime[mask]
        self._rejected += (1 - mask).type(torch.float32)
        return x

    def simulate(self, initial_point, n_steps, n_parallel=10):
        """
            Run `n_parallel ` simulations for `n_steps` starting from `initial_point`
            
            initial_point (torch tensor of shape D): starting point for all chains
            n_steps (int): number of samples in Markov chain
            n_parallel (int): number of parallel chains
            returns: dict(
                points (torch tensor of shape n_parallel x n_steps x D): samples
                n_rejected (numpy array of shape n_parallel): number of rejections for each chain
                rejection_rate (float): mean rejection rate over all chains
                means (torch tensor of shape n_parallel x n_steps x D): means[c, s] = mean(points[c, :s])
                variances (torch tensor of shape n_parallel x n_steps x D): variances[c, s, d] = variance(points[c, :s, d])
            )
        """
        xs = []
        x = initial_point.repeat(n_parallel, 1)
        self._rejected = torch.zeros(n_parallel)
        
        dim = initial_point.shape[0]
        sums = np.zeros([n_parallel, dim])
        squares_sum = np.zeros([n_parallel, dim])
        
        means = []
        variances = []        
        
        for i in range(n_steps):
            x = self._step(x)
            xs.append(x.numpy().copy())
            
            sums += xs[-1]
            squares_sum += xs[-1]**2
            
            mean, squares_mean = sums / (i + 1), squares_sum / (i + 1)
            means.append(mean.copy())
            variances.append(squares_mean - mean**2)
        
        xs = np.stack(xs, axis=1)        
        means = np.stack(means, axis=1)
        variances = np.stack(variances, axis=1)
        
        return dict(
            points=xs,
            n_rejected=self._rejected.numpy(),
            rejection_rate=(self._rejected / n_steps).mean().item(),
            means=means,
            variances=variances
        )
        
    def acceptance_prob(self, x_prime, x):
        """
            In this function you need to compute 
            probability of acceptance \rho(x' | x)

            x_prime (numpy array): new point
            x (numpy array): current point
            returns: acceptance probability \rho(x', x)
        """
    
        # TODO
        pi_new = self.distribution.log_density(x_prime)
        pi_old = self.distribution.log_density(x)
        q_new = self.proposal.log_density(x_prime, x)
        q_old = self.proposal.log_density(x, x_prime)
        ratio = torch.exp(pi_new - pi_old + q_new - q_old)
        return ratio.clamp(0., 1.)
    
def simulate(distribution, proposal, initial_point, n_samples, n_parallel=10):
    mcmc = MCMC(distribution, proposal)
    return mcmc.simulate(initial_point, n_samples, n_parallel)