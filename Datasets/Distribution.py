class Distribution:
    """Abstract class for unnormalized distribution"""
    def log_density(self,x):
        """
        Computes vectorized log of unnormalized log density
            x (torch tensor of shape BxD): B points at which we compute log density
            returns (torch tensor of shape B): \log \hat{\pi}(x) 
        """
        raise NotImplementedError
    
    def grad_log_density(self,x):
        """
         Computes vectorized gradient \nabla_x \log \pi(x)
            x (torch tensor of shape BxD): point at which we compute \nabla \log \pi
            returns (torch.tensor of shape BxD): gradients of log density
        """
        x=x.clone().requires_grad_()
        logp=self.log_density(x)
        logp.sum().backward()
        return x.grad()

    