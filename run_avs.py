from AVS_Sampler.AVC_sampler import AVS
import torch
from  torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from Datasets.funnel_dataset import Funnel
loc = torch.tensor([1,2])
scale = torch.eye(2)
normal = MultivariateNormal(loc=loc, covariance_matrix=scale)
energy_func = normal.log_prob
avs =AVS(energy_func=energy_func, 
    target_dim=2, 
    aux_dim=2, 
    hdn_dim=2)
avs.train(max_iters = 10000,     learning_rate=1e-3)
plt.plot(avs.losses)
plt.show()
Funnel_distribution = Funnel()
energy_func_funnel = Funnel_distribution.log_density
avs_funnel =AVS(energy_func=energy_func, 
    target_dim=2, 
    aux_dim=2, 
    hdn_dim=2)
avs_funnel.train(max_iters = 10000,     learning_rate=1e-3)
plt.plot(avs_funnel.losses)
plt.show()
