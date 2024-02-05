from Datasets import banana_dataset,funnel_dataset
from Datasets import normal
import matplotlib.pyplot as plt 
from utils.plotting_utils import *
from AVS_Sampler.AVC_sampler import AVS
standart_normal = normal.Normal(loc=0.,scale=1.)
Funnel_distribution = funnel_dataset.Funnel()
fig, axes = plt.subplots(ncols=1, figsize=(7, 7))
r = plot_distribution(Funnel_distribution, bounds=((-5, 5), (-5, 5)), ax=axes, num=100, n_levels=20, filled=True)[1]
axes.set_title("Neal's funnel")
axes.set_aspect('equal')
fig.colorbar(r, ax=axes)
plt.show()
banan = banana_dataset.Banan()
torch.manual_seed(14)
curr_dist = banan
energy_func = curr_dist.log_density
avs =AVS(energy_func=energy_func, 
    target_dim=2, 
    aux_dim=2)

avs.train(max_iters = 1000,     learning_rate=1e-4)
plt.plot(avs.losses)

convergence_detection_experiment(
    curr_dist,
    samplers =[avs],
    options = dict(n_steps = 1000, 
                   n_parallel = 3, 
                   initial_point = torch.tensor([0., 0.]),
                   perturb=0.01
    )
)
plt.show()