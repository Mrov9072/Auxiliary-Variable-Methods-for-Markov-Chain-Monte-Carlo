import torch
import numpy as np
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

def plot_points(xs, ax, i=0, j=1, color=True):
    ax.set_title('points')
    ax.set_xlabel(f'coordinate {i}')
    ax.set_ylabel(f'coordinate {j}')
    
    n_parallel, n_samples, _ = xs.shape
    c = np.arange(n_samples) if color else None
    for k in range(n_parallel):
        ax.scatter(xs[k, :, i], xs[k, :, j], s=5, c=c)
    return ax

def plot_log_density(xs, ax, distribution):
    ax.set_title('log_density')
    ax.set_xlabel('iteration')
    ax.set_ylabel('log density')
    
    n_parallel, n_samples, _ = xs.shape
    for k in range(n_parallel):
        density = distribution.log_density(torch.tensor(xs[k]))
        ax.plot(density.numpy(), label=f'run {k + 1}')
    ax.legend(loc='best')
    return ax

def find_first(X):
    for i in range(X.shape[0]):
        if X[i]:
            return i
    return -1
  
def integrated_autocorr(x, acf_cutoff=0.0):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = len(x)

    tau = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        f = acf(x[:,j], nlags=n, unbiased=False, fft=True)
        window = find_first((f <= acf_cutoff).astype(np.uint8))
        tau[j] = 1 + 2*f[1:window].sum()

    return tau

def compute_ess(x, acf_cutoff=0.0):
    tau = integrated_autocorr(x, acf_cutoff=acf_cutoff)
    return x.shape[0] / tau
  
def plot_autocorr(xs, axes, step=100, mean=False, label=None):
    n_parallel, _, dim = xs.shape
    
    for i in range(dim):
        axes[i].set_title(f'autocorrelation (step={step}, coordinate {i})')
        axes[i].set_xlabel('lag')
        axes[i].set_ylabel('autocorrelation')
        
        acfs = [acf(xs[k, ::step, i]) for k in range(n_parallel)]
        ess = np.stack([compute_ess(xs[k]) for k in range(n_parallel)], axis=0)
        
        if mean:
            axes[i].plot(np.mean(acfs, axis=0), marker='o', label=(label or '') + f" ESS = {ess.mean(axis=0)[i]:.2f}")
        else:
            for k, acf_ in enumerate(acfs):
                axes[i].plot(acf_, marker='o', label=f'run {k} ESS = {ess[k][i]:.2f}')
            
        axes[i].legend(loc='best')
    return axes
        
def cummean(arr, axis=0):
    if axis < 0:
        axis = axis + len(arr.shape)
    
    arange = np.arange(1, arr.shape[axis] + 1)
    arange = arange.reshape((1,) * axis + (-1,) + (1,) * (len(arr.shape) - axis - 1))
    return arr.cumsum(axis=axis) / arange
    
def plot_statistics(xs, axes, skip=0, step=1):
    xs = xs[:, skip::step]
    n_parallel, _, dim = xs.shape
    
    means = cummean(xs, axis=1)
    variances = cummean(xs**2, axis=1) - means**2
    
    for i in range(n_parallel):
        for j in range(dim):
            ax = axes[j]
            ax.set_title(f'coordinate {j} running mean and std')
            ax.set_xlabel('iteration')
            ax.set_ylabel(f'coordinate {j}')
            
            x = np.arange(means.shape[1])
            y = means[i, :, j]
            e = np.sqrt(variances[i, :, j])
            
            r = ax.plot(y, label=f'chain {i} mean')
            ax.plot(e, linestyle='--', c=r[0].get_color(), label=f'chain {i} std')
            ax.legend(loc='best')
        
    return axes
  
def plot_traceplot(xs, axes):
    n_parallel, _, dim = xs.shape
    
    for i in range(n_parallel):
        for j in range(dim):
            ax = axes[j]
            ax.set_title(f'coordinate {j} traceplot')
            ax.set_xlabel('iteration')
            ax.set_ylabel(f'coordinate {j}')
           
            ax.plot(xs[i, :, j], label=f'chain {i}')
            ax.legend(loc='best')
        
    return axes

def plot_distribution(distribution, bounds, ax, num=50, n_levels=None, filled=False, exp=False):
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    x, y = np.meshgrid(np.linspace(x_min, x_max, num=num), np.linspace(y_min, y_max, num=num))
    s = x.shape
    xy = np.stack([x.reshape(-1), y.reshape(-1)], axis=1)
    z = distribution.log_density(torch.tensor(xy, dtype=torch.float32)).numpy().reshape(s)
    if exp:
      z = np.exp(z)
    
    plot = ax.contourf if filled else ax.contour
    r = plot(x, y, z, n_levels)
    return ax, r

def plot_distribution_grad(distribution, bounds, ax, num=50):
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    x, y = np.meshgrid(np.linspace(x_min, x_max, num=num), np.linspace(y_min, y_max, num=num))
   
    s = x.shape
    xy = np.stack([x.reshape(-1), y.reshape(-1)], axis=1)
    z = distribution.grad_log_density(torch.tensor(xy, dtype=torch.float32)).numpy()
    u, v = z[..., 0], z[..., 1]
    c = np.sqrt(u**2 + v**2)
    
    ax.quiver(x, y, u, v, c, angles='xy')
    return ax

def convergence_detection_experiment(distribution, samplers, options, bounds=[(-5, 5), (-5, 5)]):
    D = options['initial_point'].shape[0]
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    gs = fig.add_gridspec(2 * len(samplers), 1 + D)

    for i, sampler in enumerate(samplers):
#         result = simulate(distribution, proposal, **options)
        result = sampler.simulate(**options)
#         print(result['means'])
        ax = [fig.add_subplot(gs[2*i:2*i+2, 0])] + [fig.add_subplot(gs[2*i, 1+k]) for k in range(D)] + [fig.add_subplot(gs[2*i+1, 1+k]) for k in range(D)]
        plot_points(result['points'], ax[0])
        plot_statistics(result['points'], ax[1:1 + D])
        plot_traceplot(result['points'], ax[1 + D: 1 + 2 * D])
        
def autocorr_experiment_avs(distribution, samplers, steps, options):
    fig, axes = plt.subplots(nrows=len(steps), ncols=options['initial_point'].shape[0], figsize=(20, 10))

    for sampler in samplers:
        result = sampler.simulate(**options)
        xs = result['points']
        
        for i, step in enumerate(steps):
            plot_autocorr(xs, axes[i], step=step, mean=True, label=f'{sampler}')
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    return fig
