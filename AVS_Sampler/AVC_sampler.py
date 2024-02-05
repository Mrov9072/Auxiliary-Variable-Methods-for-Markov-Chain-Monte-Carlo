import torch
import time
from  torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from tqdm import tqdm
from .mlp import FullyConnectedMLP

class AVS:
    def __init__(self, 
                 energy_func, 
                 target_dim, 
                 aux_dim,  
                ):
        """
            Constructs AV MCMC sampler
        """
        
        # q(a) = N (a| 0, I) 
        # q_phi(x|a) = N (x| mu_phi(a), Sigma_phi(a))
        # p_Theta(a|x) = N (a| mu_Theta(x), Sigma_Theta(x)) 
        
        # define prior distribution q(a)
        # define distribution q_phi(x, a)
        # define distribution p_Theta(a|x)
        # we fit Variational Approx:
        # KL(q_phi(x, a)||p(x)p_theta(a|x)) -> min wrt phi, Theta
        # we get phi*, Theta*
        
        # then we have: 
        # p*(a|x)
        # q*(a) = int(q*(x, a))dx
        # q*(x|a) = q*(x, a)/q*(a)
        
        # then we perform step in auxiliary space
        # a' ~ N(a'|a, perturb*I)
        
        self.target_logprob = energy_func
#         self.num_layers = num_layers
        self.target_dim = target_dim
        self.aux_dim = aux_dim

        
###############################distributions#######################################          
    def _aux_a(self, mode = 'train'): 
        """ Variational Distribution on a """  
        # q(a) - prior 
        # q(a) = N (a| 0, I)     
        
        # trn_smpls -  represents number of elements in batch (for train)
        # num_chains - represents number of parallel simulationd (for inference)
        # self.aux_dim - dimenssion of auxiliary space   
        
        if mode == 'train':
            batch = self.trn_smpls           
        elif mode == 'sampling':
#             batch = self.num_chains
            batch = self.n_parallel
            
        aux_mean = torch.zeros([batch, self.aux_dim])  
        scale = torch.eye(self.aux_dim).repeat(batch, 1)
        scale = torch.reshape(scale, (batch, self.aux_dim, self.aux_dim))       
        q_a = MultivariateNormal(loc=aux_mean, covariance_matrix=scale)

        return q_a
    
    def _aux_xgiva(self, a):
        """ Variational distribution on x given a"""
        # q_phi(x|a) = N (x| mu_phi(a), Sigma_phi(a))
        
        mean_and_sigma = self.SOME_NEURAL_NETWORK_phi(a)
        mean = mean_and_sigma[:, :self.target_dim]
        sigma = torch.exp(mean_and_sigma[:, self.target_dim:])
        sigma = torch.diag_embed(sigma)
        q_phi_x_a = MultivariateNormal(loc=mean, covariance_matrix=sigma)       
        return q_phi_x_a
    
    def _aux_agivx(self, x_samples):
        """ Variational distribution on a given x"""
        # p_Theta(a|x) = N (a| mu_Theta(x), Sigma_Theta(x))   
        # авторы также пробуют смесь нормальных распределений для каких-то экспериментов, пока ограничимся одним нормальным
        
        mean_and_sigma = self.SOME_NEURAL_NETWORK_Theta(x_samples) 
        mean = mean_and_sigma[:, :self.aux_dim]
        sigma = torch.exp(mean_and_sigma[:, self.aux_dim:])
        sigma = torch.diag_embed(sigma)
        p_Theta_a_x = MultivariateNormal(loc=mean, covariance_matrix=sigma)       
        return p_Theta_a_x
    
##############################acceptance ratio##############################################
    def _get_aratio(self):
        # calculate log of acceptance ratio at points x = self.init_x, x' = self.prop_x, a = a_samp:
        # ratio = [q_phi(x|a)*p_Theta(a|x')*p(x')]/[q_phi(x'|a)*p_Theta(a|x)*p(x')]
        self.aratio = (self.smpQxa_samp.log_prob(self.init_x) +
                       self.smpPax_prop.log_prob(self.a_prime) +
                       self.target_logprob(self.prop_x) -
                       self.smpQxa_prime.log_prob(self.prop_x) -
                       self.smpPax_init.log_prob(self.a_samp) -
                       self.target_logprob(self.init_x)
                      )

        return  self.aratio

############################sampler#####################################            
#     def sample(self, n_samples, num_chains= self.num_chains, initial_point = None):
    def simulate(self, n_steps, n_parallel=3, initial_point = None, perturb=1.0):
        """
            Run ` n_parallel ` simulations for `n_steps` starting from `initial_point`
            
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
        num_samples = n_steps
        self.n_parallel = n_parallel
        self.perturb = perturb

        
        # set up prior
        self.Qa = self._aux_a(mode = 'sampling')

        # get initial a
        self.A =  torch.zeros([self.n_parallel,
                               self.aux_dim])                               
#             self.num_chains,
                               


        # get initial x0
        self.smpQxa_samp = self._aux_xgiva(self.A)
        self.init_x = self.smpQxa_samp.sample()
        
        
        if initial_point is not None:
            self.init_x = initial_point.repeat(n_parallel, 1)

        sample_time = 0.0
        
        
        ##################
        self._rejected = torch.zeros(n_parallel)
        xs = []
        means = []
        variances = []
        sums =0
        squares_sum = 0
        ##############
        
        x_samps =[]
        for i in range(num_samples):
            
            time1 = time.time()
            
            # set up p_Theta(a|x) i.e. p_Theta_a_x init
            self.smpPax_init = self._aux_agivx(self.init_x) 

            # get samples of a
            self.a_samp = self.smpPax_init.sample()

            # set up q_phi(x|a)
            self.smpQxa_samp = self._aux_xgiva(self.a_samp)
            # perform step in auxiliary space of a, get sample a'
            sigma_pertrubed = torch.diag_embed(torch.ones_like(self.a_samp))*self.perturb

            self.a_prime = MultivariateNormal(loc=self.a_samp, covariance_matrix=sigma_pertrubed).sample()
            # set q_phi_x_a
            self.smpQxa_prime = self._aux_xgiva(self.a_prime)
            # get proposal x
            self.prop_x = self.smpQxa_prime.sample()
            # set up p_Theta_a_x
            self.smpPax_prop = self._aux_agivx(self.prop_x)
            
            
            self._get_aratio()
            u = torch.rand(self.aratio.shape)
            accept = torch.less(torch.log(u), self.aratio)


            ######################
            self._rejected += (1 - 1*accept).type(torch.float32)
            ###################
            accept = accept.repeat(self.init_x.shape[1], 1).T
            
            self.n_accept = torch.mean(torch.where(accept, torch.ones(accept.shape), torch.zeros(accept.shape)))
            self.x_samples = torch.where(accept, self.prop_x, self.init_x) 
       
            time2 = time.time()
            sample_time += time2 - time1
            
            x_samps.append(np.array(self.x_samples))
            self.init_x = self.x_samples
            
            ######################################
            
            xs.append(self.x_samples.numpy().copy())
            
            sums += xs[-1]
            squares_sum += xs[-1]**2
            
            mean, squares_mean = sums / (i + 1), squares_sum / (i + 1)
            means.append(mean.copy())
            variances.append(squares_mean - mean**2)
        
        
        xs = np.stack(xs, axis=1)        
        means = np.stack(means, axis=1)
        variances = np.stack(variances, axis=1)
            
        result = dict()
        result['points'] = np.transpose(np.array(x_samps), [1,0, 2])
        result['n_rejected'] = self._rejected.numpy()
        result['rejection_rate']=(self._rejected / num_samples).mean().item()
        result['means'] = means
        result['variances'] = variances
        
#         return np.array(x_samps), sample_time
        return result
    
################################loss################################################    
    def get_loss(self):


        loss = torch.mean(self.Qa.log_prob(self.A) +
                               self.Qxa.log_prob(self.x_samples) -
                               self.Pax.log_prob(self.A) -
                               self.target_logprob(self.x_samples))
        return loss    
    
#################################train#########################################
    def train(self, 
              max_iters=10, 
#               optimizer=tf.train.AdamOptimizer,
#               num_layers=3,
#               hdn_dim=100,

#               init=True, 
              extra_phi_steps=10, 
              learning_rate=1e-3,
              hdn_dim = 200,
              num_layers=5, 
              trn_samples=10,
              **kwargs):
        """ Trains and returns total time to train"""
        
        self.num_layers= num_layers
        self.extra_phi_steps =extra_phi_steps
        self.hdn_dim= hdn_dim
        self.hdn_dim = hdn_dim
        self.trn_smpls = trn_samples # num of train samples in the batch for NN 
        
        
        
        
        hiddens = []
        for layer_num in range(self.num_layers):
            hiddens.append(self.hdn_dim)

        self.SOME_NEURAL_NETWORK_phi = FullyConnectedMLP(input_shape=self.aux_dim, 
                                                         hiddens=hiddens, 
                                                         output_shape=2*self.target_dim)
        
        self.SOME_NEURAL_NETWORK_Theta = FullyConnectedMLP(input_shape=self.target_dim, 
                                                           hiddens=hiddens, 
                                                           output_shape=2*self.aux_dim)
        
        
        self.losses = [0.0]
        self.deltas = []
        train_time = 0.0
        
# #####################################
        self.Qa = self._aux_a()
#         self.A = self._aux_a().sample()
# #####################################      
        
                
        optimizer_Theta =torch.optim.Adam(params=self.SOME_NEURAL_NETWORK_phi.parameters(), lr=learning_rate)
        optimizer_phi =torch.optim.Adam(params=self.SOME_NEURAL_NETWORK_Theta.parameters(), lr=learning_rate)
        
        decayRate = 0.99
        
        lr_scheduler_Theta = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_Theta, gamma=decayRate)
        lr_scheduler_phi = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_phi, gamma=decayRate)
        
        for t in tqdm(range(max_iters)):
            time1 = time.time()
            
            #####################################    
            self.A = self._aux_a().sample()
            self.Qxa = self._aux_xgiva(self.A) 
            # reparametrization trick is already implemented in torch (rsample)
            self.x_samples = self.Qxa.rsample() 
            self.Pax = self._aux_agivx(self.x_samples) 
            #####################################  
            for ex in range(self.extra_phi_steps):
                optimizer_phi.step()              
            ########################################   
            # calculate loss    
            loss = self.get_loss()
            ########################################
            loss.backward()
            self.losses.append(loss.item())
            
            # Update the model parameters with the optimizer
          
            
            optimizer_Theta.step()
            optimizer_phi.step()
            
#             lr_scheduler_Theta.step()
#             lr_scheduler_phi.step()
            
            optimizer_Theta.zero_grad()
            optimizer_phi.zero_grad()
            
            
            time2 = time.time()
            train_time += time2 - time1

            self.deltas.append(abs(self.losses[-1] - self.losses[-2]))
            
        self.losses= self.losses[1:]
        self.deltas= self.deltas[1:]