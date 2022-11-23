import torch
import numpy as np
from torch_scatter import scatter_add
from scipy import integrate
from sdegen import utils

def prior_likelihood(z, sigma):
    N = z.shape[0] #edge_dim
    likelihood = -N/2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2) / (2*sigma**2)
    return likelihood
#epsilon = torch.randn_like(samples)

from torch_scatter import scatter_add

def divergence(model,data,d,time_step,eps=0.01):
    #divergence_full is dscore/dd
    device = model.device
    score = model.get_score(data,d,time_step) 
    num_edge = data.num_edges
    eps_edge = torch.ones(num_edge).unsqueeze(-1) *  torch.tensor(eps,dtype=torch.float).unsqueeze(-1)
    eps_edge = eps_edge.to(device)
    d_perturbed = d + eps_edge
    score_ = model.get_score(data,d_perturbed,time_step)
    div = (score_ - score) / eps
    return div

def divergence_hutch(model,data,d,time_step,eps=0.001):
    score = model.get_score(data,d,time_step) 
    perturb = 2 * eps * (np.random.randint(0,2,score.shape[0]) - 0.5)
    d_perturbed = d + perturb
    score_ = model(data,d_perturbed,time_step)
    diff = score_ - score
    div = diff * perturb
    div = np.array(div) / eps **2
    return div


def SDE_ode_distance_sampeler(SDE_model,data,sigma=25, num_steps = 50, 
                            eps=1e-3,num_langevin_steps=1):
    num_edge = len(data.edge_type)
    device = SDE_model.device
    t = torch.ones(1,device=device)
    init_d = torch.randn(num_edge,device=device) * utils.marginal_prob_std(t,sigma,device=device)
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    d = init_d[:,None]
    batch_size = len(data.smiles)
    dlogp = 0
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = utils.diffusion_coeff(time_step,sigma,device=device)
            score =  SDE_model.get_score(data, d, batch_time_step)

            pertrub = 0.5 * g **2 *step_size * score
            div = divergence(SDE_model,data,d,batch_time_step)
            dlogp += -0.5 * g **2 * eps * div
            d = d.squeeze(-1) + pertrub
            d = d.unsqueeze(-1)
            #d_vecs.append(d)
    return d, dlogp
