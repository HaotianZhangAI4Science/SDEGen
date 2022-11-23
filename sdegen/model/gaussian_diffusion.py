import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from sdegen import utils, layers
from sdegen.utils import GaussianFourierProjection

def extract(a, t, d, edge2graph):
    out = torch.tensor(a[t.cpu().numpy()], device=d.device, dtype=d.dtype)
    return out[edge2graph].reshape(*d.shape)
    # extend to edge dim

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class DenoisingDiffusion(torch.nn.Module):
    def __init__(self, config):
        super(DenoisingDiffusion, self).__init__()
        self.config = config
        self.loss_type = config.diffusion.loss_type
        self.betas = get_named_beta_schedule(config.diffusion.noise_schedule, 
                                             config.diffusion.timesteps,)
        # Use float64 for accuracy.
        betas = np.array(self.betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.noise_type = self.config.model.noise_type

        # time step embedding for diffusion
        self.t_embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.hidden_dim),
                                nn.Linear(self.hidden_dim, self.hidden_dim))
        self.dense1 = nn.Linear(self.hidden_dim, 1)

        self.node_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.input_mlp = layers.MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim], activation=self.config.model.mlp_act)
        self.output_mlp = layers.MultiLayerPerceptron(2 * self.hidden_dim, \
                                [self.hidden_dim, self.hidden_dim // 2, 1], activation=self.config.model.mlp_act)
        self.model = layers.GraphIsomorphismNetwork(hidden_dim=self.hidden_dim, \
                                 num_convs=self.config.model.num_convs, \
                                 activation=self.config.model.gnn_act, \
                                 readout="sum", short_cut=self.config.model.short_cut, \
                                 concat_hidden=self.config.model.concat_hidden)

        """
        Techniques from "Improved Techniques for Training Score-Based Generative Models"
        1. Choose sigma1 to be as large as the maximum Euclidean distance between all pairs of training data points.
        2. Choose sigmas as a geometric progression with common ratio gamma, where a specific equation of CDF is satisfied.
        3. Parameterize the Noise Conditional Score Networks with f_theta_sigma(x) =  f_theta(x) / sigma
        """

    def epsilon(self, data):
        """
        get random noise
        """
        assert data.edge_index.size(1) == data.edge_length.size(0)
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]  

        if self.noise_type == 'symmetry':
            num_nodes = scatter_add(torch.ones(data.num_nodes, dtype=torch.long, device=self.device), data.batch) # (num_graph)
            num_cum_nodes = num_nodes.cumsum(0) # (num_graph)
            node_offset = num_cum_nodes - num_nodes # (num_graph)
            edge_offset = node_offset[edge2graph] # (num_edge)

            num_nodes_square = num_nodes ** 2 # (num_graph)
            num_nodes_square_cumsum = num_nodes_square.cumsum(-1) # (num_graph)
            edge_start = num_nodes_square_cumsum - num_nodes_square # (num_graph)
            edge_start = edge_start[edge2graph]

            all_len = num_nodes_square_cumsum[-1]

            node_index = data.edge_index.t() - edge_offset.unsqueeze(-1)
            #node_in, node_out = node_index.t()
            node_large = node_index.max(dim=-1)[0]
            node_small = node_index.min(dim=-1)[0]
            undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start

            symm_noise = torch.cuda.FloatTensor(all_len, device=self.device).normal_()
            d_noise = symm_noise[undirected_edge_id].unsqueeze(-1) # (num_edge, 1)

        elif self.noise_type == 'rand':
            d_noise = torch.randn_like(d)

        else:
            raise NotImplementedError('noise type must in [distance_symm, distance_rand]')

        return d_noise

    def q_sample(self, d_start, t, noise=None, edge2graph=None):
        noise = torch.randn_like(d_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, d_start, edge2graph) * d_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, d_start, edge2graph) * noise
        )

    def epsilon_theta(self, data, noisy_d, t):
        """
        estimate noise (epsilon)
        """
        node_attr = self.node_emb(data.atom_type) # (num_node, hidden)
        edge_attr = self.edge_emb(data.edge_type) # (num_edge, hidden)

        d_emb = self.input_mlp(noisy_d) # (num_edge, hidden)

        assert data.edge_index.size(1) == data.edge_length.size(0)
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]  

        # if self.config.scheme.time_continuous:
        # time embedding
        t_embedding = self.t_embed(t)             #(batch_dim, hidden_dim) = (128, 256)
        t_embedding = self.dense1(t_embedding)    #(batch_dim, 1) = (128,1)
        d_emb = d_emb + t_embedding[edge2graph]

        edge_attr = d_emb * edge_attr # (num_edge, hidden)

        output = self.model(data, node_attr, edge_attr)
        h_row, h_col = output["node_feature"][data.edge_index[0]], output["node_feature"][data.edge_index[1]] # (num_edge, hidden)

        distance_feature = torch.cat([h_row*h_col, edge_attr], dim=-1) # (num_edge, 2 * hidden)
        scores = self.output_mlp(distance_feature) # (num_edge, 1)

        return scores #* (1. / used_sigmas)

    def forward(self, data, device=None, noise=None,):
        """
        Input:
            data: torch geometric batched data object
        Output:
            loss
        """
        # a workaround to get the current device, we assume all tensors in a model are on the same device.
        if device == None:
            raise
        self.device = device
        data = self.extend_graph(data, self.order)
        data = self.get_distance(data)

        assert data.edge_index.size(1) == data.edge_length.size(0)
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]  

        # sample noise level
        t = torch.randint(0, self.num_timesteps, (data.num_graphs,), device=self.device).long()
        d = data.edge_length # (num_edge, 1)
        noise = self.epsilon(data)

        assert noise.shape == d.shape
        noisy_d = self.q_sample(d_start=d, t=t, noise=noise, edge2graph=edge2graph)
        recon_d = self.epsilon_theta(data=data, noisy_d=noisy_d, t=t)

        if self.loss_type == 'l1':
            loss = (noise - recon_d).abs()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, recon_d, reduce=False)
        else:
            raise NotImplementedError()

        loss = loss.view(-1)
        loss = scatter_add(loss, edge2graph) # (num_graph)
        return loss
    
    @torch.no_grad()
    # extend the edge on the fly, second order: angle, third order: dihedral
    def extend_graph(self, data: Data, order=3):

        def binarize(x):
            return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

        def get_higher_order_adj_matrix(adj, order):
            """
            Args:
                adj:        (N, N)
                type_mat:   (N, N)
            """
            adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                        binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

            for i in range(2, order+1):
                adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
            order_mat = torch.zeros_like(adj)

            for i in range(1, order+1):
                order_mat += (adj_mats[i] - adj_mats[i-1]) * i

            return order_mat

        num_types = len(utils.BOND_TYPES)

        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)   # (N, N)
        type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N) # modify data
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
        data.is_bond = (data.edge_type < num_types)
        assert (data.edge_index == edge_index_1).all()

        return data

    @torch.no_grad()
    def get_distance(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        data.edge_length = d
        return data   


    @torch.no_grad()
    def get_score(self, data: Data, d, sigma):
        """
        Input:
            data: torch geometric batched data object
            d: edge distance, shape (num_edge, 1)
            sigma: noise level, tensor (,)
        Output:
            log-likelihood gradient of distance, tensor with shape (num_edge, 1)         
        """
        node_attr = self.node_emb(data.atom_type) # (num_node, hidden)
        edge_attr = self.edge_emb(data.edge_type) # (num_edge, hidden)      
        d_emb = self.input_mlp(d) # (num_edge, hidden)
        edge_attr = d_emb * edge_attr # (num_edge, hidden)

        output = self.model(data, node_attr, edge_attr)
        h_row, h_col = output["node_feature"][data.edge_index[0]], output["node_feature"][data.edge_index[1]] # (num_edge, hidden)
        distance_feature = torch.cat([h_row*h_col, edge_attr], dim=-1) # (num_edge, 2 * hidden)
        scores = self.output_mlp(distance_feature) # (num_edge, 1)
        scores = scores * (1. / sigma) # f_theta_sigma(x) =  f_theta(x) / sigma, (num_edge, 1)
        return scores

    @torch.no_grad()
    def convert_score_d(self, score_d, pos, edge_index, edge_length):
        dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])   # (num_edge, 3)
        score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0)

        return score_pos


    @torch.no_grad()
    def distance_Langevin_Dynamics(self, data, d_mod, scorenet, sigmas,
                                   n_steps_each=100, 
                                   step_lr=0.00002, 
                                   clip=1000, 
                                   min_sigma=0,):
        """
        d_mod: initial distance vector. (num_edge, 1)
        """
        scorenet.eval()
        d_vecs = []
        cnt_sigma = 0

        for i, sigma in tqdm(enumerate(sigmas), total=sigmas.size(0), desc="Sampling distances"):
            if sigma < min_sigma:
                break
            cnt_sigma += 1
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for step in range(n_steps_each):
                noise = torch.randn_like(d_mod) * torch.sqrt(step_size * 2)
                score_d = scorenet.get_score(data, d_mod, sigma) # (num_edge, 1)
                score_d = utils.clip_norm(score_d, limit=clip)

                d_mod = d_mod + step_size * score_d + noise

                d_vecs.append(d_mod)
        d_vecs = torch.stack(d_vecs, dim=0).view(cnt_sigma, n_steps_each, -1, 1) # (sigams, 100, num_edge, 1)
        
        return data, d_vecs

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape

        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        model_variance, model_log_variance = (
            np.append(self.posterior_variance[1], self.betas[1:]),
            np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        )

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        )
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        # return {"sample": sample, "pred_xstart": out["pred_xstart"]}
        return sample

    @torch.no_grad()
    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        # if progress:
            # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm
        indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    @torch.no_grad()
    def position_Langevin_Dynamics(self, data, pos_init, scorenet, sigmas, 
                                   n_steps_each=100, step_lr=0.00002,
                                   clip=1000, min_sigma=0):
        """
        # 1. initial pos: (N, 3) 
        # 2. get d: (num_edge, 1)
        # 3. get score of d: score_d = self.get_grad(d).view(-1) (num_edge)
        # 4. get score of pos:
        #        dd_dr = (1/d) * (pos[edge_index[0]] - pos[edge_index[1]]) (num_edge, 3)
        #        edge2node = edge_index[0] (num_edge)
        #        score_pos = scatter_add(dd_dr * score_d, edge2node) (num_node, 3)
        # 5. update pos:
        #    pos = pos + step_size * score_pos + noise

        sampling scheme.
        """

        scorenet.eval()
        pos_vecs = []
        pos = pos_init
        cnt_sigma = 0

        indices = list(range(self.num_timesteps))[::-1]
        # if progress:
            # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm
        indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            d = utils.get_d_from_pos(pos, data.edge_index).unsqueeze(-1) # (num_edge, 1)
            noise = torch.randn_like(pos) * torch.sqrt(step_size * 2)
            score_d = scorenet.get_score(data, d, sigma) # (num_edge, 1)
            score_pos = self.convert_score_d(score_d, pos, data.edge_index, d)
            score_pos = utils.clip_norm(score_pos, limit=clip)

            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

        # for i, sigma in tqdm(enumerate(sigmas), total=sigmas.size(0), desc="Sampling positions"):
            
        #     if sigma < min_sigma:
        #         break

        #     cnt_sigma += 1            
        #     # step_size = step_lr * (sigma / sigmas[-1]) ** 2

        #     # for step in range(n_steps_each):
        #         d = utils.get_d_from_pos(pos, data.edge_index).unsqueeze(-1) # (num_edge, 1)
        #         noise = torch.randn_like(pos) * torch.sqrt(step_size * 2)
        #         score_d = scorenet.get_score(data, d, sigma) # (num_edge, 1)
        #         score_pos = self.convert_score_d(score_d, pos, data.edge_index, d)
        #         score_pos = utils.clip_norm(score_pos, limit=clip)
                
        #         out = self.p_sample(
        #                             model,
        #                             img,
        #                             t,
        #                             clip_denoised=clip_denoised,
        #                             denoised_fn=denoised_fn,
        #                             cond_fn=cond_fn,
        #                             model_kwargs=model_kwargs,
        #                         )

        #         pos = pos + step_size * score_pos + noise # (num_node, 3)
                
        #         pos_vecs.append(pos)

        # pos_vecs = torch.stack(pos_vecs, dim=0).view(cnt_sigma, n_steps_each, -1, 3) # (sigams, 100, num_node, 3)
        
        return data, pos_vecs

    def ConfGF_generator(self, data, config, pos_init=None):

        """
        The ConfGF generator that generates conformations using the score of atomic coordinates
        Return: 
            The generated conformation (pos_gen)
            Distance of the generated conformation (d_recover)
        """

        if pos_init is None:
            pos_init = torch.randn(data.num_nodes, 3).to(data.pos)
        data, pos_traj = self.position_Langevin_Dynamics(data, pos_init, self._model, self._model.sigmas.data.clone(), \
                                            n_steps_each=config.steps_pos, step_lr=config.step_lr_pos, \
                                            clip=config.clip, min_sigma=config.min_sigma)
        pos_gen = pos_traj[-1, -1] #(num_node, 3) fetch the lastest pos

        d_recover = utils.get_d_from_pos(pos_gen, data.edge_index) # (num_edges)

        data.pos_gen = pos_gen.to(data.pos)
        data.d_recover = d_recover.view(-1, 1).to(data.edge_length)
        return pos_gen, d_recover.view(-1), data, pos_traj


    def ConfGFDist_generator(self, data, config, embedder=utils.Embed3D(), pos_init=None):

        d = torch.rand(data.edge_index.size(1), 1, device=self.device) # (num_edge, 1)
        data, d_traj = self.distance_Langevin_Dynamics(data, d, self._model, self._model.sigmas.data.clone(), \
                                            n_steps_each=config.steps_d, step_lr=config.step_lr_d, \
                                            clip=config.clip, min_sigma=config.min_sigma)

        d_gen = d_traj[-1, -1].view(-1) # fetch the lastest d (num_edge, )
        if pos_init is None:
            pos_init = torch.randn(data.num_nodes, 3).to(data.pos)

        pos_traj, _ = embedder(d_gen.view(-1),
                         data.edge_index,
                         pos_init,
                         data.edge_order) # (num_steps, num_node, 3)
        pos_gen = pos_traj[-1] # (num_nodes, 3) 
        d_recover = utils.get_d_from_pos(pos_gen, data.edge_index) # (num_edges)

        data.pos_gen = pos_gen.to(data.pos)
        data.d_gen = d_gen.view(-1, 1).to(data.edge_length)
        data.d_recover = d_recover.view(-1, 1).to(data.edge_length)
        return pos_gen, d_recover.view(-1), data, pos_traj


    def generate_samples_from_smiles(self, smiles, generator, num_repeat=1, keep_traj=False, out_path=None):
        pass
    #     if keep_traj:
    #         assert num_repeat == 1, "to generate the trajectory of conformations, you must set num_repeat to 1"
        
    #     data = dataset.smiles_to_data(smiles)

    #     if data is None:
    #         raise ValueError('invalid smiles: %s' % smiles)

    #     return_data = copy.deepcopy(data)
    #     batch = utils.repeat_data(data, num_repeat).to(self.device)

    #     if generator == 'ConfGF':
    #         _, _, batch, pos_traj = self.ConfGF_generator(batch, self.config.test.gen) # (sigams, 100, num_node, 3)
    #     elif generator == 'ConfGFDist':
    #         embedder = utils.Embed3D(step_size=self.config.test.gen.dg_step_size, \
    #                                  num_steps=self.config.test.gen.dg_num_steps, \
    #                                  verbose=self.config.test.gen.verbose)
    #         _, _, batch, pos_traj = self.ConfGFDist_generator(batch, self.config.test.gen, embedder) # (num_steps, num_node, 3)
    #     else:
    #         raise NotImplementedError

    #     batch = batch.to('cpu').to_data_list()
    #     pos_traj = pos_traj.view(-1, 3).to('cpu')
    #     pos_traj_step = pos_traj.size(0) // return_data.num_nodes


    #     all_pos = []
    #     for i in range(len(batch)):
    #         all_pos.append(batch[i].pos_gen)
    #     return_data.pos_gen = torch.cat(all_pos, 0) # (num_repeat * num_node, 3)
    #     return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)
    #     if keep_traj:
    #         return_data.pos_traj = pos_traj
    #         return_data.num_pos_traj = torch.tensor([pos_traj_step], dtype=torch.long)

    #     if out_path is not None:
    #         with open(os.path.join(out_path, '%s_%s.pkl' % (generator, return_data.smiles)), "wb") as fout:
    #             pickle.dump(return_data, fout)
    #         logger.log('save generated %s samples to %s done!' % (generator, out_path))

    #     logger.log('pos generation of %s done' % return_data.smiles) 

    #     return return_data


    def generate_samples_from_testset(self, start, end, generator, num_repeat=None, out_path=None):
        
        test_set = self.test_set

        generate_start = time()

        all_data_list = []
        logger.log('len of all data: %d' % len(test_set))

        for i in tqdm(range(len(test_set))):
            if i < start or i >= end:
                continue
            return_data = copy.deepcopy(test_set[i])
            num_repeat_ = num_repeat if num_repeat is not None else 2 * test_set[i].num_pos_ref.item()
            batch = utils.repeat_data(test_set[i], num_repeat_).to(self.device)

            if generator == 'ConfGF':
                _, _, batch, _ = self.ConfGF_generator(batch, self.config.test.gen)
            elif generator == 'ConfGFDist':
                embedder = utils.Embed3D(step_size=self.config.test.gen.dg_step_size, \
                                         num_steps=self.config.test.gen.dg_num_steps, \
                                         verbose=self.config.test.gen.verbose)
                _, _, batch, _ = self.ConfGFDist_generator(batch, self.config.test.gen, embedder)
            else:
                raise NotImplementedError

            batch = batch.to('cpu').to_data_list()

            all_pos = []
            for i in range(len(batch)):
                all_pos.append(batch[i].pos_gen)
            return_data.pos_gen = torch.cat(all_pos, 0) # (num_repeat * num_node, 3)
            return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)
            all_data_list.append(return_data)

        if out_path is not None:
            with open(os.path.join(out_path, '%s_s%de%depoch%dmin_sig%.3f.pkl' % (generator, start, end, self.config.test.epoch, self.config.test.gen.min_sigma)), "wb") as fout:
                pickle.dump(all_data_list, fout)
            logger.log('save generated %s samples to %s done!' % (generator, out_path))
        logger.log('pos generation[%d-%d] done  |  Time: %.5f' % (start, end, time() - generate_start))  

        return all_data_list

