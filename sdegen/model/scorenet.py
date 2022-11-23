import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from sdegen import utils, layers
from time import time
import copy
from tqdm import tqdm

class DistanceScoreMatch(torch.nn.Module):

    def __init__(self, config):
        super(DistanceScoreMatch, self).__init__()
        self.config = config
        self.anneal_power = self.config.train.anneal_power
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.noise_type = self.config.model.noise_type

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

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)

        # time step embedding for continous SDE
        # if self.config.scheme.time_continuous:
        #     self.t_embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.hidden_dim),
        #                             nn.Linear(self.hidden_dim, self.hidden_dim))
        #     self.dense1 = nn.Linear(self.hidden_dim, 1)

        """
        Techniques from "Improved Techniques for Training Score-Based Generative Models"
        1. Choose sigma1 to be as large as the maximum Euclidean distance between all pairs of training data points.
        2. Choose sigmas as a geometric progression with common ratio gamma, where a specific equation of CDF is satisfied.
        3. Parameterize the Noise Conditional Score Networks with f_theta_sigma(x) =  f_theta(x) / sigma
        """

    def forward(self, data, device=None):
        """
        Input:
            data: torch geometric batched data object
        Output:
            loss
        """
        # a workaround to get the current device, we assume all tensors in a model are on the same device.
        self.device = self.sigmas.device

        data = self.extend_graph(data, self.order)
        data = self.get_distance(data)

        # print(data, 'data show ')

        assert data.edge_index.size(1) == data.edge_length.size(0)
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]  
        # print(node2graph.shape, edge2graph.shape)      

        # sample noise level
        noise_level = torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device) # (num_graph)
        used_sigmas = self.sigmas[noise_level] # (num_graph)
        # ensure that one graph shares with the same noise level
        used_sigmas = used_sigmas[edge2graph].unsqueeze(-1) # (num_edge, 1)

        # perturb
        d = data.edge_length # (num_edge, 1)

        if self.noise_type == 'symmetry':
            num_nodes = scatter_add(torch.ones(data.num_nodes, dtype=torch.long, device=self.device), node2graph) # (num_graph)
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

        assert d_noise.shape == d.shape
        perturbed_d = d + d_noise * used_sigmas   
        # perturbed_d = torch.clamp(perturbed_d, min=0.1, max=float('inf'))    # distances must be greater than 0
        # get target, origin_d minus perturbed_d
        target = -1 / (used_sigmas ** 2) * (perturbed_d - d) # (num_edge, 1)

        # estimate scores
        node_attr = self.node_emb(data.atom_type) # (num_node, hidden)
        edge_attr = self.edge_emb(data.edge_type) # (num_edge, hidden)

        d_emb = self.input_mlp(perturbed_d) # (num_edge, hidden)

        # if self.config.scheme.time_continuous:
        #     # time embedding
        #     t_embedding = self.t_embed(noise_level)             #(batch_dim, hidden_dim) = (128, 256)
        #     t_embedding = self.dense1(t_embedding)    #(batch_dim, 1) = (128,1)
        #     # print(d_emb.shape, t_embedding[edge2graph].shape)
        #     d_emb = d_emb + t_embedding[edge2graph]

        edge_attr = d_emb * edge_attr # (num_edge, hidden)

        output = self.model(data, node_attr, edge_attr)
        h_row, h_col = output["node_feature"][data.edge_index[0]], output["node_feature"][data.edge_index[1]] # (num_edge, hidden)

        distance_feature = torch.cat([h_row*h_col, edge_attr], dim=-1) # (num_edge, 2 * hidden)
        scores = self.output_mlp(distance_feature) # (num_edge, 1)
        scores = scores * (1. / used_sigmas) # f_theta_sigma(x) =  f_theta(x) / sigma, (num_edge, 1)

        target = target.view(-1) # (num_edge)
        scores = scores.view(-1) # (num_edge)
        loss =  0.5 * ((scores - target) ** 2) * (used_sigmas.squeeze(-1) ** self.anneal_power) # (num_edge)
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
                                   n_steps_each=100, step_lr=0.00002, 
                                   clip=1000, min_sigma=0):
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
        """
        scorenet.eval()
        pos_vecs = []
        pos = pos_init
        cnt_sigma = 0
        for i, sigma in tqdm(enumerate(sigmas), total=sigmas.size(0), desc="Sampling positions"):
            if sigma < min_sigma:
                break
            cnt_sigma += 1            
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for step in range(n_steps_each):
                d = utils.get_d_from_pos(pos, data.edge_index).unsqueeze(-1) # (num_edge, 1)

                noise = torch.randn_like(pos) * torch.sqrt(step_size * 2)
                score_d = scorenet.get_score(data, d, sigma) # (num_edge, 1)
                score_pos = self.convert_score_d(score_d, pos, data.edge_index, d)
                score_pos = utils.clip_norm(score_pos, limit=clip)

                pos = pos + step_size * score_pos + noise # (num_node, 3)
                pos_vecs.append(pos)

        pos_vecs = torch.stack(pos_vecs, dim=0).view(cnt_sigma, n_steps_each, -1, 3) # (sigams, 100, num_node, 3)
        
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
        
        if keep_traj:
            assert num_repeat == 1, "to generate the trajectory of conformations, you must set num_repeat to 1"
        
        data = dataset.smiles_to_data(smiles)

        if data is None:
            raise ValueError('invalid smiles: %s' % smiles)

        return_data = copy.deepcopy(data)
        batch = utils.repeat_data(data, num_repeat).to(self.device)

        if generator == 'ConfGF':
            _, _, batch, pos_traj = self.ConfGF_generator(batch, self.config.test.gen) # (sigams, 100, num_node, 3)
        elif generator == 'ConfGFDist':
            embedder = utils.Embed3D(step_size=self.config.test.gen.dg_step_size, \
                                     num_steps=self.config.test.gen.dg_num_steps, \
                                     verbose=self.config.test.gen.verbose)
            _, _, batch, pos_traj = self.ConfGFDist_generator(batch, self.config.test.gen, embedder) # (num_steps, num_node, 3)
        else:
            raise NotImplementedError

        batch = batch.to('cpu').to_data_list()
        pos_traj = pos_traj.view(-1, 3).to('cpu')
        pos_traj_step = pos_traj.size(0) // return_data.num_nodes


        all_pos = []
        for i in range(len(batch)):
            all_pos.append(batch[i].pos_gen)
        return_data.pos_gen = torch.cat(all_pos, 0) # (num_repeat * num_node, 3)
        return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)
        if keep_traj:
            return_data.pos_traj = pos_traj
            return_data.num_pos_traj = torch.tensor([pos_traj_step], dtype=torch.long)

        if out_path is not None:
            with open(os.path.join(out_path, '%s_%s.pkl' % (generator, return_data.smiles)), "wb") as fout:
                pickle.dump(return_data, fout)
            print('save generated %s samples to %s done!' % (generator, out_path))

        print('pos generation of %s done' % return_data.smiles) 

        return return_data


    def generate_samples_from_testset(self, start, end, generator, 
                                    test_set,
                                    num_repeat=None, 
                                    out_path=None,):
        
        # test_set = self.test_set

        generate_start = time()

        all_data_list = []
        print('len of all data: %d' % len(test_set))

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
            print('save generated %s samples to %s done!' % (generator, out_path))
        print('pos generation[%d-%d] done  |  Time: %.5f' % (start, end, time() - generate_start))  

        return all_data_list
