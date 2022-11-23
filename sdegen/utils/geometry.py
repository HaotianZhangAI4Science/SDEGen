import torch
#from confgf


def embed_3D(d_target, edge_index, init_pos, edge_order=None, alpha=0.5, mu=0, step_size=None, num_steps=None, verbose=0):
    assert torch.is_grad_enabled, '`embed_3D` requires gradients'
    step_size = 8.0 if step_size is None else step_size
    num_steps = 1000 if num_steps is None else num_steps
    pos_vecs = []

    d_target = d_target.view(-1)
    pos = init_pos.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pos], lr=step_size)

    if edge_order is not None:
        coef = alpha ** (edge_order.view(-1).float() - 1)
    else:
        coef = 1.0
    
    if mu > 0:
        noise = torch.randn_like(coef) * coef * mu + coef
        coef = torch.clamp_min(coef + noise, min=0)

    for i in range(num_steps):
        optimizer.zero_grad()
        d_new = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        #print(d_new)
        loss = (coef * ((d_target - d_new) ** 2)).sum()
        loss.backward()
        optimizer.step()
        pos_vecs.append(pos.detach().cpu())

    pos_vecs = torch.stack(pos_vecs, dim=0) # (num_steps, num_node, 3)

    if verbose:
        print('Embed 3D: AvgLoss %.6f' % (loss.item() / d_target.size(0)))

    return pos_vecs, loss.detach() / d_target.size(0)


class Embed3D(object):

    def __init__(self, alpha=0.5, mu=0, step_size=8.0, num_steps=1000, verbose=0):
        super().__init__()
        self.alpha = alpha
        self.mu = mu
        self.step_size = step_size
        self.num_steps = num_steps
        self.verbose = verbose

    def __call__(self, d_target, edge_index, init_pos, edge_order=None):
        return embed_3D(
            d_target, edge_index, init_pos, edge_order,
            alpha=self.alpha,
            mu=self.mu,
            step_size=self.step_size,
            num_steps=self.num_steps,
            verbose=self.verbose
        )

def get_d_from_pos(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1) # (num_edge)

def compute_euclidean_distances_matrix(X, Y):
    #X = X.double()
    #Y = Y.double()
    
    dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,    axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
    return dists**0.5

def inter_dis_to_ligand(bgp_pos, inter_dis, initial_xyz=None, device='cpu', step_size=8, num_steps=1000, verbose=0):
    num_bgl_node = inter_dis.shape[0] // bgp_pos.shape[0]
    if initial_xyz is None:
        initial_xyz = bgp_pos.mean(0) + torch.randn(num_bgl_node,3)  #help to converge
        initial_xyz = initial_xyz.to(device)
    xyz = initial_xyz.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([xyz], lr=step_size)
    pos_vecs = []
    for i in range(num_steps):
        optimizer.zero_grad()
        inter_new = torch.cdist(xyz, bgp_pos).reshape(-1,1)
        loss = ((inter_dis - inter_new)**2).sum()
        loss.backward()
        optimizer.step()
        pos_vecs.append(xyz.detach().cpu())
    
    if verbose:
        print('reconstruct interaction distance matrix: AvgLoss %.6f'%(loss.item()/inter_dis.shape[0]))

    return pos_vecs


def double_dis_to_ligand(bgp_pos, inter_dis, edge_index, ligand_dis, initial_xyz=None, device='cpu', step_size=8, num_steps=1000, verbose=0):

    num_bgl_node = inter_dis.shape[0] // bgp_pos.shape[0]
    if initial_xyz is None:
        initial_xyz = bgp_pos.mean(0) + torch.randn(num_bgl_node,3)
        initial_xyz = initial_xyz.to(device)
    xyz = initial_xyz.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([xyz], lr=step_size)
    pos_vecs = []
    for i in range(num_steps):
        optimizer.zero_grad()
        inter_new = torch.cdist(xyz, bgp_pos).reshape(-1,1)
        inter_loss = ((inter_dis - inter_new)**2).sum()
        iner_new = (xyz[edge_index[0]] - xyz[edge_index[1]]).norm(dim=-1)
        iner_loss = (iner_new - ligand_dis).sum()
        loss = iner_loss + inter_loss
        loss.backward()
        optimizer.step()
        pos_vecs.append(xyz.detach().cpu())
    
    if verbose:
        print('reconstruct interaction distance matrix: AvgLoss %.6f'%(loss.item()/inter_dis.shape[0]))
    
    return pos_vecs

