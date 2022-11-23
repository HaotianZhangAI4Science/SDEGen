import copy
import torch
from torch_geometric.data import Data
from torch_scatter import scatter
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from torch_geometric.utils import to_networkx
from sdegen import utils
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_sparse import coalesce


def rdmol_to_data(mol:Mol, smiles=None):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()

    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [utils.BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float32)

    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)
    data.nx = to_networkx(data, to_undirected=True)

    return data



def mol_to_data(mol):
    smiles = Chem.MolToSmiles(mol)        
    N = mol.GetNumAtoms()
    pos = torch.rand((N, 3), dtype=torch.float32)

    atomic_number = []
    aromatic = []

    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [utils.BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)
    
    transform = Compose([
        AddHigherOrderEdges(order=3),
        AddEdgeLength(),
        AddPlaceHolder(),
        AddEdgeName()
    ])
    
    return transform(data)

class AddHigherOrderEdges(object):

    def __init__(self, order, num_types=len(utils.BOND_TYPES)):
        super().__init__()
        self.order = order
        self.num_types = num_types

    def binarize(self, x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(self, adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        """
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    self.binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

        for i in range(2, order+1):
            adj_mats.append(self.binarize(adj_mats[i-1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order+1):
            order_mat += (adj_mats[i] - adj_mats[i-1]) * i

        return order_mat

    def __call__(self, data: Data):


        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = self.get_higher_order_adj_matrix(adj, self.order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)   # (N, N)
        type_highorder = torch.where(adj_order > 1, self.num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N) # modify data
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
        data.is_bond = (data.edge_type < self.num_types)
        assert (data.edge_index == edge_index_1).all()

        return data

class AddEdgeLength(object):

    def __call__(self, data: Data):

        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        data.edge_length = d
        return data    


# Add attribute placeholder for data object, so that we can use batch.to_data_list
class AddPlaceHolder(object):
    def __call__(self, data: Data):
        data.pos_gen = -1. * torch.ones_like(data.pos)
        data.d_gen = -1. * torch.ones_like(data.edge_length)
        data.d_recover = -1. * torch.ones_like(data.edge_length)
        return data


class AddEdgeName(object):

    def __init__(self, asymmetric=True):
        super().__init__()
        self.bonds = copy.deepcopy(utils.BOND_NAMES)
        self.bonds[len(utils.BOND_NAMES) + 1] = 'Angle'
        self.bonds[len(utils.BOND_NAMES) + 2] = 'Dihedral'
        self.asymmetric = asymmetric

    def __call__(self, data:Data):
        data.edge_name = []
        for i in range(data.edge_index.size(1)):
            tail = data.edge_index[0, i]
            head = data.edge_index[1, i]
            if self.asymmetric and tail >= head:
                data.edge_name.append('')
                continue
            tail_name = utils.get_atom_symbol(data.atom_type[tail].item())
            head_name = utils.get_atom_symbol(data.atom_type[head].item())
            name = '%s_%s_%s_%d_%d' % (
                self.bonds[data.edge_type[i].item()] if data.edge_type[i].item() in self.bonds else 'E'+str(data.edge_type[i].item()),
                tail_name,
                head_name,
                tail,
                head,
            )
            if hasattr(data, 'edge_length'):
                name += '_%.3f' % (data.edge_length[i].item())
            data.edge_name.append(name)
        return data        



class AddAngleDihedral(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def iter_angle_triplet(bond_mat):
        n_atoms = bond_mat.size(0)
        for j in range(n_atoms):
            for k in range(n_atoms):
                for l in range(n_atoms):
                    if bond_mat[j, k].item() == 0 or bond_mat[k, l].item() == 0: continue
                    if (j == k) or (k == l) or (j >= l): continue
                    yield(j, k, l)

    @staticmethod
    def iter_dihedral_quartet(bond_mat):
        n_atoms = bond_mat.size(0)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i >= j: continue
                if bond_mat[i,j].item() == 0:continue
                for k in range(n_atoms):
                    for l in range(n_atoms):
                        if (k in (i,j)) or (l in (i,j)): continue
                        if bond_mat[k,i].item() == 0 or bond_mat[l,j].item() == 0: continue
                        yield(k, i, j, l)

    def __call__(self, data:Data):
        N = data.num_nodes
        if 'is_bond' in data:
            bond_mat = to_dense_adj(data.edge_index, edge_attr=data.is_bond).long().squeeze(0) > 0
        else:
            bond_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).long().squeeze(0) > 0

        # Note: if the name of attribute contains `index`, it will automatically
        #       increases during batching.
        data.angle_index = torch.LongTensor(list(self.iter_angle_triplet(bond_mat))).t()
        data.dihedral_index = torch.LongTensor(list(self.iter_dihedral_quartet(bond_mat))).t()

        return data
