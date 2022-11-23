import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import networkx as nx
from rdkit.Chem import rdMolAlign as MA
import numpy as np
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule


from rdkit.Chem.rdchem import BondType as BT
BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}
from rdkit.Chem import PeriodicTable as PT
from rdkit.Chem.rdchem import Mol, GetPeriodicTable

def reconstruct_xyz(d_target, edge_index, init_pos, edge_order=None, alpha=0.5, mu=0, step_size=None, num_steps=None, verbose=0):
    assert torch.is_grad_enabled, 'the optimization procedure needs gradient to iterate'
    step_size = 8.0 if step_size is None else step_size
    num_steps = 1000 if num_steps is None else num_steps
    pos_vecs = []

    d_target = d_target.view(-1)
    pos = init_pos.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pos], lr=step_size)

    #different hop contributes to different loss 
    if edge_order is not None:
        coef = alpha ** (edge_order.view(-1).float() - 1)
    else:
        coef = 1.0
    
    if mu>0:
        noise = torch.randn_like(coef) * coef * mu + coef
        noise = torch.clamp_min(coef+noise, min=0)
    
    for i in range(num_steps):
        optimizer.zero_grad()
        d_new = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        loss = (coef * (d_target - d_new)**2).sum()
        loss.backward()
        optimizer.step()
        pos_vecs.append(pos.detach().cpu())
        avg_loss = loss.item() / d_target.size(0)
        #if verbose & (i%10 == 0):
        #    print('Reconstruction Loss: AvgLoss %.6f' % avg_loss)
    pos_vecs = torch.stack(pos_vecs, dim=0)
    avg_loss = loss.item() / d_target.size(0)
    if verbose:
        print('Reconstruction Loss: AvgLoss %.6f' % avg_loss)
    
    return pos_vecs, avg_loss

class Reconstruct_xyz(object):
    
    def __init__(self, alpha=0.5, mu=0, step_size=8.0, num_steps=1000, verbose=0):
        super().__init__()
        self.alpha = alpha
        self.mu = mu
        self.step_size = step_size
        self.num_stpes = num_steps
        self.verbose = verbose
    
    def __call__(self, d_target, edge_index, init_pos, edge_order=None):
        return reconstruct_xyz(
            d_target, edge_index, init_pos, edge_order, 
            alpha=self.alpha, 
            mu = self.mu, 
            step_size = self.step_size,
            num_steps = self.num_stpes,
            verbose = self.verbose
        )

def get_d_from_pos(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

def set_mol_positions(mol, pos):
    mol = copy.deepcopy(mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol 

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def mmff_energy(mol):
    energy = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')).CalcEnergy()
    return energy

def get_torsion_angles(mol):
    torsions_list = []
    G = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i)
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        l = list(sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append(
            (n0[0], e[0], e[1], n1[0])
        )
    return torsions_list

def get_atom_symbol(atomic_number):
    return PT.GetElementSymbol(GetPeriodicTable(), atomic_number)

def GetBestRMSD(probe, ref):
    probe = Chem.RemoveHs(probe)
    ref = Chem.RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd

def computeRMSD(xyz_1, xyz_2):
    if type(xyz_1) == np.ndarray:
        rmsd = np.sqrt(np.mean(np.sum((xyz_1-xyz_2)**2, axis=-1)))
    if type(xyz_1) == torch.Tensor:
        rmsd = torch.sqrt((((xyz_1 - xyz_2)**2).sum(axis=-1)).mean())
    return rmsd

def optimize_mol(mol):
    mol_ = copy.deepcopy(mol)
    MMFFOptimizeMolecule(mol_)
    return mol_