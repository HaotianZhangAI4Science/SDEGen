import torch
import warnings
import pickle
#scheduler is used in the validation stage 
import torch
from torch_geometric.data import Data, Batch
import copy
from collections import defaultdict
from rdkit import Chem
import numpy as np
import random
from collections import defaultdict

class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma    
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma, last_epoch, verbose)
    def get_lr(self) -> float:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch==0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]
    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


def get_optimizer(config, model):
    if config.type == "Adam":
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = config.lr,
            weight_decay = config.weight_decay)
    else:
        raise NotImplementedError("Optimizer not supported: %s" % config.type)


def get_scheduler(config, optimizer):
    if config.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.factor,
            patience=config.patience
        )
    elif config.train.scheduler == 'expmin':
        return ExponentialLR_with_minLr(
            optimizer, 
            gamma = config.factor,
            min_lr=config.min_lr,)
    else:
        raise NotImplementedError('Scheduler not supported: %s' % config.type)


def write_txt(list,file):
    f = open(file,'w')
    for line in list:
        f.write(line + '\n')
    f.close()
    print('txt file saved at {}'.format(f))


def read_txt(file):
    f = open(file,'r')
    text = f.read().splitlines()
    f.close()
    print('read file from {}'.format(file))
    return text


def write_pkl(list,file):
    with open(file,'wb') as f:
        pickle.dump(list,f)
        print('pkl file saved at {}'.format(file))


def read_pkl(file):
    with open(file,'rb') as f:
        data = pickle.load(f)
    return data

def packedmol_wsmiles(mol_list):
    dict_smi = defaultdict(list)
    #产生packed_data
    for i in tqdm(range(len(mol_list))):
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_list[i]))
        dict_smi[smiles].append(mol_list[i])
    return dict_smi

def write_sdf(mol_list, file):
    writer = Chem.SDWriter(file)
    for i in mol_list:
        writer.write(mol_list)
    writer.close()
    
class EMA():
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.decay) * x + self.decay * self.shadow[name]
        self.shadow[name] = new_average.clone()


def dict_record(ls):
    dic = {}
    for item in ls:
        dic[item] = dic.get(item,0) +1 
    return dic

def sort_mol_list_via_smiles(mol_list):

    dic = defaultdict(list)
    smiles_list = []
    for i in range(len(smiles_list)):
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_list[i]))
        dic[smiles].append(mol_list[i])
        smiles_list.append(smiles)
    return dic, smiles_list

def repeat_data(data: Data, num_repeat) -> Batch:
    datas = [copy.deepcopy(data) for i in range(num_repeat)]
    return Batch.from_data_list(datas)


def repeat_batch(batch: Batch, num_repeat) -> Batch:
    datas = batch.to_data_list()
    new_data = []
    for i in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)

def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom

def set_seed(seed):
    # set random seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
