# python sde_sample.py --config_path bash_sde/drugs_ema.yml --start 0 --end 200
import argparse
import numpy as np
import random
import os
import pickle
import yaml
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.transforms import Compose
from tqdm import tqdm
from sdegen import model, runner, utils
from sdegen.data import dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--config_path', type=str, help='path of dataset', required=True)
    #parser.add_argument('--generator', type=str, help='type of generator [ConfGF, ConfGFDist]', required=True)
    parser.add_argument('--num_repeat', type=int, default=None, help='end idx of test generation')
    parser.add_argument('--start', type=int, default=-1, help='start idx of test generation')
    parser.add_argument('--end', type=int, default=-1, help='end idx of test generation')
    parser.add_argument('--smiles', type=str, default=None, help='smiles for generation')
    parser.add_argument('--seed', type=int, default=2021, help='overwrite config seed')

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.seed != 2021:
        config.train.seed = args.seed

    if config.test.output_path is not None:
        config.test.output_path = os.path.join(config.test.output_path, config.model.name)
        if not os.path.exists(config.test.output_path):
            os.makedirs(config.test.output_path)

    # check device
    gpus = list(filter(lambda x: x is not None, config.train.gpus))
    assert torch.cuda.device_count() >= len(gpus), 'do you set the gpus in config correctly?'
    device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')

    print("Let's use", len(gpus), "GPUs!")
    print("Using device %s as main device" % device)    
    config.train.device = device
    config.train.gpus = gpus

    print(config)

    # set random seed
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)        
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.benchmark = True
    print('set seed for random, numpy and torch')


    load_path = os.path.join(config.data.base_path, '%s_processed' % config.data.dataset)
    print('loading data from %s' % load_path)

    test_data = []
    train_data=[]
    val_data=[]
    if config.data.test_set is not None:
        with open(os.path.join(load_path, config.data.test_set), "rb") as fin:
            test_data = pickle.load(fin)             
    else:
        raise ValueError("do you set the test data ?")              
    transform = Compose([
        utils.AddHigherOrderEdges(order=config.model.order),
        utils.AddEdgeLength(),
        utils.AddPlaceHolder(),
        utils.AddEdgeName()
    ])

    test_data = dataset.GEOMDataset_PackedConf(data=test_data, transform=transform)
    print('len of test data: %d' % len(test_data))

    model = model.SDE(config)
    optimizer = None
    scheduler = None
    solver = runner.DefaultRunner(train_data, val_data, test_data, model, optimizer, scheduler, gpus, config)
    solver.load(config.test.init_checkpoint, epoch=config.test.epoch)
    if args.start != -1 and args.end != -1:
        solver.sde_generate_samples_from_testset(args.start, args.end,\
            num_repeat=args.num_repeat,
            out_path=config.test.output_path,
            file_name='sdegen_drugs_gen_{}_{}_3.pkl'.format(config.test.gen.num_euler_steps, \
                config.test.gen.num_langevin_steps))


