#coding: utf-8
from re import L
from time import time
from tqdm import tqdm
import os
import numpy as np
import pickle
import copy

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import DataLoader
from torch_scatter import scatter_add

from sdegen import utils, feats
from sdegen.data import dataset
from sdegen.utils import logger
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

class DefaultRunner(object):
    def __init__(self, train_set, val_set, test_set, model, optimizer, scheduler, gpus, config):
        self.train_set = train_set 
        self.val_set = val_set
        self.test_set = test_set
        self.gpus = gpus
        self.device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')            
        self.config = config
        
        self.batch_size = self.config.train.batch_size

        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler

        self.best_loss = 100.0
        self.start_epoch = 0

        if self.device.type == 'cuda':
            self._model = self._model.cuda(self.device)


    def save(self, checkpoint, epoch=None, var_list={}):

        state = {
            **var_list, 
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "config": self.config,
        }
        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
        torch.save(state, checkpoint)


    def load(self, checkpoint, epoch=None, load_optimizer=False, load_scheduler=False):
        
        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
        logger.log("Load checkpoint from %s" % checkpoint)

        state = torch.load(checkpoint, map_location=self.device)   
        self._model.load_state_dict(state["model"])
        #self._model.load_state_dict(state["model"], strict=False)
        self.best_loss = state['best_loss']
        self.start_epoch = state['cur_epoch'] + 1

        if load_optimizer:
            self._optimizer.load_state_dict(state["optimizer"])
            if self.device.type == 'cuda':
                for state in self._optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.device)

        if load_scheduler:
            self._scheduler.load_state_dict(state["scheduler"])

 
    @torch.no_grad()
    def evaluate(self, split, verbose=0):
        """
        Evaluate the model.
        Parameters:
            split (str): split to evaluate. Can be ``train``, ``val`` or ``test``.
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError('split should be either train, val, or test.')

        test_set = getattr(self, "%s_set" % split)
        dataloader = DataLoader(test_set, batch_size=self.config.train.batch_size, \
                                shuffle=False, num_workers=self.config.train.num_workers)
        model = self._model
        model.eval()
        # code here
        eval_start = time()
        eval_losses = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = batch.to(self.device)  

            loss = model(batch, self.device)
            loss = loss.mean()  
            eval_losses.append(loss.item())       
        average_loss = sum(eval_losses) / len(eval_losses)

        if verbose:
            logger.log('Evaluate %s Loss: %.5f | Time: %.5f' % (split, average_loss, time() - eval_start))
        return average_loss


    def train(self, verbose=1):
        train_start = time()
        
        num_epochs = self.config.train.epochs
        dataloader = DataLoader(self.train_set, 
                                batch_size=self.config.train.batch_size,
                                shuffle=self.config.train.shuffle, 
                                num_workers=self.config.train.num_workers)

        model = self._model  

        if self.config.train.ema: 
            ema = utils.EMA(self.config.train.ema_decay)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema.register(name, param.data)

        train_losses = []
        val_losses = []
        best_loss = self.best_loss
        start_epoch = self.start_epoch
        logger.log('start training...')
        
        for epoch in range(num_epochs):
            # train
            model.train()
            epoch_start = time()
            batch_losses = []
            batch_cnt = 0
            for batch in dataloader:
                batch_cnt += 1
                if self.device.type == "cuda":
                    batch = batch.to(self.device)  

                # logger.log(batch, 'batch shape')

                loss = model(data=batch, device=self.device)
                loss = loss.mean()
                if not loss.requires_grad:
                    raise RuntimeError("loss doesn't require grad")
                    
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self.config.train.ema:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            ema.update(name, param.data)

                batch_losses.append(loss.item())

                if batch_cnt % self.config.train.log_interval == 0 or (epoch==0 and batch_cnt <= 10):
                # if batch_cnt % self.config.train.log_interval == 0:
                    logger.log('Epoch: %d | Step: %d | loss: %.5f | Lr: %.7f' % \
                                (epoch + start_epoch, batch_cnt, batch_losses[-1], self._optimizer.param_groups[0]['lr']))


            average_loss = sum(batch_losses) / len(batch_losses)
            train_losses.append(average_loss)

            if verbose:
                logger.log('Epoch: %d | Train Loss: %.5f | Time: %.5f' % (epoch + start_epoch, average_loss, time() - epoch_start))

            # evaluate
            if self.config.train.eval:
                average_eval_loss = self.evaluate('val', verbose=1)
                val_losses.append(average_eval_loss)
            else:
                # use train loss as surrogate loss
                average_eval_loss = average_loss              
                val_losses.append(average_loss)

            if self.config.train.scheduler.type == "plateau":
                self._scheduler.step(average_eval_loss)
            else:
                self._scheduler.step()

            if val_losses[-1] < best_loss:
                best_loss = val_losses[-1]
                if self.config.train.save:
                    val_list = {
                                'cur_epoch': epoch + start_epoch,
                                'best_loss': best_loss,
                               }
                    self.save(self.config.train.save_path, epoch + start_epoch, val_list)
        
        self.best_loss = best_loss
        self.start_epoch = start_epoch + num_epochs               
        logger.log('optimization finished.')
        logger.log('Total time elapsed: %.5fs' % (time() - train_start))


    
    def SDE_PC_sampeler(self,data,SDE_model,sigma=25,snr=0.16, num_steps = 250,
                             eps=1e-3,num_langevin_steps=1):
        '''
        Args:
            data: GEOM-data, usually the batched data
            SDE_model: trained SDEGen model
            sigma: the diffusion coefficient chosen for adding noise
            snr: signal-to-noise ratio
            num_steps: the iteration number of the Euler-Maruyama solver
            eps: a small number used to prevent division by zero
            num_langevin_steps: the number of Langevin-MCMC steps at each time

        Returns: atomic distances

        '''
        d_vecs = []
        num_edge = len(data.edge_type)
        device = data.edge_index.device
        t = torch.ones(1,device=device)
        init_d = torch.randn(num_edge,device=device) * utils.marginal_prob_std(t,sigma,device=device)
        pos_init = torch.randn(data.num_nodes,3).to(data.pos)
        time_steps = torch.linspace(1., eps, num_steps, device=device)
        step_size = time_steps[0] - time_steps[1]
        d = init_d[:,None]
        batch_size = len(data.smiles)
        with torch.no_grad():
            for time_step in time_steps:
                batch_time_step = torch.ones(batch_size, device=device) * time_step
                g = utils.diffusion_coeff(time_step,sigma,device=device)
                # Corretor Step
                for i in range(num_langevin_steps):
                    grad = SDE_model.get_score(data, d, batch_time_step)
                    grad_norm = torch.norm(grad)         #||grad||
                    noise_norm = np.sqrt(d.shape[0])     #sqrt(d_dim)
                    langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                    d = d + (langevin_step_size * grad)[:,None] + (torch.sqrt(2*langevin_step_size)*torch.randn_like(d).squeeze(-1))[:,None]
                # Predictor Step
                mean_d = d + ((g**2) * SDE_model.get_score(data, d, batch_time_step) * step_size)[:,None]
                d = mean_d + (torch.sqrt(step_size) * g * torch.randn_like(d).squeeze(-1))[:,None]
        return mean_d


    def sde_generator(self,data, model,num_steps,num_langevin_steps):
        '''

        Args:
            data: GEOM-MOL data
            model: trained_SDE model
        Returns: data, pos_traj, d_gen, d_recover

        '''

        #get the generated distance
        d_gen = self.SDE_PC_sampeler(data,model,num_steps=num_steps,num_langevin_steps=num_langevin_steps)
        (d_gen - data.edge_length).max()
        #get the position traj
        num_nodes = len(data.atom_type)
        pos_init = torch.randn(num_nodes, 3).to(data.pos)
        embedder = utils.Embed3D()
        pos_traj ,_ = embedder(d_gen.view(-1),data.edge_index,pos_init,data.edge_order)
        pos_gen = pos_traj[-1]
        d_recover = utils.get_d_from_pos(pos_gen, data.edge_index)
        
        data.pos_gen = pos_gen.to(data.pos)
        data.d_recover = d_recover.view(-1, 1).to(data.edge_length)
        data.d_gen = d_gen.view(-1, 1).to(data.edge_length)
        return data, pos_traj, d_gen, d_recover


    def sde_generate_samples_from_mol(self,mol,num_repeat=1,out_path=None,file_name=None,num_steps=250,num_langevin_steps=2):
        data = feats.mol_to_data(mol)
        smiles = Chem.MolToSmiles(mol)
        '''
        generate sample based on smiles representation, this module mainly includes processing smiles and 
        '''

        if data is None:
            raise ValueError('invalid smiles: %s' % smiles)
        return_data = copy.deepcopy(data)
        batch = utils.repeat_data(data, num_repeat).to(self.device)
        batch,pos_traj,d_gen,d_recover = self.sde_generator(batch,self._model,num_steps, num_langevin_steps)
        batch = batch.to('cpu').to_data_list()
        all_pos = []
        for i in range(len(batch)):
            all_pos.append(batch[i].pos_gen)
        return_data.pos_gen = torch.cat(all_pos, 0)# (num_repeat * num_node, 3)
        return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)
        
        return_data.d_gen = d_gen

        if out_path is not None:

            m2 = return_data.rdmol
            AllChem.EmbedMolecule(m2)
            return_data.rdmol = m2

            pos_gen = return_data.pos_gen.view(-1,return_data.num_nodes,3)
            num_gen = pos_gen.size(0)
            gen_mol_list = []
            for i in range(num_gen):
                gen_mol = utils.set_mol_positions(return_data.rdmol,pos_gen[i])
                MMFFOptimizeMolecule(gen_mol)
                gen_mol_list.append(gen_mol)
            file_name = os.path.join(out_path,file_name)
            if file_name[-4:] == '.sdf':
                writer = Chem.SDWriter(file_name)
                for i in range(len(gen_mol_list)):
                    writer.write(gen_mol_list[i])
                writer.close()
            else:
                file = open(file_name, 'wb')
                pickle.dump(gen_mol_list,file)
                file.close()
        return return_data



