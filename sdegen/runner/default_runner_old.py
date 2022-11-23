import torch
from tqdm import tqdm 
from time import time
import os
from sdegen.utils import logger
import numpy as np
import pickle
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import DataLoader
from torch_scatter import scatter_add
from sdegen import utils, data, feats
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

        if load_scheduler:
            self._scheduler.load_state_dict(state["scheduler"])
            
        if load_optimizer:
            self._optimizer.load_state_dict(state["optimizer"])
            if self.device.type == 'cuda':
                for state in self._optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.device)



 
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


            average_loss = sum(batch_losses) / (len(batch_losses)+1)
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

    '''
    def sample_from_testset(self, start, end, generator, num_repeat, out_path):
        sample_list = self._model.generate_samples_from_testset(start, end, \
                                                        generator, self.test_set, num_repeat, out_path)
        return sample_list

    def sample_from_smiles(self, smiles, generator, test_set, num_repeat, keep_traj, out_path):
        sample_list = self._model.generate_samples_from_smiles(smiles, generator, \
                                                        num_repeat, test_set, keep_traj, out_path)
        return sample_list
    '''


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


    def generate_samples_from_smiles(self, smiles, generator, num_repeat=1, keep_traj=False, out_path=None,file_name=None,use_FF=True,format='mol'):
        
        if keep_traj:
            assert num_repeat == 1, "to generate the trajectory of conformations, you must set num_repeat to 1"
        
        data = feats.smiles_to_data(smiles)

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
            if format=='mol':
                m2 = return_data.rdmol
                AllChem.EmbedMolecule(m2)
                return_data.rdmol = m2
                pos_gen = return_data.pos_gen.view(-1,return_data.num_nodes,3)
                num_gen = pos_gen.size(0)
                gen_mol_list = []
                for i in range(num_gen):
                    gen_mol = utils.set_rdmol_positions(return_data.rdmol,pos_gen[i])
                    if use_FF == True:
                        MMFFOptimizeMolecule(gen_mol)
                    gen_mol_list.append(gen_mol)
                file = open(os.path.join(out_path,file_name), 'wb')
                pickle.dump(gen_mol_list,file)
                file.close()
                print('save generated %s samples to %s done!' % (generator, out_path))
                print('pos generation of %s done' % return_data.smiles) 
            if format == 'sdf':
                m2 = return_data.rdmol
                AllChem.EmbedMolecule(m2)
                pos_gen = return_data.pos_gen.view(-1,return_data.num_nodes,3)
                num_gen = pos_gen.size(0)
                gen_mol_list = []
                for i in range(num_gen):
                    gen_mol = utils.set_rdmol_positions(m2,pos_gen[i])
                    if use_FF == True:
                        MMFFOptimizeMolecule(gen_mol)
                    gen_mol_list.append(gen_mol)
                file_name = os.path.join(out_path,file_name)
                writer = Chem.SDWriter(file_name+'.sdf')
                for mol_idx in range(len(gen_mol_list)):
                    writer.write(gen_mol_list[mol_idx])
                writer.close()
                print('pos generation of %s done' % len(gen_mol_list)) 

        return return_data

    def extract_samples_from_testset(self, start, end, generator, num_repeat=None, out_path=None, extract_path=None):
        
        test_set = self.test_set

        generate_start = time()

        all_data_list = []
        print('len of all data: %d' % len(test_set))

        files = open('task3_confVAE_mol.txt','r')
        smile_names = files.readlines()

        smile_names = [name[:-2] for name in smile_names]
        # print(smile_names, 'smiles names')

        for i in tqdm(range(len(test_set))):
            if i < start or i >= end:
                continue
            return_data = copy.deepcopy(test_set[i])
            if return_data.smiles not in smile_names:
                # print('skip')
                continue

            all_data_list.append(return_data)

        if out_path is not None:
            with open(os.path.join(out_path, '%s_s%de%depoch%dmin_sig%.3f_extract_samples.pkl' % (generator, start, end, self.config.test.epoch, self.config.test.gen.min_sigma)), "wb") as fout:
                pickle.dump(all_data_list, fout)
            print('save generated %s samples to %s done!' % (generator, out_path))
        # print('pos generation[%d-%d] done  |  Time: %.5f' % (start, end, time() - generate_start))  
        
        return all_data_list
        

    def generate_samples_from_testset(self, start, end, generator, num_repeat=None, out_path=None):
        
        test_set = self.test_set

        generate_start = time()

        all_data_list = []
        print('len of all data: %d' % len(test_set))

        for i in tqdm(range(len(test_set))):
            if i < start or i >= end:
                continue
            return_data = copy.deepcopy(test_set[i])

            num_repeat_ = num_repeat if num_repeat is not None else 2 * test_set[i].num_pos_ref.item()
            batch = utils.repeat_data(test_set[i], num_repeat_).to(self.device)

            print(return_data.smiles, 'return smile')
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
    
    def SDE_PC_distance_sampeler(self,data,SDE_model,sigma=25,snr=0.16, num_steps = 250, 
                             eps=1e-3,num_langevin_steps=1):
        d_vecs = []
        num_edge = len(data.edge_type)
        device = SDE_model.device
        t = torch.ones(1,device=device)
        init_d = torch.randn(num_edge,device=device) * utils.marginal_prob_std(t,sigma,device=device)
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
                #d_vecs.append(d)
        return mean_d
    
    def sde_convert_score_d(self,score_d, pos, edge_index, edge_length):
        dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])   # (num_edge, 3)
        score_pos = scatter_add(dd_dr * score_d[:,None], edge_index[0], dim=0)
        return score_pos

    def SDE_PC_pos_sampeler(self,data,SDE_model,sigma=25,snr=0.16, step_lr=0.01, num_steps = 250, 
                            eps=1e-3,num_langevin_steps=10):
        pos_vecs = []
        device = SDE_model.device
        t = torch.ones(1,device=device)
        init_pos = torch.randn(data.num_nodes,3,device=device) * utils.marginal_prob_std(t,sigma,device=device)
        time_steps = torch.linspace(1., eps, num_steps, device=device)
        step_size = time_steps[0] - time_steps[1]
        pos = init_pos
        batch_size = len(data.smiles)
        with torch.no_grad():        
            for time_step in time_steps:
                batch_time_step = torch.ones(batch_size, device=device) * time_step
                g = utils.diffusion_coeff(time_step,sigma,device=device)
                # Corretor Step
                for i in range(num_langevin_steps):

                    d = utils.get_d_from_pos(pos, data.edge_index).unsqueeze(-1) 
                    score_d = SDE_model.get_score(data, d, batch_time_step) # (num_edge, 1)
                    score_pos = self.sde_convert_score_d(score_d, pos, data.edge_index, d)
                    score_pos = utils.clip_norm(score_pos, limit=1000)

                    grad_norm = torch.norm(score_pos)         #||grad||
                    noise_norm = np.sqrt(pos.shape[0] * 3)     #sqrt(d_dim)
                    langevin_step_size = 2 * step_lr* (snr * noise_norm / grad_norm)**2 * step_size**2   #Annealed Langevin Dynamics
                    #print("pos:{},langevin_step_size:{},score_pos:{}".format(pos.shape,langevin_step_size.shape,score_pos.shape))
                    pos = pos + (langevin_step_size * score_pos) + (torch.sqrt(2*langevin_step_size)*torch.randn_like(pos))
                #Predictor Step 
                d = utils.get_d_from_pos(pos, data.edge_index).unsqueeze(-1) 
                score_d = SDE_model.get_score(data, d, batch_time_step) # (num_edge, 1)
                score_pos = self.sde_convert_score_d(score_d, pos, data.edge_index, d)
                score_pos = utils.clip_norm(score_pos, limit=1000)
                mean_pos = pos + ((g**2) * score_pos * step_size)
                pos = mean_pos + (torch.sqrt(step_size) * g * torch.randn_like(pos))
                pos_vecs.append(mean_pos)
        return pos_vecs
    
    def sde_generator(self,data, model,num_steps,num_langevin_steps,step_lr=0.2):
        '''
        input: data & SDE_model
        return data, d_recover.view(-1), data, pos_traj
        '''
        #get the generated distance
        d_gen = self.SDE_PC_distance_sampeler(data,model,step_lr=step_lr,num_steps=num_steps,num_langevin_steps=num_langevin_steps)
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
    
    def sde_generate_samples_from_testset(self,start,end,num_repeat=None,out_path=None,file_name='sample_from_testset'):
        '''
        >> start&end: suppose the length of testset is 200, choosing the start=0, end=100 means 
            we sample data from testset[0:100].
        >> num_repeat: suppose one of the datas contained in the testset has 61 conformation, 
            num_repeat =2 means we generate 2 conformation for evaluation task, but if num_repeat=None,
            this means we are under the 2*num_pos_ref mode.
        >> out_path is self-explanatory.

        This function we use the packed testset to generate num_repeat times conformations
        than the reference conformations, and this is just because we want to compute the 
        COV and MAT metrics for sde method on a specific testset.
        For user who wants to generate conformations merely through throwing a smiles of a 
        molecule, we recommand he/she to use the sde_generate_samples_from_smiles.
        '''

        test_set = self.test_set
        generate_start = time()
        all_data_list = []
        print('len of all data : %d' % len(test_set))
        SDE_model = self._model
        for i in tqdm(range(len(test_set))):
            if i < start or i >= end:
                continue
            return_data = copy.deepcopy(test_set[i])
            num_repeat_ = num_repeat if num_repeat is not None else 2 * test_set[i].num_pos_ref.item()
            batch = utils.repeat_data(test_set[i], num_repeat_).to(self.device)
            embedder = utils.Embed3D(step_size=self.config.test.gen.dg_step_size, \
                                         num_steps=self.config.test.gen.dg_num_steps, \
                                         verbose=self.config.test.gen.verbose)

            batch,pos_traj,d_gen,d_recover = self.sde_generator(batch,SDE_model,self.config.test.gen.num_euler_steps,self.config.test.gen.num_langevin_steps)

            batch = batch.to('cpu').to_data_list()

            all_pos = []
            for i in range(len(batch)):
                all_pos.append(batch[i].pos_gen)
            return_data.pos_gen = torch.cat(all_pos, 0)# (num_repeat * num_node, 3)
            return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)
            all_data_list.append(return_data)
            return_data.d_gen = d_gen
        if out_path is not None:
            with open(os.path.join(out_path, file_name), "wb") as fout:
                pickle.dump(all_data_list, fout)
            print('save generated %s samples to %s done!' % ('sde', out_path))
        print('pos generation[%d-%d] done  |  Time: %.5f' % (start, end, time() - generate_start))  

        return all_data_list

    def sde_generate_samples_from_smiles(self,smiles,num_repeat=1,out_path=None,file_name=None,num_steps=250,num_langevin_steps=2,useFF=False,pos_show=False,format='sdf'):
        '''
        choose pos_show must satisfy the condition -- num_repeat=1
        '''
        data = feats.smiles_to_data(smiles)
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
        if pos_show == True:
            return return_data,pos_traj
        else: 
            if out_path is not None:
                if format == 'mol':
                    m2 = return_data.rdmol
                    AllChem.EmbedMolecule(m2)
                    return_data.rdmol = m2

                    pos_gen = return_data.pos_gen.view(-1,return_data.num_nodes,3)
                    num_gen = pos_gen.size(0)
                    gen_mol_list = []
                    for i in range(num_gen):
                        gen_mol = utils.set_rdmol_positions(return_data.rdmol,pos_gen[i])
                        if useFF == True:
                            MMFFOptimizeMolecule(gen_mol)
                        gen_mol_list.append(gen_mol)
                    file = open(os.path.join(out_path,file_name), 'wb')
                    pickle.dump(gen_mol_list,file)
                    file.close()
                if format == 'sdf':
                    m2 = return_data.rdmol
                    AllChem.EmbedMolecule(m2)
                    pos_gen = return_data.pos_gen.view(-1,return_data.num_nodes,3)
                    num_gen = pos_gen.size(0)
                    gen_mol_list = []
                    for i in range(num_gen):
                        gen_mol = utils.set_rdmol_positions(m2,pos_gen[i])
                        if useFF == True:
                            MMFFOptimizeMolecule(gen_mol)
                        gen_mol_list.append(gen_mol)
                    file_name = os.path.join(out_path,file_name)
                    writer = Chem.SDWriter(file_name+'.sdf')
                    for mol_idx in range(len(gen_mol_list)):
                        writer.write(gen_mol_list[mol_idx])
                    writer.close()
            return return_data

    def sde_pos_generator(self,data,step_lr=0.01,num_steps=250,num_langevin_steps=10):
        SDE_model = self._model
        pos_traj = self.SDE_PC_pos_sampeler(data,self._model,num_langevin_steps=num_langevin_steps, \
            num_steps=num_steps, step_lr=step_lr)
        pos_gen = pos_traj[-1]
        d_recover = utils.get_d_from_pos(pos_gen, data.edge_index)
        data.pos_gen = pos_gen
        data.d_recover = d_recover
        return pos_gen, d_recover.view(-1), data, pos_traj