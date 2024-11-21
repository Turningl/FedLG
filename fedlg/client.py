# -*- coding: utf-8 -*-
# @Author : liang
# @File : client.py


import ast
import copy
import torch
import random
import numpy as np
import scipy as sp
import configparser
import torch.nn as nn
import hyperopt as hy
from torch.optim import Adam, SGD, RMSprop
from torch_geometric.loader import DataLoader
from utils.chemutils import extract_batch_data
from utils.lanczos import Lanczos
from tqdm import tqdm


class SimulatedDatabase:
    def __init__(self, train, indices, args):
        super(SimulatedDatabase, self).__init__()
        self.train = train  # train dataset for local training
        self.indices = indices  # index value obtained by splitting algorithm
        self.dataset_size = len(indices)  # training dataset size
        self.batch_size = args.batch_size  # batch size
        self.local_round = args.local_round  # local training round
        self.dp = args.dp  # differential privacy value
        # self.grad_norm = args.grad_norm  # grad norm value
        self.device = args.device  # device is cuda or cpu
        self.lr = args.lr  # learning rate size
        self.weight_decay = args.weight_decay # weight decay size
        self.clip = args.clip # clip value
        self.random_seed = args.seed  # random seed value
        self.constant = args.constant  # constant
        self.alg = args.alg  # fedavg, fedprox, fedsgd, fedlg, fedadam and flit algorithm can be chose
        # self.anti_noise_ability = args.anti_noise  # anti noise ability magnitude
        self.model = None  # The global model starts with None
        self.optimizer = None  # optimizer starts with None
        self.is_private = None  # None
        self.weight_denomaitor = None  # None
        self.mu = 0.1  # fedprox mu value
        self.warmup = True  # flit warm up value
        self.factor_ema = 0.8  # flit factor ema value
        self.tmp = 0.5  # flit tmp value

        self.proj_dims = args.proj_dims
        self.lanczos_iter = args.lanczos_iter

        self.state_dict_key = None
        self.shape_vars = None
        self.model_states = None
        self.means = None

        self.init = args.init
        self.max_step = args.max_step
        self.updates = None
        self.rstate = None
        self.train_dataset = None # train dataset
        self.valid_dataset = None # valid dataset

    # download model parameter
    def download(self, model):
        if self.device:
            self.model = model.to(self.device)
        else:
            self.model = model
        self.state_dict_key = self.model.state_dict().keys()

    # update communication efficiency information value
    def update_comm_optimization(self, model_states=None, means=None, participant=None):
        self.model_states = model_states
        self.means = means
        self.is_participant = participant

    # set optimizer, here we set this in local update function directly
    def set_optimizer(self):
        pass

    # set local differential privacy value
    def set_local_differential_privacy(self, epsilon):
        self.epsilon = epsilon

    # if using differential privacy, set laplace noise following by Wu et al.
    # see https://github.com/wuch15/FedPerGNN/blob/main/run.py
    def set_local_distribution(self, updates):
        # if self.epsilon >= 1.0: self.clip = 0.5

        if self.dp:
            scale = torch.tensor(self.constant * self.lr * self.clip / np.sqrt(self.batch_size) / self.epsilon)
            # print('the scale is:', scale)
            if self.random_seed:
                random.seed(self.random_seed)
                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)
                torch.cuda.manual_seed(self.random_seed)
                torch.cuda.manual_seed_all(self.random_seed)

            laplace_dist = torch.distributions.Laplace(0, scale)
            laplace_noise = laplace_dist.sample(updates.shape)

            if self.device:
                laplace_noise = laplace_noise.to(self.device)

            updates += laplace_noise.type(updates.dtype)
            return updates
        return updates

    def __standardize(self, M):
        n, m = M.shape
        if m == 1:
            return M, torch.zeros(n, device=self.device)
        mean = torch.mul(M, torch.ones((m, 1), dtype=torch.float32, device=self.device)) / m

        return M - mean, mean.flatten()

    def __eigen_by_lanczos(self, mat, lanczos_iter):
        Tri_Mat, Orth_Mat = Lanczos(mat, lanczos_iter)  # getting a tridiagonal matrix T and an orthogonal matrix V

        T_evals_, T_evecs_ = np.linalg.eig(Tri_Mat)  # calculating the eigenvalues and eigenvectors of a tridiagonal matrix
        T_evals, T_evecs = sp.sparse.linalg.eigsh(Tri_Mat, k=2, which='LM')

        idx = T_evals.argsort()[-1: -(self.proj_dims + 1): -1]  # finding the index of the largest element in the eigenvalue array T evals

        proj_eigenvecs = np.dot(Orth_Mat.T, T_evecs[:, idx])  # the eigenvector corresponding to the maximum eigenvalue is projected into the new eigenspace

        if proj_eigenvecs.size >= 2:
            proj_eigenvecs = torch.from_numpy(proj_eigenvecs).to(torch.float32).to(self.device).squeeze()
        else:
            proj_eigenvecs = torch.from_numpy(proj_eigenvecs).to(torch.float32).to(self.device).squeeze(0)

        return proj_eigenvecs

    # local model update function
    def local_update(self):
        model = self.model.train()
        w_model = copy.deepcopy(model)

        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dataset = self.train[self.indices]
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

        train_dataset = DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=False)
        valid_dataset = DataLoader(self.valid_dataset, batch_size=self.batch_size, drop_last=False)

        criterion = None
        if dataset.dataset_name in dataset.dataset_names['regression']:
            criterion = nn.MSELoss()
        elif dataset.dataset_name in dataset.dataset_names['classification']:
            criterion = nn.BCEWithLogitsLoss()

        # if self.dp:
        #     privacy_engine = PrivacyEngine(secure_mode=False)
        #     model, optimizer, train_loader = privacy_engine.make_private(module=model,
        #                                                                  optimizer=optimizer,
        #                                                                  data_loader=data_loader,
        #                                                                  noise_multiplier=self.budget_accountant.noise_multiplier,
        #                                                                  max_grad_norm=self.grad_norm)

        # local training process
        for epoch in range(self.local_round):
            train_losses, val_losses = 0, 0

            model = model.train()  # training
            for step, batch in enumerate(train_dataset):
                if dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                    mol1_batch, mol2_batch = extract_batch_data(train_dataset.dataset.dataset.mol_dataset, batch)
                    mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(
                        self.device)
                    y_pred = model(mol1_batch, mol2_batch)

                elif dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                    mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                    y_pred = model(mol_data)

                train_loss = criterion(y_pred, y_true)

                if self.alg == 'FedProx':
                    proximal_term = 0
                    for w, w_global in zip(model.parameters(), w_model.parameters()):
                        proximal_term += self.mu / 2 * torch.norm(w - w_global, 2)
                    train_loss += proximal_term

                elif self.alg == 'FLIT':
                    if self.warmup:
                        with torch.no_grad():
                            if dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                                y_pred_global = w_model(mol1_batch, mol2_batch)
                            elif dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                                y_pred_global = w_model(mol_data)

                        lossg_label = criterion(y_pred_global, y_true)
                        lossl_label = criterion(y_pred, y_true)
                        weightloss = lossl_label + torch.relu(lossl_label - lossg_label.detach())

                        if self.weight_denomaitor is None:
                            self.weight_denomaitor = weightloss.mean(dim=0, keepdim=True).detach()
                        else:
                            self.weight_denomaitor = self.factor_ema * self.weight_denomaitor + (
                                        1 - self.factor_ema) * weightloss.mean(dim=0, keepdim=True).detach()

                        train_loss = (1 - torch.exp(
                            -weightloss / (self.weight_denomaitor + 1e-7)) + 1e-7) ** self.tmp * lossl_label
                    else:
                        self.warmup += 1

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_losses += train_loss.item() * len(batch)

            model.eval()  # eval
            with torch.no_grad():
                for step, batch in enumerate(valid_dataset):
                    if dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                        mol1_batch, mol2_batch = extract_batch_data(valid_dataset.dataset.dataset.mol_dataset, batch)
                        mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(self.device)
                        y_pred = model(mol1_batch, mol2_batch)

                    elif dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                        mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                        y_pred = model(mol_data)

                    val_loss = criterion(y_pred, y_true)
                    val_losses += val_loss.item() * len(batch)

            print('Epoch is: %d, Train loss: %.4f, Valid loss: %.4f' % ((epoch + 1), train_losses / int(0.9 * len(dataset)), val_losses / (len(dataset) - int(0.9 * len(dataset)))))

        # collecting the weights of local model
        updates = [weight.data for weight in model.state_dict().values()]

        # adding differently privacy by set laplace noise
        if self.dp:
            updates = [self.set_local_distribution(updates[i]) for i in range(len(updates))]

            self.updates = updates
            self.shape_vars = [var.shape for var in self.updates]

        # fetch Bayesian optimization
        if self.model_states and self.is_participant:

            # execute bayesian optimization process
            best_pbounds = self.bayesian_optimization()

            # get project updates with the best parameter from bayesian optimization
            _, proj_updates = self.fetch_model_update(best_pbounds)

            return proj_updates
        return updates

    # bayesian optimization process
    def bayesian_optimization(self):
        pbounds = {
            'lanczos_iter': self.hy_parameter_setting(label='lanczos_iter'),  # set lanczos iter number
        }

        # print("Parameter bounds:")
        # for param, value in pbounds.items():
        #     print(f"{param}: {value}")

        print('Using Bayesian Optimization with TPE')

        # set tree-structured parzen estimator for bayesian optimization
        algo = hy.partial(hy.tpe.suggest, n_startup_jobs=self.init)

        if self.random_seed:
            self.rstate = np.random.RandomState(self.random_seed)

        trials = hy.Trials()

        best = hy.fmin(fn=self.model_prediction,
                       space=pbounds,
                       algo=algo,
                       max_evals=self.max_step,
                       trials=trials,
                       rstate=self.rstate,
                       )
        return best

    def fetch_model_update(self, params):
        # get update of flatten
        updates = [update.flatten() for update in self.updates]
        proj_updates = [0] * len(self.model_states)

        # execute lanczos and project algorithms
        for i in range(len(self.model_states)):
            proj_eigenvecs = self.__eigen_by_lanczos(self.model_states[i].T, lanczos_iter=params['lanczos_iter'])
            proj_updates[i] = (torch.mul(proj_eigenvecs, torch.dot(proj_eigenvecs.T, (updates[i] - self.means[i]))) + self.means[i]).reshape(self.shape_vars[i])

        # loading model state dict
        model = self.model.eval()
        model.load_state_dict(dict(zip(self.state_dict_key, proj_updates)))

        return model, proj_updates

    @torch.no_grad()
    def model_prediction(self, params):
        dataset = self.train[self.indices]
        val_dataset = DataLoader(self.valid_dataset, batch_size=self.batch_size, drop_last=True)

        criterion = None
        if dataset.dataset_name in dataset.dataset_names['regression']:
            criterion = nn.MSELoss()
        elif dataset.dataset_name in dataset.dataset_names['classification']:
            criterion = nn.BCEWithLogitsLoss()

        model, _ = self.fetch_model_update(params)

        # eval process
        losses = 0
        for step, batch in enumerate(val_dataset):
            if dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                mol1_batch, mol2_batch = extract_batch_data(val_dataset.dataset.dataset.mol_dataset, batch)
                mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(self.device)
                y_pred = model(mol1_batch, mol2_batch)

            elif dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                y_pred = model(mol_data)

            loss = criterion(y_pred, y_true)
            losses += loss.item() * len(batch)

                # print('Epoch is: %d, loss: %.4f' % ((epoch + 1), len(self.valid_dataset)))

        return losses / len(self.valid_dataset)

    def hy_parameter_setting(self, label):
        config = configparser.RawConfigParser()
        config.read('./parameters/{}_range.in'.format('optimization'))

        try:
            val = ast.literal_eval(config.get('range', label))
        except Exception as e:
            print(f"Error reading config for {label}: {e}")
            return None

        # set hyperopt parameter
        if type(val) is list:
            if label == 'lanczos_iter':
                return hy.hp.choice(label, np.arange(val[0], val[1], 2).tolist())
            else:
                print('Unsupported parameter. Check configuration or label name.')
        else:
            print('Invalid data format in config file. Expected a list of two values.')

        return None
