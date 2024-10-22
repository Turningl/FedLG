# -*- coding: utf-8 -*-
# @Author : liang
# @File : client.py


import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop
from functools import reduce
from operator import mul
from torch_geometric.loader import DataLoader
from utils.chemutils import extract_batch_data
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
        self.alg = args.alg  # fedavg, fedprox, fedsgd, fedlg, fedadam and fedchem algorithm can be chose
        # self.anti_noise_ability = args.anti_noise  # anti noise ability magnitude
        self.model = None  # The global model starts with None
        self.optimizer = None  # optimizer starts with None
        self.is_private = None  # None
        self.weight_denomaitor = None  # None
        self.mu = 0.1  # fedprox mu value
        self.warmup = True  # fedchem warm up value
        self.factor_ema = 0.8  # fedchem factor ema value
        self.tmp = 0.5  # fedchem tmp value
        self.Vks = None
        self.means = None

    # download model parameter
    def download(self, model):
        if self.device:
            self.model = model.to(self.device)
        else:
            self.model = model

    # set projection value
    def set_projection(self, Vks=None, means=None, is_private=None):
        self.Vks = Vks
        self.means = means
        self.is_private = is_private

    # set optimizer, here we set this in local update function directly
    def set_optimizer(self):
        pass

    # set local differential privacy value
    def set_local_differential_privacy(self, epsilon):
        self.epsilon = epsilon

    # if using differential privacy, set laplace noise following by Wu et al.,
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

    # local model update function
    def local_update(self):
        model = self.model.train()
        global_model = copy.deepcopy(model)

        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dataset = self.train[self.indices]
        train_dataset = DataLoader(dataset, batch_size=self.batch_size, drop_last=False)

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
            train_loss = 0

            for step, batch in tqdm(enumerate(train_dataset), total=len(train_dataset)):
                if dataset.related_title == 'DrugBank' or dataset.related_title == 'BIOSNAP' or dataset.related_title == 'CoCrystal':
                    mol1_batch, mol2_batch = extract_batch_data(train_dataset.dataset.mol_dataset, batch)
                    mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(
                        self.device)
                    y_pred = model(mol1_batch, mol2_batch)

                elif dataset.related_title == 'MoleculeNet' or dataset.related_title == 'LITPCBA':
                    mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                    y_pred = model(mol_data)

                loss = criterion(y_pred, y_true)

                if self.alg == 'FedProx':
                    proximal_term = 0
                    for w, w_global in zip(model.parameters(), global_model.parameters()):
                        proximal_term += self.mu / 2 * torch.norm(w - w_global, 2)
                    loss += proximal_term

                elif self.alg == 'FedChem':
                    if self.warmup:
                        with torch.no_grad():
                            if dataset.related_title == 'DrugBank' or dataset.related_title == 'BIOSNAP' or dataset.related_title == 'CoCrystal':
                                y_pred_global = global_model(mol1_batch, mol2_batch)
                            elif dataset.related_title == 'MoleculeNet' or dataset.related_title == 'LITPCBA':
                                y_pred_global = global_model(mol_data)
                        lossg_label = criterion(y_pred_global, y_true)
                        lossl_label = criterion(y_pred, y_true)

                        weightloss = lossl_label + torch.relu(lossl_label - lossg_label.detach())
                        if self.weight_denomaitor is None:
                            self.weight_denomaitor = weightloss.mean(dim=0, keepdim=True).detach()
                        else:
                            self.weight_denomaitor = self.factor_ema * self.weight_denomaitor + (
                                        1 - self.factor_ema) * weightloss.mean(dim=0, keepdim=True).detach()

                        loss = (1 - torch.exp(
                            -weightloss / (self.weight_denomaitor + 1e-7)) + 1e-7) ** self.tmp * lossl_label
                    else:
                        self.warmup += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print('Epoch is: %d, Train loss: %.4f' % ((epoch + 1), train_loss / self.dataset_size))

        # collecting the weights of local model
        updates = [weight.data for weight in model.state_dict().values()]

        # adding differently privacy by set laplace noise
        if self.dp:
            updates = [self.set_local_distribution(updates[i]) for i in range(len(updates))]

        return updates
