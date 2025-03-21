# -*- coding: utf-8 -*-
# @Author : liang
# @File : server.py
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy as sp
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils.lanczos import Lanczos
from utils.saveutils import print_init_alg_info, print_average_info
from utils.chemutils import extract_batch_data


def add_weights(num_vars, model_state, agg_model_state):
    # for i in range(num_vars):
    #     if not len(agg_model_state):
    #         a = torch.unsqueeze(model_state[i], 0)
    #     else:
    #         b = torch.cat([agg_model_state[i], torch.unsqueeze(model_state[i], 0)], 0)
    return [torch.unsqueeze(model_state[i], 0)
            if not len(agg_model_state) else torch.cat([agg_model_state[i],
                                                        torch.unsqueeze(model_state[i], 0)], 0)
            for i in range(num_vars)]


class FedAvg:
    def __init__(self):
        self.__model_state = []
        self.num_vars = None
        self.shape_vars = None

    def aggregate(self, model_state):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)
        update_model_state = [state.flatten() for state in model_state]
        self.__model_state = add_weights(self.num_vars, update_model_state, self.__model_state)

    def average(self):
        mean_updates = [torch.mean(self.__model_state[i], 0).reshape(self.shape_vars[i])
                        for i in range(self.num_vars)]
        self.__model_state = []
        return mean_updates


class FedAdam:
    def __init__(self, beta1, beta2, epsilon, lr, device):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr
        self.device = device

        self.__model_state = []
        self.num_vars = None
        self.shape_vars = None
        self.m = []
        self.v = []
        self.t = 0

    def aggregate(self, model_state):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)
        update_model_state = [state.flatten() for state in model_state]

        self.m = [torch.zeros(var.shape).to(self.device) for var in update_model_state]
        self.v = [torch.zeros(var.shape).to(self.device) for var in update_model_state]

        # if not self.__model_state:
        #     self.__model_state = [torch.zeros_like(update) for update in update_model_state]

        self.__model_state = add_weights(self.num_vars, update_model_state, self.__model_state)

        # Update moment estimates
        self.t += 1
        for i in range(self.num_vars):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * self.__model_state[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (self.__model_state[i] ** 2)

    def average(self):
        mean_updates = []
        for i in range(self.num_vars):
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            update = self.lr * m_hat / (
                        torch.sqrt(v_hat) + torch.from_numpy(self.epsilon).resize(len(self.epsilon), 1).to(self.device))
            mean_updates.append(torch.mean(update, 0).reshape(self.shape_vars[i]))

        self.__model_state = []
        return mean_updates


class FedSGD:
    def __init__(self, lr=0.01):
        self.__model_state = []
        self.num_vars = None
        self.shape_vars = None
        self.lr = lr

    def aggregate(self, model_state):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)
        update_model_state = [state.flatten() for state in model_state]
        self.__model_state = add_weights(self.num_vars, update_model_state, self.__model_state)

    def average(self):
        mean_updates = []
        for i in range(self.num_vars):
            # Calculate mean of updates
            mean_update = self.__model_state[i] / self.__model_state[i].shape[0]
            # Parameter update
            update = self.lr * mean_update
            mean_updates.append(torch.mean(update, 0).reshape(self.shape_vars[i]))

        self.__model_state = []
        return mean_updates


class FedDF:
    def __init__(self, model, lr, weight_decay, batch_size, device):
        self.dataset = None
        self.__model_state_sum  = []
        self.__valid_idx_sum = []

        self.batch_size = batch_size
        self.device = device

        self.model = model
        self.global_model = copy.deepcopy(model).to(self.device)

        self.state_dict_key = self.model.state_dict().keys()
        self.optimizer = Adam(self.global_model.parameters(), lr=lr, weight_decay=weight_decay)

    def update_teacher_model(self, model_state):
        teacher_model = copy.deepcopy(self.model)
        teacher_model.load_state_dict(dict(zip(self.state_dict_key, model_state)))
        return teacher_model.to(self.device)

    def aggregate(self, model_state):
        model_state, dataset, valid_idx = model_state

        self.dataset = dataset
        self.__valid_idx_sum.extend(valid_idx)
        self.__model_state_sum.append(model_state)

    def average(self):
        self.dataloader = self.dataset[self.__valid_idx_sum]
        train_dataset = DataLoader(self.dataloader, batch_size=self.batch_size, drop_last=False)

        teacher_models = [self.update_teacher_model(teacher_model_state) for teacher_model_state in self.__model_state_sum]

        self.global_model.train()
        for step, batch in enumerate(train_dataset):
            if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                mol1_batch, mol2_batch = extract_batch_data(train_dataset.dataset.mol_dataset, batch)
                mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(
                    self.device)
                y_pred = self.global_model(mol1_batch, mol2_batch)

                teacher_logits = sum([model(mol1_batch, mol2_batch).detach() for model in teacher_models]) / len(teacher_models)

            elif self.dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                y_pred = self.global_model(mol_data)

                teacher_logits = sum([model(mol_data).detach() for model in teacher_models]) / len(teacher_models)

            loss = self.divergence(y_pred, teacher_logits)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.dataset = None
        self.__valid_idx_sum = []
        self.__model_state_sum = []

        mean_model_state = [weight.data for weight in self.global_model.state_dict().values()]

        return mean_model_state

    def divergence(self, student_logits, teacher_logits):
        divergence = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )  # forward KL
        return divergence


class AdaFedSemi:
    def __init__(self, model, lr, weight_decay, batch_size, device,
                 alpha=0.5, max_participation=1.0, min_participation=0.1,
                 max_confidence=0.99, min_confidence=0.8):

        self.dataset = None
        self.__unlabeled_data_sum = []
        self.__model_state_sum = []

        self.model = model
        self.global_model = copy.deepcopy(model).to(device)
        self.state_dict_key = self.model.state_dict().keys()
        self.optimizer = Adam(self.global_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.batch_size = batch_size
        self.alpha = alpha
        self.max_participation = max_participation
        self.min_participation = min_participation
        self.max_confidence = max_confidence
        self.min_confidence = min_confidence
        self.teacher_model = None

        self.participation_actions = np.linspace(min_participation, max_participation, 10)
        self.confidence_actions = np.linspace(min_confidence, max_confidence, 10)
        self.participation_rewards = np.zeros_like(self.participation_actions)
        self.confidence_rewards = np.zeros_like(self.confidence_actions)
        self.participation_probabilities = np.ones_like(self.participation_actions) / len(self.participation_actions)
        self.confidence_probabilities = np.ones_like(self.confidence_actions) / len(self.confidence_actions)

        self.num_vars = None
        self.shape_vars = None

    def select_participation_fraction(self):
        return np.random.choice(self.participation_actions, p=self.participation_probabilities)

    def select_confidence_threshold(self):
        return np.random.choice(self.confidence_actions, p=self.confidence_probabilities)

    def update_teacher_model(self, model_state):
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.load_state_dict(dict(zip(self.state_dict_key, model_state)))
        self.teacher_model.to(self.device)

    def generate_pseudo_labels(self, unlabeled_data):
        self.dataloader = self.dataset[unlabeled_data]
        train_dataset = DataLoader(self.dataloader, batch_size=self.batch_size, drop_last=False)

        self.teacher_model.eval()
        pseudo_labels = []
        pseudo_confidences = []

        with torch.no_grad():
            for step, batch in enumerate(train_dataset):
                if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                    mol1_batch, mol2_batch = extract_batch_data(train_dataset.dataset.mol_dataset, batch)
                    mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(
                        self.device)
                    y_pred = self.teacher_model(mol1_batch, mol2_batch)

                elif self.dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                    mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                    y_pred = self.teacher_model(mol_data)

                probabilities = torch.sigmoid(y_pred).squeeze()
                predicted_labels = (probabilities > 0.5).int()
                confidences = torch.max(torch.stack([probabilities, 1 - probabilities], dim=1), dim=1).values

                pseudo_labels.extend(predicted_labels.cpu().numpy())
                pseudo_confidences.extend(confidences.cpu().numpy())

        return pseudo_labels, pseudo_confidences

    def aggregate(self, model_state):
        model_state, dataset, unlabeled_data = model_state

        self.dataset = dataset
        self.__unlabeled_data_sum.extend(unlabeled_data)
        self.__model_state_sum.append(model_state)

        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)

    def __average(self, client_num, participation_fraction):
        selected_clients = int(client_num * participation_fraction)
        selected_models = np.random.choice(range(client_num), selected_clients, replace=False)

        if len(selected_models) == 0:
            model_states = [self.__model_state_sum[i] for i in range(0, 1)]
        else:
            model_states = [self.__model_state_sum[i] for i in selected_models]

        aggregated_state = {}
        for model_state in model_states:
            for key, value in enumerate(model_state):
                if key not in aggregated_state:
                    aggregated_state[key] = value
                else:
                    aggregated_state[key] += value
            for key in aggregated_state:
                aggregated_state[key] /= len(model_states)

        mean_model_state = [weight for weight in aggregated_state.values()]

        return mean_model_state

    def average(self):
        participation_fraction = self.select_participation_fraction()
        confidence_threshold = self.select_confidence_threshold()

        mean_states = self.__average(len(self.__model_state_sum), participation_fraction)
        self.global_model.load_state_dict(dict(zip(self.state_dict_key, mean_states)))

        self.update_teacher_model(mean_states)

        pseudo_labels, confidences = self.generate_pseudo_labels(self.__unlabeled_data_sum)
        high_confidence_indices = [i for i, conf in enumerate(confidences) if conf >= confidence_threshold]
        high_confidence_data = [self.__unlabeled_data_sum[i] for i in high_confidence_indices]
        high_confidence_labels = [pseudo_labels[i] for i in high_confidence_indices]
        self.dataloader = self.dataset[high_confidence_data]

        high_confidence_graphs = []
        for data, label in zip(self.dataloader, high_confidence_labels):
            if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                new_data = Data(smi1 = data.smi1, smi2=data.smi2, y=torch.FloatTensor([label]).unsqueeze(0))
            elif self.dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                new_data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=torch.FloatTensor([label]).unsqueeze(0))
            high_confidence_graphs.append(new_data)
        dataloader = DataLoader(high_confidence_graphs, batch_size=self.batch_size, drop_last=False)

        criterion = None
        if self.dataset.dataset_name in self.dataset.dataset_names['regression']:
            criterion = nn.MSELoss()
        elif self.dataset.dataset_name in self.dataset.dataset_names['classification']:
            criterion = nn.BCEWithLogitsLoss()

        self.global_model.train()
        for step, batch in enumerate(dataloader):
            if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                mol1_batch, mol2_batch = extract_batch_data(self.dataset.mol_dataset, batch)
                mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(
                    self.device)
                y_pred = self.global_model(mol1_batch, mol2_batch)

            elif self.dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                y_pred = self.global_model(mol_data)

            loss = criterion(y_pred, y_true)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_mab_probabilities()

        self.dataset = None
        self.__unlabeled_data_sum = []
        self.__model_state_sum = []

        mean_model_state = [weight.data for weight in self.global_model.state_dict().values()]

        return mean_model_state

    def update_mab_probabilities(self):

        self.participation_probabilities = np.random.rand(len(self.participation_actions)) / self.participation_probabilities.sum()
        # self.participation_probabilities /= self.participation_probabilities.sum()
        self.confidence_probabilities = np.random.rand(len(self.confidence_actions)) / self.confidence_probabilities.sum()
        # self.confidence_probabilities /= self.confidence_probabilities.sum()


class SelectiveFD:
    def __init__(self, model, lr, weight_decay, batch_size, device):
        self.model = model
        self.global_model = copy.deepcopy(model).to(device)
        self.state_dict_key = self.model.state_dict().keys()
        self.device = device
        self.batch_size = batch_size
        self.server_selector_threshold = 0.5
        self.lr = lr
        self.weight_decay = weight_decay

        self.dataset = None
        self.__proxy_idx_sum = []
        self.__model_state_sum = []
        self.__new_model_state = []

        self.shape_vars = None
        self.num_vars = None

    def aggregate(self, model_state):
        model_state, dataset, proxy_label_idxs = model_state

        self.dataset = dataset
        self.__proxy_idx_sum.extend(proxy_label_idxs)
        self.__model_state_sum.append(model_state)

        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)

    def proxy_pred_softlabel(self, model, proxy_samples):
        dataset = self.dataset[proxy_samples]
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, drop_last=False)

        model.eval()
        batch_softlabel = []

        with torch.no_grad():
            for step, batch in enumerate(train_dataloader):
                if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                    mol1_batch, mol2_batch = extract_batch_data(train_dataloader.dataset.mol_dataset, batch)
                    mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(
                        self.device)
                    y_pred = model(mol1_batch, mol2_batch)

                elif self.dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                    mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                    y_pred = model(mol_data)

                batch_softlabel.extend((y_pred * torch.reshape(batch.y, (-1, 1)).to(self.device)).cpu().detach().numpy())

        return model, batch_softlabel

    def client_distillation(self, model, proxy_samples, batch_softlabel):
        dataset = self.dataset[proxy_samples]
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        distillation_graphs = []
        for data, label in zip(dataset, batch_softlabel):
            if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                new_data = Data(smi1=data.smi1, smi2=data.smi2, y=torch.FloatTensor(label).unsqueeze(0))
            elif self.dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                new_data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=torch.FloatTensor(label).unsqueeze(0))
            distillation_graphs.append(new_data)
        new_dataloader = DataLoader(distillation_graphs, batch_size=self.batch_size, drop_last=False)

        criterion = None
        if self.dataset.dataset_name in self.dataset.dataset_names['regression']:
            criterion = nn.MSELoss()
        elif self.dataset.dataset_name in self.dataset.dataset_names['classification']:
            criterion = nn.BCEWithLogitsLoss()

        model.train()
        for step, batch in enumerate(new_dataloader):
            if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                mol1_batch, mol2_batch = extract_batch_data(self.dataset.mol_dataset, batch)
                mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(
                    self.device)
                y_pred = model(mol1_batch, mol2_batch)

            elif self.dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                y_pred = model(mol_data)

            loss = criterion(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model.state_dict().values()

    def average(self):
        for model_state in self.__model_state_sum:
            local_model = copy.deepcopy(self.model).to(self.device)
            local_model.load_state_dict(dict(zip(self.state_dict_key, model_state)))

            model, batch_softlabel = self.proxy_pred_softlabel(local_model, self.__proxy_idx_sum)
            new_model = self.client_distillation(local_model, self.__proxy_idx_sum, batch_softlabel)

            new_model_state = [weight.data for weight in new_model]
            update_model_state = [state.flatten() for state in new_model_state]
            self.__new_model_state = add_weights(self.num_vars, update_model_state, self.__new_model_state)

        mean_updates = [torch.mean(self.__new_model_state[i], 0).reshape(self.shape_vars[i])
                        for i in range(self.num_vars)]

        self.__proxy_idx_sum = []
        self.__model_state_sum = []
        self.__new_model_state = []

        return mean_updates

class FedLG:
    def __init__(self, proj_dims, lanczos_iter, device='cuda', comm_optimization=None):
        self.proj_dims = proj_dims
        self.lanczos_iter = lanczos_iter
        self.device = device

        self.__num_open_access_databases = 0
        self.__open_access_model_state = []
        self.__open_access_eps = []

        self.__num_privacy_institution_databases = 0
        self.__private_institutional_model_state = []
        self.__private_institutional_eps = []

        self.num_vars = None
        self.shape_vars = None

        self.comm_optimization = comm_optimization
        self.open_access_model_states = None
        self.means = None

    def aggregate(self, eps, model_state, is_open_access):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)
        update_model_state = [state.flatten() for state in model_state]

        if is_open_access:
            self.__num_open_access_databases += 1
            self.__open_access_eps.append(eps)
            self.__open_access_model_state = add_weights(self.num_vars, update_model_state, self.__open_access_model_state)

        else:
            self.__num_privacy_institution_databases += 1
            self.__private_institutional_eps.append(eps)
            self.__private_institutional_model_state = add_weights(self.num_vars, update_model_state, self.__private_institutional_model_state)

    def __standardize(self, M):
        n, m = M.shape
        if m == 1:
            return M, torch.zeros(n, device=self.device)
        mean = torch.mul(M, torch.ones((m, 1), dtype=torch.float32, device=self.device)) / m

        return M - mean, mean.flatten()

    def __eigen_by_lanczos(self, mat):
        Tri_Mat, Orth_Mat = Lanczos(mat, self.lanczos_iter)  # getting a tridiagonal matrix T and an orthogonal matrix V

        # T_evals_, T_evecs_ = np.linalg.eig(T)  # calculating the eigenvalues and eigenvectors of a tridiagonal matrix
        T_evals, T_evecs = sp.sparse.linalg.eigsh(Tri_Mat, k=2, which='LM')

        idx = T_evals.argsort()[-1: -(self.proj_dims + 1): -1]  # finding the index of the largest element in the eigenvalue array T evals

        proj_eigenvecs = np.dot(Orth_Mat.T, T_evecs[:, idx])  # the eigenvector corresponding to the maximum eigenvalue is projected into the new eigenspace

        if proj_eigenvecs.size >= 2:
            proj_eigenvecs = torch.from_numpy(proj_eigenvecs).to(torch.float32).to(self.device).squeeze()
        else:
            proj_eigenvecs = torch.from_numpy(proj_eigenvecs).to(torch.float32).to(self.device).squeeze(0)

        return proj_eigenvecs

    def __lanczos_graph_proj(self):
        if len(self.__private_institutional_model_state):
            private_institutional_weights = (torch.Tensor(self.__private_institutional_eps) / sum(self.__private_institutional_eps)).view(self.__num_privacy_institution_databases, 1).to(self.device)
            open_access_weights = (torch.Tensor(self.__open_access_eps) / sum(self.__open_access_eps)).view(self.__num_open_access_databases, 1).to(self.device)

            mean_priv_model_state = [
                torch.sum(self.__private_institutional_model_state[i].to(self.device) * private_institutional_weights, 0) / torch.sum(private_institutional_weights) for i in range(self.num_vars)]
            mean_pub_model_state = [
                torch.sum(self.__open_access_model_state[i].to(self.device) * open_access_weights, 0) / torch.sum(open_access_weights) for i in range(self.num_vars)]
            mean_proj_priv_model_state = [0] * self.num_vars
            mean_model_state = [0] * self.num_vars

            for i in range(self.num_vars):
                open_access_model_state, mean = self.__standardize(self.__open_access_model_state[i].T)
                proj_eigenvecs = self.__eigen_by_lanczos(open_access_model_state.T)
                mean_proj_priv_model_state[i] = torch.mul(proj_eigenvecs, torch.dot(proj_eigenvecs.T, (mean_priv_model_state[i] - mean))) + mean
                mean_model_state[i] = ((mean_proj_priv_model_state[i] * sum(self.__private_institutional_eps) + mean_pub_model_state[i] * sum(self.__open_access_eps)) / sum(self.__private_institutional_eps + self.__open_access_eps)).reshape( self.shape_vars[i])

            return mean_model_state

    def __lanczos_graph_proj_communication_optimization(self, warmup):
        if len(self.__private_institutional_model_state):
            private_institutional_weights = (torch.Tensor(self.__private_institutional_eps) / sum(self.__private_institutional_eps)).view(self.__num_privacy_institution_databases, 1).to(self.device)
            open_access_weights = (torch.Tensor(self.__open_access_eps) / sum(self.__open_access_eps)).view(self.__num_open_access_databases, 1).to(self.device)

            mean_priv_model_state = [
                torch.sum(self.__private_institutional_model_state[i].to(self.device) * private_institutional_weights, 0) / torch.sum(private_institutional_weights) for i in range(self.num_vars)]
            mean_pub_model_state = [
                torch.sum(self.__open_access_model_state[i].to(self.device) * open_access_weights, 0) / torch.sum(open_access_weights) for i in range(self.num_vars)]
            mean_proj_priv_model_state = [0] * self.num_vars
            mean_model_state = [0] * self.num_vars

            open_access_model_states = []
            means = []

            if warmup:
                for i in range(self.num_vars):
                    open_access_model_state, mean = self.__standardize(self.__open_access_model_state[i].T)
                    proj_eigenvecs = self.__eigen_by_lanczos(open_access_model_state.T)
                    mean_proj_priv_model_state[i] = torch.mul(proj_eigenvecs, torch.dot(proj_eigenvecs.T, (mean_priv_model_state[i] - mean))) + mean
                    mean_model_state[i] = ((mean_proj_priv_model_state[i] * sum(self.__private_institutional_eps) + mean_pub_model_state[i] * sum(self.__open_access_eps))
                                           / sum(self.__private_institutional_eps + self.__open_access_eps)).reshape(self.shape_vars[i])

                    open_access_model_states.append(open_access_model_state)
                    means.append(mean)
            else:
                for i in range(self.num_vars):
                    mean_model_state[i] = ((mean_proj_priv_model_state[i] * sum(self.__private_institutional_eps) + mean_pub_model_state[i] * sum(self.__open_access_eps))
                                           / sum(self.__private_institutional_eps + self.__open_access_eps)).reshape(self.shape_vars[i])
                    open_access_model_state, mean = self.__standardize(self.__open_access_model_state[i].T)
                    open_access_model_states.append(open_access_model_state)
                    means.append(mean)

            self.open_access_model_states = open_access_model_states
            self.means = means
            return mean_model_state

    def average(self):
        # mean_updates = None
        if self.comm_optimization:
            mean_updates = self.__lanczos_graph_proj_communication_optimization(warmup=(self.open_access_model_states is None))
        else:
            mean_updates = self.__lanczos_graph_proj()

        self.__num_open_access_databases = 0
        self.__num_privacy_institution_databases = 0

        self.__open_access_model_state = []
        self.__private_institutional_model_state = []

        self.__open_access_eps = []
        self.__private_institutional_eps = []

        return mean_updates

class GlobalServer:
    def __init__(self, model, args):
        super(GlobalServer, self).__init__()
        self.num_clients = args.num_clients
        self.device = args.device

        self.model = model
        self.state_dict_key = self.model.state_dict().keys()

        self.proj_dims = args.proj_dims
        self.lanczos_iter = args.lanczos_iter

        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.lr = args.lr
        self.comm_optimization = args.comm_optimization
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        # self.warmup = args.warmup

        self.num_vars = None
        self.shape_vars = None
        self.__alg = None
        self.open_access = None
        self.__epsilons = None

        self.alpha = args.alpha
        self.max_participation = args.max_participation
        self.min_participation = args.min_participation
        self.max_confidence = args.max_confidence
        self.min_confidence = args.min_confidence

    def set_open_access_database(self, epsilons):
        self.__epsilons = epsilons
        threshold = np.max(self.__epsilons)

        self.open_access = list(np.where(np.array(self.__epsilons) >= threshold)[0])

    def init_global_model(self):
        return self.model

    def fetch_comm_optimization(self):
        return self.__alg.open_access_model_states, self.__alg.means

    @print_init_alg_info('Initialization')
    def init_alg(self, alg):
        if alg == 'FedAvg' or alg == 'FedProx' or alg == 'FLIT':
            self.__alg = FedAvg()
        elif alg == 'FedSGD':
            self.__alg = FedSGD(self.lr)
        elif alg == 'FedAdam':
            self.__alg = FedAdam(beta1=self.beta1, beta2=self.beta2, epsilon=self.__epsilons, lr=self.lr, device=self.device)
        elif alg == 'FedLG':
            self.__alg = FedLG(self.proj_dims, self.lanczos_iter, self.device, self.comm_optimization)
        elif alg == 'FedDF':
            self.__alg = FedDF(self.model, self.lr, self.weight_decay, self.batch_size, self.device)
        elif alg == 'AdaFedSemi':
            self.__alg = AdaFedSemi(self.model, self.lr, self.weight_decay, self.batch_size, self.device, self.alpha,
                                    self.max_participation, self.min_participation, self.max_confidence, self.min_confidence)
        elif alg == 'SelectiveFD':
            self.__alg = SelectiveFD(self.model, self.lr, self.weight_decay, self.batch_size, self.device)
        else:
            raise ValueError('\nSelect an algorithm to get the aggregated model.\n')


        print('\n{} algorithm.\n'.format(str(alg)))

    def aggregate(self, participant, model_state, alg):
        if alg == 'FedLG':
            self.__alg.aggregate(self.__epsilons[participant], model_state,
                                 is_open_access=True if (participant in self.open_access) else False)
        elif alg == 'FedAvg' or alg == 'FedProx' or alg == 'FLIT':
            self.__alg.aggregate(model_state)
        elif alg == 'FedSGD':
            self.__alg.aggregate(model_state)
        elif alg == 'FedAdam':
            self.__alg.aggregate(model_state)
        elif alg == 'FedDF':
            self.__alg.aggregate(model_state)
        elif alg == 'AdaFedSemi':
            self.__alg.aggregate(model_state)
        elif alg == 'SelectiveFD':
            self.__alg.aggregate(model_state)
        else:
            raise ValueError('\nSelect an algorithm to get the aggregated model.\n')

    @print_average_info
    def update(self):
        mean_state = self.__alg.average()
        # mean_updates = dict(zip(self.state_dict_key, mean_state))
        self.model.load_state_dict(dict(zip(self.state_dict_key, mean_state)))

        return self.model
