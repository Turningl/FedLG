# -*- coding: utf-8 -*-
# @Author : liang
# @File : server.py

import torch
import numpy as np
import scipy as sp
from utils.lanczos import Lanczos


def add_weights(num_vars, model_state, agg_model_state):
    for i in range(num_vars):
        if not len(agg_model_state):
            a = torch.unsqueeze(model_state[i], 0)
        else:
            b = torch.cat([agg_model_state[i], torch.unsqueeze(model_state[i], 0)], 0)
    return [torch.unsqueeze(model_state[i], 0)
            if not len(agg_model_state) else torch.cat([agg_model_state[i], torch.unsqueeze(model_state[i], 0)], 0) for
            i in range(num_vars)]


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
        mean_updates = [torch.mean(self.__model_state[i], 0).reshape(self.shape_vars[i]) for i in range(self.num_vars)]
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


class FedLG:
    def __init__(self, proj_dims, lanczos_iter, device='cuda'):
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
        self.Vks = None
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
        T, V = Lanczos(mat, self.lanczos_iter)  # getting a tridiagonal matrix T and an orthogonal matrix V

        # T_evals_, T_evecs_ = np.linalg.eig(T)  # calculating the eigenvalues and eigenvectors of a tridiagonal matrix
        T_evals, T_evecs = sp.sparse.linalg.eigsh(T, k=2, which='LM')

        idx = T_evals.argsort()[-1: -(self.proj_dims + 1): -1]  # finding the index of the largest element in the eigenvalue array T evals

        Vk = np.dot(V.T, T_evecs[:, idx])  # the eigenvector corresponding to the maximum eigenvalue is projected into the new eigenspace

        if Vk.size >= 2:
            Vk = torch.from_numpy(Vk).to(torch.float32).to(self.device).squeeze()
        else:
            Vk = torch.from_numpy(Vk).to(torch.float32).to(self.device).squeeze(0)

        return Vk

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
                Vk = self.__eigen_by_lanczos(open_access_model_state.T)
                mean_proj_priv_model_state[i] = torch.mul(Vk, torch.dot(Vk.T, (mean_priv_model_state[i] - mean))) + mean
                mean_model_state[i] = ((mean_proj_priv_model_state[i] * sum(self.__private_institutional_eps) + mean_pub_model_state[i] * sum(self.__open_access_eps)) / sum(self.__private_institutional_eps + self.__open_access_eps)).reshape( self.shape_vars[i])

            return mean_model_state

    def average(self):
        # mean_updates = None
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
        # self.warmup = args.warmup

        self.num_vars = None
        self.shape_vars = None
        self.__alg = None
        self.open_access = None
        self.__epsilons = None

    def set_open_access_database(self, epsilons):
        self.__epsilons = epsilons
        threshold = np.max(self.__epsilons)

        self.open_access = list(np.where(np.array(self.__epsilons) >= threshold)[0])

    def init_global_model(self):
        return self.model

    def init_alg(self, alg):
        if alg == 'FedAvg' or alg == 'FedProx' or alg == 'FedChem':
            self.__alg = FedAvg()
        elif alg == 'FedSGD':
            self.__alg = FedSGD(self.lr)
        elif alg == 'FedAdam':
            self.__alg = FedAdam(beta1=self.beta1, beta2=self.beta2, epsilon=self.__epsilons, lr=self.lr, device=self.device)
        elif alg == 'FedLG':
            self.__alg = FedLG(self.proj_dims, self.lanczos_iter, self.device)
        else:
            raise ValueError('\nSelect an algorithm to get the aggregated model.\n')

        print('\nUsing {} algorithm.\n'.format(str(alg)))

    def aggregate(self, participant, model_state, alg):
        if alg == 'FedLG':
            self.__alg.aggregate(self.__epsilons[participant], model_state,
                                 is_open_access=True if (participant in self.open_access) else False)
        elif alg == 'FedAvg' or alg == 'FedProx' or alg == 'FedChem':
            self.__alg.aggregate(model_state)
        elif alg == 'FedSGD':
            self.__alg.aggregate(model_state)
        elif alg == 'FedAdam':
            self.__alg.aggregate(model_state)
        else:
            raise ValueError('\nSelect an algorithm to get the aggregated model.\n')

    def update(self):
        mean_state = self.__alg.average()
        # mean_updates = dict(zip(self.state_dict_key, mean_state))
        self.model.load_state_dict(dict(zip(self.state_dict_key, mean_state)))

        return self.model
