# -*- coding: utf-8 -*-
# @Author : liang
# @File : nnutils.py


import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from utils.chemutils import scaffold_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error
from collections import defaultdict
from utils.chemutils import extract_batch_data


@torch.no_grad()
def inference_test_classification(args, model, test):
    model.eval().to(args.device)
    test_dataset = DataLoader(test, batch_size=args.batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    test_loss = []
    test_acc = []

    print(' ')
    for step, test_batch in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        if test.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
            mol1_test, mol2_test = extract_batch_data(test_dataset.dataset.mol_dataset, test_batch)
            mol1_test, mol2_test, y_test = mol1_test.to(args.device), mol2_test.to(
                args.device), test_batch.y.to(args.device)
            y_pred = model(mol1_test, mol2_test)

        elif test.related_title in ['MoleculeNet', 'LITPCBA']:
            x_test, y_test = test_batch.to(args.device), test_batch.y.to(args.device)
            y_pred = model(x_test)

        loss = criterion(y_pred, y_test)

        accuracy = []
        for k in range(y_pred.shape[1]):
            try:
                accuracy.append(
                    roc_auc_score(y_test[:, k].cpu().detach().numpy(), y_pred[:, k].cpu().detach().numpy()))

                test_loss.append(loss.item())
                test_acc.append(np.mean(accuracy))

            except Exception as e:
                # print(e)
                continue

            # _, test_pred = torch.max(output, 1)
            # correct = (test_pred == y_test).sum()
            # test_acc += correct.item()

    print()
    return np.mean(test_acc), np.mean(test_loss)


@torch.no_grad()
def inference_test_regression(args, model, test):
    model.eval().to(args.device)
    test_dataset = DataLoader(test, batch_size=args.batch_size, shuffle=True)

    criterion = nn.MSELoss()

    test_loss = []
    test_rmse = []

    print(' ')
    for i, test_batch in enumerate(test_dataset):
        try:
            if test.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                mol1_test, mol2_test = extract_batch_data(test_dataset.dataset.mol_dataset, test_batch)
                mol1_test, mol2_test, y_test = mol1_test.to(args.device), mol2_test.to(
                    args.device), test_batch.y.to(args.device)
                y_pred = model(mol1_test, mol2_test)

            elif test.related_title in ['MoleculeNet', 'LITPCBA']:
                x_test, y_test = test_batch.to(args.device), test_batch.y.to(args.device)
                y_pred = model(x_test)

            loss = criterion(y_pred, y_test)
            rmse = np.sqrt(mean_squared_error(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))

            test_loss.append(loss.item())
            test_rmse.append(rmse)

        except Exception as e:
            # test_loss.append(0)
            # test_rmse.append(0)
            continue

    print()
    return np.mean(test_rmse), np.mean(test_loss)

def uniform_samples_collection(max_value = 2.82157,
                               min_value = -0.4242,
                               dim = 15,
                               sample_num = 1000):

    auxiliary_samples = np.random.uniform(low = min_value, high = max_value, size = (sample_num, dim))

    #print("auxiliary dataset:",auxiliary_samples.shape)

    return auxiliary_samples

def bulid_estimator_client(dataset, Gaussian_kernel_width=5):

    KuLSIF_estimator = KuLSIF_density_ratio_estimation(Gaussian_kernel_width = Gaussian_kernel_width,
                                                       known_samples = dataset,
                                                       auxiliary_samples = uniform_samples_collection(sample_num = 250),
                                                       lamda = 250 ** (-0.5)) # kernel with = 4 or 5 is ok. 20 and 30 are too large.


    ratio_eval = KuLSIF_estimator.ratio_estimator(dataset)

    eval_mean, eval_median, eval_std = np.mean(ratio_eval), np.median(ratio_eval), np.std(ratio_eval)
    eval_first_quartile = np.quantile(ratio_eval, q = 0.25)

    KuLSIF_estimator.eval_mean = eval_mean
    KuLSIF_estimator.eval_median = eval_median
    KuLSIF_estimator.eval_std = eval_std
    KuLSIF_estimator.eval_first_quartile = eval_first_quartile

    return KuLSIF_estimator


class KuLSIF_density_ratio_estimation:
    """
    https://github.com/shaojiawei07/Selective-FD/blob/main/KuLSIF.py
    """
    # estimate the density r(x) = p(x) / q(x)
    # p(x) is the known distribution (known samples)
    # q(x) is the auxiliary distribution
    # Particularly, supp Q should contain supp P.
    # Here, we assume X is in a compact set/space,
    # and q(x) could be a uniform distribution in
    # this space.

    def __init__(self,
                 Gaussian_kernel_width,
                 known_samples,
                 auxiliary_samples,
                 lamda):

        self.n = auxiliary_samples.shape[0] # number of auxiliary_samples
        self.m = known_samples.shape[0] # number of known samples
        self.dim = auxiliary_samples.shape[1]
        self.Gaussian_kernel_width = Gaussian_kernel_width
        self.known_samples = known_samples
        self.auxiliary_samples = auxiliary_samples
        self.lamda = lamda
        self.K11 = self._compute_K11_matrix()
        self.K12 = self._compute_K12_matrix()
        self.alpha_vector = self._compute_alpha_vector()
        self.K11 = None
        self.K12 = None

    def _compute_K11_matrix(self):
        #return a squared matrix with size aux_num * aux_num

        Gaussian_kernel_width = self.Gaussian_kernel_width
        auxiliary_samples = self.auxiliary_samples

        auxiliary_samples1 = np.expand_dims(auxiliary_samples, axis = 0) # 1 * aux_num * dim
        auxiliary_samples2 = np.expand_dims(auxiliary_samples, axis = 1) # aux_num * 1 * dim

        distance_matrix = auxiliary_samples1 - auxiliary_samples2
        distance_matrix = np.linalg.norm(distance_matrix, ord = 2, axis = 2)
        K11 = np.exp(-distance_matrix ** 2 / Gaussian_kernel_width ** 2 / 2)

        return K11

    def _compute_K12_matrix(self):
        #return a matrix with size aux_num * sample_num

        Gaussian_kernel_width = self.Gaussian_kernel_width
        known_samples = self.known_samples
        auxiliary_samples = self.auxiliary_samples

        known_samples = np.expand_dims(known_samples, axis = 0) # 1 * sample_num * dim
        auxiliary_samples = np.expand_dims(auxiliary_samples, axis = 1) # aux_num * 1 * dim

        distance_matrix = auxiliary_samples - known_samples
        distance_matrix = np.linalg.norm(distance_matrix, ord = 2, axis = 2)
        K12 = np.exp(-distance_matrix ** 2 / Gaussian_kernel_width ** 2 / 2)

        return K12

    def _compute_alpha_vector(self):

        K11 = self.K11
        K12 = self.K12
        LHS_matrix = K11 / self.n + self.lamda * np.eye(self.n)
        try:
            inverse_LHS_matrix = np.linalg.inv(LHS_matrix)
        except:
            inverse_LHS_matrix = np.linalg.pinv(LHS_matrix)

        one_vector = np.ones((self.m,1))

        RHS_vector = - K12.dot(one_vector) / (self.n * self.m * self.lamda)

        #print(RHS_vector,RHS_vector.shape,"RHS_vector")

        alpha_vector = np.dot(LHS_matrix,RHS_vector)


        return alpha_vector

    def ratio_estimator(self, test_samples):
        # test_samples (num_test_samples * dim)
        # aux_num is self.n
        # num is self.m

        test_samples = np.expand_dims(test_samples, axis = 1) # (num_test_samples, 1, dim)
        auxiliary_samples = np.expand_dims(self.auxiliary_samples, axis = 0) # (1, aux_num, dim)
        distance_matrix1 = test_samples - auxiliary_samples # (num_test_sample, aux_num, dim)
        distance_matrix1 = np.linalg.norm(distance_matrix1, ord = 2 ,axis = 2) # (num_test_sample, aux_num)
        distance_matrix1 = np.exp(-distance_matrix1 ** 2 / self.Gaussian_kernel_width ** 2 / 2) # (num_test_sample, aux_num)
        alpha_vector = np.reshape(self.alpha_vector, (self.n,))
        negative_term = np.dot(distance_matrix1,alpha_vector)


        known_samples = np.expand_dims(self.known_samples, axis = 0) # (1, num, dim)
        distance_matrix2 = test_samples - known_samples # (num_test_sample, num, dim)
        distance_matrix2 = np.linalg.norm(distance_matrix2, ord = 2 ,axis = 2) # (num_test_sample, num)
        distance_matrix2 = np.exp(-distance_matrix2 ** 2 / self.Gaussian_kernel_width ** 2 / 2) # (num_test_sample, num)
        positive_term = np.mean(distance_matrix2, axis = 1) / self.lamda

        return negative_term + positive_term

    def save_parameters(self, path):
        param_dict = {}
        param_dict["Gaussian_kernel_width"] = self.Gaussian_kernel_width
        param_dict["known_samples"] = self.known_samples
        param_dict["auxiliary_samples"] = self.auxiliary_samples
        param_dict["lamda"] = self.lamda
        param_dict["alpha_vector"] = self.alpha_vector

        torch.save(param_dict, path)


class KuLSIF_estimator(KuLSIF_density_ratio_estimation):
    def __init__(self,
                 Gaussian_kernel_width,
                 known_samples,
                 auxiliary_samples,
                 lamda,
                 alpha_vector):

        self.n = auxiliary_samples.shape[0] # number of auxiliary_samples
        self.m = known_samples.shape[0] # number of known samples
        self.dim = auxiliary_samples.shape[1]
        self.Gaussian_kernel_width = Gaussian_kernel_width
        self.known_samples = known_samples
        self.auxiliary_samples = auxiliary_samples
        self.lamda = lamda
        self.alpha_vector = alpha_vector
