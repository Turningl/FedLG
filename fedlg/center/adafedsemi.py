# -*- coding: utf-8 -*-
# @Author : liang
# @File : adafedsemi.py


import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy as sp
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from fedlg.utils.chemutils import extract_batch_data


class AdaFedSemi:
    """
    Attributes:
        dataset (torch.utils.data.Dataset): The dataset used for semi-supervised learning.
        __unlabeled_data_sum (list): Internal storage for aggregated unlabeled data.
        __model_state_sum (list): Internal storage for aggregated model states.
        model (torch.nn.Module): The local model.
        global_model (torch.nn.Module): The global model.
        state_dict_key (list): Keys of the model state dictionary.
        optimizer (torch.optim.Optimizer): Optimizer for fine-tuning.
        device (torch.device): Device to use for computations (e.g., 'cuda' or 'cpu').
        batch_size (int): Batch size for fine-tuning.
        alpha (float): Hyperparameter for the participation fraction.
        max_participation (float): Maximum participation fraction.
        min_participation (float): Minimum participation fraction.
        max_confidence (float): Maximum confidence threshold.
        min_confidence (float): Minimum confidence threshold.
        teacher_model (torch.nn.Module): The teacher model for generating pseudo-labels.
        participation_actions (numpy.ndarray): Possible participation fractions.
        confidence_actions (numpy.ndarray): Possible confidence thresholds.
        participation_rewards (numpy.ndarray): Rewards for each participation fraction.
        confidence_rewards (numpy.ndarray): Rewards for each confidence threshold.
        participation_probabilities (numpy.ndarray): Probabilities for selecting participation fractions.
        confidence_probabilities (numpy.ndarray): Probabilities for selecting confidence thresholds.
        num_vars (int): Number of model variables (param).
        shape_vars (list): Shapes of the model variables.
    """

    def __init__(self,
                 model,
                 lr,
                 weight_decay,
                 batch_size,
                 device,
                 alpha=0.5,
                 max_participation=1.0,
                 min_participation=0.1,
                 max_confidence=0.99,
                 min_confidence=0.8
                 ):
        """
        Args:
            model (torch.nn.Module): The local model.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            batch_size (int): Batch size for fine-tuning.
            device (torch.device): Device to use for computations.
            alpha (float, optional): Hyperparameter for the participation fraction.
            max_participation (float, optional): Maximum participation fraction.
            min_participation (float, optional): Minimum participation fraction.
            max_confidence (float, optional): Maximum confidence threshold.
            min_confidence (float, optional): Minimum confidence threshold.
        """

        self.dataset = None
        self.__unlabeled_data = None
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
        """
        Select a participation fraction based on the current probabilities.

        Returns:
            float: The selected participation fraction.
        """

        return np.random.choice(self.participation_actions, p=self.participation_probabilities)

    def select_confidence_threshold(self):
        """
        Select a confidence threshold based on the current probabilities.

        Returns:
            float: The selected confidence threshold.
        """
        return np.random.choice(self.confidence_actions, p=self.confidence_probabilities)

    def update_teacher_model(self, model_state):
        """
        Update the teacher model with the given model state.

        Args:
            model_state (list of torch.Tensor): Model state to update the teacher model.
        """
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.load_state_dict(dict(zip(self.state_dict_key, model_state)))
        self.teacher_model.to(self.device)

    def generate_pseudo_labels(self, unlabeled_data):
        """
        Generate pseudo-labels for the given unlabeled data using the teacher model.

        Args:
            unlabeled_data (list): Unlabeled data to generate pseudo-labels for.

        Returns:
            tuple: A tuple containing the pseudo-labels and their confidences.
        """
        self.dataloader = self.dataset[unlabeled_data]
        train_dataset = DataLoader(self.dataloader, batch_size=self.batch_size, drop_last=True)

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
        """
        Aggregate a new model state and unlabeled data into the internal storage.

        Args:
            model_state (tuple): A tuple containing the model state, dataset, and unlabeled data.
        """
        model_state, dataset, unlabeled_data = model_state

        self.dataset = dataset
        self.__unlabeled_data = unlabeled_data
        self.__model_state_sum.append(model_state)

        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)

    def __average(self, client_num, participation_fraction):
        """
        Compute the average of the aggregated model states based on the participation fraction.

        Args:
            client_num (int): Number of clients.
            participation_fraction (float): Participation fraction.

        Returns:
            list of torch.Tensor: The averaged model state.
        """

        selected_clients = int(client_num * participation_fraction)
        selected_models = np.random.choice(range(client_num), selected_clients, replace=False)

        if len(selected_models) == 0:
            model_states = [self.__model_state_sum[i] for i in range(0, 1)]
        else:
            model_states = [self.__model_state_sum[i] for i in selected_models]

        # Aggregate the new state
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
        """
        Compute the average of the aggregated model states using semi-supervised learning.

        Returns:
            list of torch.Tensor: The averaged model state.
        """

        participation_fraction = self.select_participation_fraction()
        confidence_threshold = self.select_confidence_threshold()

        mean_states = self.__average(len(self.__model_state_sum), participation_fraction)
        self.global_model.load_state_dict(dict(zip(self.state_dict_key, mean_states)))

        self.update_teacher_model(mean_states)

        pseudo_labels, confidences = self.generate_pseudo_labels(self.__unlabeled_data)
        high_confidence_indices = [i for i, conf in enumerate(confidences) if np.mean(conf) >= confidence_threshold]
        high_confidence_data = [self.__unlabeled_data[i] for i in high_confidence_indices]
        high_confidence_labels = [pseudo_labels[i] for i in high_confidence_indices]
        self.dataloader = self.dataset[high_confidence_data]

        high_confidence_graphs = []
        for data, label in zip(self.dataloader, high_confidence_labels):
            if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                if isinstance(label, np.ndarray):
                    new_data = Data(smi1 = data.smi1, smi2=data.smi2, y=torch.FloatTensor(label).unsqueeze(0))
                else:
                    new_data = Data(smi1 = data.smi1, smi2=data.smi2, y=torch.FloatTensor([label]).unsqueeze(0))

            elif self.dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                if isinstance(label, np.ndarray):
                    new_data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=torch.FloatTensor(label).unsqueeze(0))
                else:
                    new_data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=torch.FloatTensor([label]).unsqueeze(0))

            high_confidence_graphs.append(new_data)
        dataloader = DataLoader(high_confidence_graphs, batch_size=self.batch_size, drop_last=True)

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
        self.__unlabeled_data = None
        self.__model_state_sum = []

        mean_model_state = [weight.data for weight in self.global_model.state_dict().values()]

        return mean_model_state

    def update_mab_probabilities(self):

        self.participation_probabilities = np.random.rand(len(self.participation_actions)) / self.participation_probabilities.sum()
        self.confidence_probabilities = np.random.rand(len(self.confidence_actions)) / self.confidence_probabilities.sum()

        self.participation_probabilities /= self.participation_probabilities.sum()
        self.confidence_probabilities /= self.confidence_probabilities.sum()
