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
    """
    Args:
        num_vars (int): Number of variables (model parameters) to process.
        model_state (list of torch.Tensor): New model state to be added.
        agg_model_state (list of torch.Tensor): Existing aggregated model state.

    Returns:
        list of torch.Tensor: Updated aggregated model state.
    """
    # Use list comprehension to iterate over each variable in the model state
    return [
        # If the aggregated model state is empty, unsqueeze the new model state to create a new dimension
        torch.unsqueeze(model_state[i], 0) if not len(agg_model_state) else
        # If the aggregated model state is not empty, concatenate the new model state with the existing aggregated state
        torch.cat([agg_model_state[i], torch.unsqueeze(model_state[i], 0)], 0)
        for i in range(num_vars)
    ]


class FedAvg:
    """
    Attributes:
        __model_state (list): Internal storage for aggregated model states.
        num_vars (int): Number of model variables (parameters).
        shape_vars (list): Shapes of the model variables.
    """

    def __init__(self):
        """
        Initialize the FedAvg object.

        Initializes the internal storage for aggregated model states and sets the number of variables to None.
        """
        self.__model_state = []  # Internal storage for aggregated model states
        self.num_vars = None  # Number of model variables
        self.shape_vars = None  # Shapes of the model variables

    def aggregate(self, model_state):
        """
        Aggregate a new model state into the internal storage.

        Args:
            model_state (list of torch.Tensor): Model state to be aggregated.

        This method updates the internal storage with the new model state.
        """
        if not self.shape_vars:
            # Initialize the shape of variables if not already done
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)  # Update the number of variables
        update_model_state = [state.flatten() for state in model_state]  # Flatten each variable
        self.__model_state = add_weights(self.num_vars, update_model_state, self.__model_state)  # Aggregate the new state

    def average(self):
        """
        Compute the average of the aggregated model states.

        Returns:
            list of torch.Tensor: The averaged model state.

        This method computes the mean of the aggregated states and resets the internal storage.
        """
        mean_updates = [
            torch.mean(self.__model_state[i], 0).reshape(self.shape_vars[i])  # Compute mean and reshape
            for i in range(self.num_vars)
        ]
        self.__model_state = []  # Reset the internal storage
        return mean_updates


class FedAdam:
    """
    Attributes:
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small value added for numerical stability.
        lr (float): Learning rate for the Adam optimizer.
        device (torch.device): Device to use for computations (e.g., 'cuda' or 'cpu').

        __model_state (list): Internal storage for aggregated model states.
        num_vars (int): Number of model variables (parameters).
        shape_vars (list): Shapes of the model variables.
        m (list): First moment estimates.
        v (list): Second moment estimates.
        t (int): Time step (iteration counter).
    """

    def __init__(self, beta1, beta2, epsilon, lr, device):
        """
        Initialize the FedAdam object.

        Args:
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small value added for numerical stability.
            lr (float): Learning rate for the Adam optimizer.
            device (torch.device): Device to use for computations.
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr
        self.device = device

        self.__model_state = []  # Internal storage for aggregated model states
        self.num_vars = None  # Number of model variables
        self.shape_vars = None  # Shapes of the model variables
        self.m = []  # First moment estimates
        self.v = []  # Second moment estimates
        self.t = 0  # Time step (iteration counter)

    def aggregate(self, model_state):
        """
        Aggregate a new model state into the internal storage.

        Args:
            model_state (list of torch.Tensor): Model state to be aggregated.

        This method updates the internal storage with the new model state and updates the moment estimates.
        """
        if not self.shape_vars:
            # Initialize the shape of variables if not already done
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)  # Update the number of variables
        update_model_state = [state.flatten() for state in model_state]  # Flatten each variable

        # Initialize first and second moment estimates if not already done
        self.m = [torch.zeros(var.shape).to(self.device) for var in update_model_state]
        self.v = [torch.zeros(var.shape).to(self.device) for var in update_model_state]

        # Aggregate the new state
        self.__model_state = add_weights(self.num_vars, update_model_state, self.__model_state)

        # Update moment estimates
        self.t += 1  # Increment time step
        for i in range(self.num_vars):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * self.__model_state[i]  # Update first moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (self.__model_state[i] ** 2)  # Update second moment

    def average(self):
        """
        Compute the average of the aggregated model states using Adam optimization.

        Returns:
            list of torch.Tensor: The averaged model state.

        This method computes the Adam update and resets the internal storage.
        """
        mean_updates = []
        for i in range(self.num_vars):
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Compute the Adam update
            update = self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

            # Compute the mean update and reshape
            mean_updates.append(torch.mean(update, 0).reshape(self.shape_vars[i]))

        # Reset the internal storage
        self.__model_state = []
        return mean_updates


class FedSGD:
    """
    Attributes:
        __model_state (list): Internal storage for aggregated model states.
        num_vars (int): Number of model variables (parameters).
        shape_vars (list): Shapes of the model variables.
        lr (float): Learning rate for the gradient descent step.
    """

    def __init__(self, lr=0.01):
        """
        Initialize the FedSGD object.

        Args:
            lr (float, optional): Learning rate for the gradient descent step. Defaults to 0.01.
        """
        self.__model_state = []  # Internal storage for aggregated model states
        self.num_vars = None  # Number of model variables
        self.shape_vars = None  # Shapes of the model variables
        self.lr = lr  # Learning rate for the gradient descent step

    def aggregate(self, model_state):
        """
        Aggregate a new model state into the internal storage.

        Args:
            model_state (list of torch.Tensor): Model state to be aggregated.

        This method updates the internal storage with the new model state.
        """
        if not self.shape_vars:
            # Initialize the shape of variables if not already done
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)  # Update the number of variables
        update_model_state = [state.flatten() for state in model_state]  # Flatten each variable

        # Aggregate the new state
        self.__model_state = add_weights(self.num_vars, update_model_state, self.__model_state)

    def average(self):
        """
        Compute the average of the aggregated model states using a simple gradient descent step.

        Returns:
            list of torch.Tensor: The averaged model state.

        This method computes the mean of the aggregated states, applies the gradient descent step,
        and resets the internal storage.
        """
        mean_updates = []
        for i in range(self.num_vars):
            # Calculate the mean of updates
            mean_update = self.__model_state[i] / self.__model_state[i].shape[0]
            # Apply the gradient descent step
            update = self.lr * mean_update
            # Reshape the update to the original shape
            mean_updates.append(torch.mean(update, 0).reshape(self.shape_vars[i]))

        # Reset the internal storage
        self.__model_state = []
        return mean_updates


class FedDF:
    """
    Attributes:
        dataset (torch.utils.data.Dataset): The dataset used for fine-tuning.
        __model_state_sum (list): Internal storage for aggregated model states.
        __valid_idx_sum (list): Internal storage for aggregated validation indices.
        batch_size (int): Batch size for fine-tuning.
        device (torch.device): Device to use for computations (e.g., 'cuda' or 'cpu').
        model (torch.nn.Module): The local model.
        global_model (torch.nn.Module): The global model.
        state_dict_key (list): Keys of the model state dictionary.
        optimizer (torch.optim.Optimizer): Optimizer for fine-tuning.
    """

    def __init__(self, model, lr, weight_decay, batch_size, device):
        """
        Initialize the FedDF object.

        Args:
            model (torch.nn.Module): The local model.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            batch_size (int): Batch size for fine-tuning.
            device (torch.device): Device to use for computations.
        """
        self.dataset = None
        self.__model_state_sum = []
        self.__valid_idx_sum = []

        self.batch_size = batch_size
        self.device = device

        self.model = model
        self.global_model = copy.deepcopy(model).to(self.device)

        self.state_dict_key = self.model.state_dict().keys()
        self.optimizer = Adam(self.global_model.parameters(), lr=lr, weight_decay=weight_decay)

    def update_teacher_model(self, model_state):
        """
        Update the teacher model with the given model state.

        Args:
            model_state (list of torch.Tensor): Model state to update the teacher model.

        Returns:
            torch.nn.Module: The updated teacher model.
        """
        teacher_model = copy.deepcopy(self.model)
        teacher_model.load_state_dict(dict(zip(self.state_dict_key, model_state)))
        return teacher_model.to(self.device)

    def aggregate(self, model_state):
        """
        Aggregate a new model state and validation indices into the internal storage.

        Args:
            model_state (tuple): A tuple containing the model state, dataset, and validation indices.
        """
        model_state, dataset, valid_idx = model_state

        self.dataset = dataset
        self.__valid_idx_sum.extend(valid_idx)
        self.__model_state_sum.append(model_state)

    def average(self):
        """
        Compute the average of the aggregated model states using knowledge distillation.

        Returns:
            list of torch.Tensor: The averaged model state.
        """
        self.dataloader = self.dataset[self.__valid_idx_sum]
        train_dataset = DataLoader(self.dataloader, batch_size=self.batch_size, drop_last=False)

        teacher_models = [self.update_teacher_model(teacher_model_state) for teacher_model_state in self.__model_state_sum]

        self.global_model.train()
        for step, batch in enumerate(train_dataset):
            if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                mol1_batch, mol2_batch = extract_batch_data(train_dataset.dataset.mol_dataset, batch)
                mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(self.device)
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
        """
        Compute the KL divergence between student and teacher logits.

        Args:
            student_logits (torch.Tensor): Logits from the student model.
            teacher_logits (torch.Tensor): Logits from the teacher model.

        Returns:
            torch.Tensor: The KL divergence.
        """
        divergence = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )  # forward KL
        return divergence


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
        num_vars (int): Number of model variables (parameters).
        shape_vars (list): Shapes of the model variables.
    """

    def __init__(self, model, lr, weight_decay, batch_size, device,
                 alpha=0.5, max_participation=1.0, min_participation=0.1,
                 max_confidence=0.99, min_confidence=0.8):
        """
        Initialize the AdaFedSemi object.

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
        """
        Aggregate a new model state and unlabeled data into the internal storage.

        Args:
            model_state (tuple): A tuple containing the model state, dataset, and unlabeled data.
        """
        model_state, dataset, unlabeled_data = model_state

        self.dataset = dataset
        self.__unlabeled_data_sum.extend(unlabeled_data)
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
    """
    Attributes:
        model (torch.nn.Module): The local model.
        global_model (torch.nn.Module): The global model.
        state_dict_key (list): Keys of the model state dictionary.
        device (torch.device): Device to use for computations (e.g., 'cuda' or 'cpu').
        batch_size (int): Batch size for fine-tuning.
        server_selector_threshold (float): Threshold for server-side selection.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        dataset (torch.utils.data.Dataset): The dataset used for fine-tuning.
        __proxy_idx_sum (list): Internal storage for aggregated proxy indices.
        __model_state_sum (list): Internal storage for aggregated model states.
        __new_model_state (list): Internal storage for new model states.
        shape_vars (list): Shapes of the model variables.
        num_vars (int): Number of model variables (parameters).
    """

    def __init__(self, model, lr, weight_decay, batch_size, device):
        """
        Initialize the SelectiveFD object.

        Args:
            model (torch.nn.Module): The local model.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            batch_size (int): Batch size for fine-tuning.
            device (torch.device): Device to use for computations.
        """
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
        """
        Aggregate a new model state and proxy indices into the internal storage.

        Args:
            model_state (tuple): A tuple containing the model state, dataset, and proxy indices.
        """
        model_state, dataset, proxy_label_idxs = model_state

        self.dataset = dataset
        self.__proxy_idx_sum.extend(proxy_label_idxs)
        self.__model_state_sum.append(model_state)

        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)

    def proxy_pred_softlabel(self, model, proxy_samples):
        """
        Generate soft labels for the given proxy samples using the provided model.

        Args:
            model (torch.nn.Module): Model to generate soft labels.
            proxy_samples (list): Proxy samples to generate soft labels for.

        Returns:
            tuple: A tuple containing the model and the generated soft labels.
        """
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
        """
        Perform client-side distillation using the provided model, proxy samples, and soft labels.

        Args:
            model (torch.nn.Module): Model to perform distillation.
            proxy_samples (list): Proxy samples to use for distillation.
            batch_softlabel (list): Soft labels for the proxy samples.

        Returns:
            list of torch.Tensor: The updated model state after distillation.
        """
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
        """
        Compute the average of the aggregated model states using selective distillation.

        Returns:
            list of torch.Tensor: The averaged model state.
        """
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
    """
    Attributes:
        proj_dims (int): Number of projection dimensions.
        lanczos_iter (int): Number of Lanczos iterations.
        device (str): Device to use for computations (e.g., 'cuda' or 'cpu').
        comm_optimization (bool): Whether to use communication optimization.

        __num_open_access_databases (int): Number of open-access databases.
        __open_access_model_state (list): Internal storage for open-access model states.
        __open_access_eps (list): Internal storage for open-access epsilons.

        __num_privacy_institution_databases (int): Number of privacy-institution databases.
        __private_institutional_model_state (list): Internal storage for private-institutional model states.
        __private_institutional_eps (list): Internal storage for private-institutional epsilons.

        num_vars (int): Number of model variables (parameters).
        shape_vars (list): Shapes of the model variables.
        open_access_model_states (list): Open-access model states for communication optimization.
        means (list): Means for communication optimization.
    """

    def __init__(self, proj_dims, lanczos_iter, device='cuda', comm_optimization=None):
        """
        Initialize the FedLG object.

        Args:
            proj_dims (int): Number of projection dimensions.
            lanczos_iter (int): Number of Lanczos iterations.
            device (str, optional): Device to use for computations. Defaults to 'cuda'.
            comm_optimization (bool, optional): Whether to use communication optimization. Defaults to None.
        """

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
        """
        Aggregate a new model state and epsilon into the internal storage.

        Args:
            eps (float): Epsilon value for the model state.
            model_state (list of torch.Tensor): Model state to be aggregated.
            is_open_access (bool): Whether the model state is from an open-access database.
        """

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
        """
        Standardize the given matrix by subtracting the mean.

        Args:
            M (torch.Tensor): Matrix to be standardized.

        Returns:
            torch.Tensor: Standardized matrix.
            torch.Tensor: Mean of the matrix.
        """

        n, m = M.shape
        if m == 1:
            return M, torch.zeros(n, device=self.device)
        mean = torch.mul(M, torch.ones((m, 1), dtype=torch.float32, device=self.device)) / m

        return M - mean, mean.flatten()

    def __eigen_by_lanczos(self, mat):
        """
        Compute the largest eigenvalues and eigenvectors using Lanczos algorithm.

        Args:
            mat (torch.Tensor): Matrix to compute eigenvalues and eigenvectors.

        Returns:
            torch.Tensor: Eigenvectors corresponding to the largest eigenvalues.
        """

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
        """
        Perform Lanczos-based graph projection.

        Returns:
            list of torch.Tensor: Projected model state.
        """

        if len(self.__private_institutional_model_state):
            # Compute the weights for the private institutional model states based on their epsilons
            private_institutional_weights = (
                    torch.Tensor(self.__private_institutional_eps) / sum(self.__private_institutional_eps)
            ).view(self.__num_privacy_institution_databases, 1).to(self.device)

            # Compute the weights for the open-access model states based on their epsilons
            open_access_weights = (
                    torch.Tensor(self.__open_access_eps) / sum(self.__open_access_eps)
            ).view(self.__num_open_access_databases, 1).to(self.device)

            # Compute the weighted average of the private institutional model states
            mean_priv_model_state = [
                torch.sum(self.__private_institutional_model_state[i].to(self.device) * private_institutional_weights,
                          0) /
                torch.sum(private_institutional_weights)
                for i in range(self.num_vars)
            ]

            # Compute the weighted average of the open-access model states
            mean_pub_model_state = [
                torch.sum(self.__open_access_model_state[i].to(self.device) * open_access_weights, 0) /
                torch.sum(open_access_weights)
                for i in range(self.num_vars)
            ]

            # Initialize lists to store the projected private institutional model states and the final mean model states
            mean_proj_priv_model_state = [0] * self.num_vars
            mean_model_state = [0] * self.num_vars

            # Process each model variable
            for i in range(self.num_vars):
                # Standardize the open-access model state for the current variable
                open_access_model_state, mean = self.__standardize(self.__open_access_model_state[i].T)

                # Compute the eigenvectors using the Lanczos algorithm for projection
                proj_eigenvecs = self.__eigen_by_lanczos(open_access_model_state.T)

                # Project the private institutional model state onto the space spanned by the Lanczos eigenvectors
                mean_proj_priv_model_state[i] = (
                        torch.mul(proj_eigenvecs, torch.dot(proj_eigenvecs.T, (mean_priv_model_state[i] - mean))) + mean
                )

                # Compute the final mean model state by combining the projected private institutional model state
                # and the open-access model state, weighted by their respective epsilons
                mean_model_state[i] = (
                        (mean_proj_priv_model_state[i] * sum(self.__private_institutional_eps) +
                         mean_pub_model_state[i] * sum(self.__open_access_eps)) /
                        sum(self.__private_institutional_eps + self.__open_access_eps)
                ).reshape(self.shape_vars[i])

            # Return the computed mean model states
            return mean_model_state

    def __lanczos_graph_proj_communication_optimization(self, warmup):
        """
        Perform Lanczos-based graph projection with communication optimization.

        Args:
            warmup (bool): Whether this is the warm-up phase.

        Returns:
            list of torch.Tensor: Projected model state.
        """

        if len(self.__private_institutional_model_state):
            # Compute the normalized weights for the private institutional model states based on their epsilons.
            private_institutional_weights = (
                    torch.Tensor(self.__private_institutional_eps) / sum(self.__private_institutional_eps)
            ).view(self.__num_privacy_institution_databases, 1).to(self.device)

            # Compute the normalized weights for the open-access model states based on their epsilons.
            open_access_weights = (
                    torch.Tensor(self.__open_access_eps) / sum(self.__open_access_eps)
            ).view(self.__num_open_access_databases, 1).to(self.device)

            # Compute the weighted average of the private institutional model states.
            mean_priv_model_state = [
                torch.sum(self.__private_institutional_model_state[i].to(self.device) * private_institutional_weights,
                          0) /
                torch.sum(private_institutional_weights)
                for i in range(self.num_vars)
            ]

            # Compute the weighted average of the open-access model states.
            mean_pub_model_state = [
                torch.sum(self.__open_access_model_state[i].to(self.device) * open_access_weights, 0) /
                torch.sum(open_access_weights)
                for i in range(self.num_vars)
            ]

            # Initialize lists to store the projected private institutional model states and the final mean model states.
            mean_proj_priv_model_state = [0] * self.num_vars
            mean_model_state = [0] * self.num_vars

            # Initialize lists to store the standardized open-access model states and their means.
            open_access_model_states = []
            means = []

            if warmup:
                # If this is the warm-up phase, perform the Lanczos projection and compute the final mean model state.
                for i in range(self.num_vars):
                    # Standardize the open-access model state for the current variable.
                    open_access_model_state, mean = self.__standardize(self.__open_access_model_state[i].T)

                    # Compute the eigenvectors using the Lanczos algorithm for projection.
                    proj_eigenvecs = self.__eigen_by_lanczos(open_access_model_state.T)

                    # Project the private institutional model state onto the space spanned by the Lanczos eigenvectors.
                    mean_proj_priv_model_state[i] = torch.mul(proj_eigenvecs, torch.dot(proj_eigenvecs.T, (
                                mean_priv_model_state[i] - mean))) + mean

                    # Compute the final mean model state by combining the projected private institutional model state
                    # and the open-access model state, weighted by their respective epsilons.
                    mean_model_state[i] = (
                            (mean_proj_priv_model_state[i] * sum(self.__private_institutional_eps) +
                             mean_pub_model_state[i] * sum(self.__open_access_eps)) /
                            sum(self.__private_institutional_eps + self.__open_access_eps)
                    ).reshape(self.shape_vars[i])

                    # Store the standardized open-access model state and its mean.
                    open_access_model_states.append(open_access_model_state)
                    means.append(mean)
            else:
                # If this is not the warm-up phase, directly compute the final mean model state.
                for i in range(self.num_vars):
                    # Compute the final mean model state by combining the projected private institutional model state
                    # and the open-access model state, weighted by their respective epsilons.
                    mean_model_state[i] = (
                            (mean_proj_priv_model_state[i] * sum(self.__private_institutional_eps) +
                             mean_pub_model_state[i] * sum(self.__open_access_eps)) /
                            sum(self.__private_institutional_eps + self.__open_access_eps)
                    ).reshape(self.shape_vars[i])

                    # Standardize the open-access model state for the current variable.
                    open_access_model_state, mean = self.__standardize(self.__open_access_model_state[i].T)

                    # Store the standardized open-access model state and its mean.
                    open_access_model_states.append(open_access_model_state)
                    means.append(mean)

            # Store the standardized open-access model states and their means for future use.
            self.open_access_model_states = open_access_model_states
            self.means = means

            # Return the computed mean model states.
            return mean_model_state

    def average(self):
        """
        Compute the aggregated model updates using Lanczos-based graph projection.

        This method combines private and open-access model states to produce the final mean model state.
        It supports communication optimization to reduce the communication overhead.

        Returns:
            list of torch.Tensor: The aggregated mean model state.
        """
        # mean_updates = None
        if self.comm_optimization:
            # If communication optimization is enabled, use the optimized projection method.
            # The warmup flag is set based on whether open_access_model_states is None.
            mean_updates = self.__lanczos_graph_proj_communication_optimization(
                warmup=(self.open_access_model_states is None))
        else:
            # If communication optimization is disabled, use the standard projection method.
            mean_updates = self.__lanczos_graph_proj()

        # Reset the counters and lists for the next aggregation round.
        self.__num_open_access_databases = 0
        self.__num_privacy_institution_databases = 0

        self.__open_access_model_state = []
        self.__private_institutional_model_state = []

        self.__open_access_eps = []
        self.__private_institutional_eps = []

        return mean_updates


class GlobalServer:
    """
    Attributes:
        num_clients (int): Number of clients participating in federated learning.
        device (str): Device to use for computations (e.g., 'cuda' or 'cpu').
        model (torch.nn.Module): The global model.
        state_dict_key (list): Keys of the model state dictionary.
        proj_dims (int): Number of projection dimensions for communication optimization.
        lanczos_iter (int): Number of Lanczos iterations for communication optimization.
        beta1 (float): Beta1 parameter for Adam optimizer.
        beta2 (float): Beta2 parameter for Adam optimizer.
        lr (float): Learning rate for the optimizer.
        comm_optimization (bool): Whether to use communication optimization.
        batch_size (int): Batch size for training.
        weight_decay (float): Weight decay for the optimizer.
        num_vars (int): Number of model variables (parameters).
        shape_vars (list): Shapes of the model variables.
        __alg (object): The federated learning algorithm instance.
        open_access (list): List of open-access client indices.
        __epsilons (list): List of epsilon values for differential privacy.
        alpha (float): Alpha parameter for AdaFedSemi algorithm.
        max_participation (float): Maximum participation fraction for AdaFedSemi algorithm.
        min_participation (float): Minimum participation fraction for AdaFedSemi algorithm.
        max_confidence (float): Maximum confidence threshold for AdaFedSemi algorithm.
        min_confidence (float): Minimum confidence threshold for AdaFedSemi algorithm.
    """

    def __init__(self, model, args):
        """
        Initialize the GlobalServer object.

        Args:
            model (torch.nn.Module): The global model.
            args (object): Configuration arguments containing various parameters.
        """
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
        """
        Set the open-access database based on the provided epsilon values.

        Args:
            epsilons (list): List of epsilon values for differential privacy.
        """
        self.__epsilons = epsilons
        threshold = np.max(self.__epsilons)

        self.open_access = list(np.where(np.array(self.__epsilons) >= threshold)[0])

    def init_global_model(self):
        """
        Initialize and return the global model.
        """
        return self.model

    def fetch_comm_optimization(self):
        """
        Fetch the communication optimization states.

        Returns:
            tuple: A tuple containing the open-access model states and their means.
        """
        return self.__alg.open_access_model_states, self.__alg.means

    @print_init_alg_info('Initialization')
    def init_alg(self, alg):
        """
        Initialize the federated learning algorithm.

        Args:
            alg (str): Name of the federated learning algorithm to use.
        """
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
        """
        Aggregate the model state from a participant client.

        Args:
            participant (int): Index of the participant client.
            model_state (list of torch.Tensor): Model state to aggregate.
            alg (str): Name of the federated learning algorithm.
        """
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
        """
        Update the global model using the aggregated model states.

        Returns:
            torch.nn.Module: The updated global model.
        """
        mean_state = self.__alg.average()
        self.model.load_state_dict(dict(zip(self.state_dict_key, mean_state)))

        return self.model
