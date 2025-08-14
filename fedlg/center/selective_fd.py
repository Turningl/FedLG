# -*- coding: utf-8 -*-
# @Author : liang
# @File : selective_fd.py


import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from fedlg.utils.nnutils import add_weights
from fedlg.utils.chemutils import extract_batch_data


class SelectiveFD:
    """
    Attributes:
        model (torch.nn.Module): The local model.
        global_model (torch.nn.Module): The global model.
        state_dict_key (list): Keys of the model state dictionary.
        device (torch.device): Device to use for computations (e.g., 'cuda' or 'cpu').
        batch_size (int): Batch size for fine-tuning.
        server_selector_threshold (float): Threshold for center-side selection.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        dataset (torch.utils.data.Dataset): The dataset used for fine-tuning.
        __proxy_idx_sum (list): Internal storage for aggregated proxy indices.
        __model_state_sum (list): Internal storage for aggregated model states.
        __new_model_state (list): Internal storage for new model states.
        shape_vars (list): Shapes of the model variables.
        num_vars (int): Number of model variables (param).
    """

    def __init__(self,
                 model,
                 lr,
                 weight_decay,
                 batch_size,
                 device
                 ):
        """
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
        self.__proxy_idx = None
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
        self.__proxy_idx = proxy_label_idxs
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
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)

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

                batch_softlabel.extend((y_pred * batch.y.to(self.device)).cpu().detach().numpy())

        return model, batch_softlabel

    def client_distillation(self, model, proxy_samples, batch_softlabel):
        """
        Perform database-side distillation using the provided model, proxy samples, and soft labels.

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
        new_dataloader = DataLoader(distillation_graphs, batch_size=self.batch_size, drop_last=True)

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

            model, batch_softlabel = self.proxy_pred_softlabel(local_model, self.__proxy_idx)
            new_model = self.client_distillation(local_model, self.__proxy_idx, batch_softlabel)

            new_model_state = [weight.data for weight in new_model]
            update_model_state = [state.flatten() for state in new_model_state]
            self.__new_model_state = add_weights(self.num_vars, update_model_state, self.__new_model_state)

        mean_updates = [torch.mean(self.__new_model_state[i], 0).reshape(self.shape_vars[i])
                        for i in range(self.num_vars)]

        self.__proxy_idx = None
        self.__model_state_sum = []
        self.__new_model_state = []

        return mean_updates
