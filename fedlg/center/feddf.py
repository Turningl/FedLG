# -*- coding: utf-8 -*-
# @Author : liang
# @File : feddf.py


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from fedlg.utils.chemutils import extract_batch_data


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
        self.dataset = None
        self.__public_idx = None
        self.__model_state_sum = []

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
        model_state, dataset, public_idx = model_state

        self.dataset = dataset
        self.__public_idx = public_idx
        self.__model_state_sum.append(model_state)

    def average(self):
        """
        Compute the average of the aggregated model states using knowledge distillation.

        Returns:
            list of torch.Tensor: The averaged model state.
        """
        self.dataloader = self.dataset[self.__public_idx]
        train_dataset = DataLoader(self.dataloader, batch_size=self.batch_size, drop_last=False)

        teacher_models = [self.update_teacher_model(teacher_model_state) for teacher_model_state in self.__model_state_sum]

        self.global_model.train()
        y_pred, teach_logits = None, None
        for step, batch in enumerate(train_dataset):
            if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                mol1_batch, mol2_batch = extract_batch_data(train_dataset.dataset.mol_dataset, batch)
                mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(self.device)
                y_pred = self.global_model(mol1_batch, mol2_batch)

                teach_logits = sum([model(mol1_batch, mol2_batch).detach() for model in teacher_models]) / len(teacher_models)

            elif self.dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                y_pred = self.global_model(mol_data)

                teach_logits = sum([model(mol_data).detach() for model in teacher_models]) / len(teacher_models)

            loss = self.divergence(y_pred, teach_logits)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.dataset = None
        self.__public_idx = None
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