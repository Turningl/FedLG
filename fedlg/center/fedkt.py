# -*- coding: utf-8 -*-
# @Author : liang
# @File : fedkt.py


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from fedlg.utils.chemutils import extract_batch_data


class FedKT:
    """
    Attributes:
        dataset (torch.utils.data.Dataset): Public dataset used for distillation.
        __model_state_sum (list): Internal storage for aggregated student model states.
        __public_idx (list): Internal storage for indices of the public subset.
        batch_size (int): Mini-batch size for distillation.
        device (torch.device): Device to run computations ('cuda' or 'cpu').
        model (torch.nn.Module): Local model architecture.
        global_model (torch.nn.Module): Global model to be distilled.
        state_dict_key (list): Ordered keys of the model state dictionary.
        optimizer (torch.optim.Optimizer): Optimizer for distillation phase.
    """


    def __init__(self,
                 model,
                 lr,
                 weight_decay,
                 epochs,
                 batch_size,
                 gamma,
                 device='cuda'
                 ):
        """
        Args:
            model (torch.nn.Module): Local model architecture to be distilled.
            lr (float): Learning rate for the distillation optimizer.
            weight_decay (float): Weight decay for the optimizer.
            epochs (int): Number of epoch to fine-tune on public pseudo-labels.
            batch_size (int): Mini-batch size for distillation.
            gamma (float): Laplace noise scale for DP (ignored if dp_level=0).
            num_classes (int): Number of output classes.
            device (torch.device): Device to run computations.
        """

        self.__model_state_sum = []
        self.__public_idx = None
        self.num_vars = None     # Number of classes
        self.shape_vars = None   # Shape of logits

        self.model = copy.deepcopy(model)
        self.global_model = copy.deepcopy(model).to(device)
        self.batch_size = batch_size

        self.lr = lr
        self.epochs = epochs
        self.gamma = gamma
        self.dp_level = 0
        self.weight_decay = weight_decay
        self.device = device
        self.stu_round= round

        self.state_dict_key = self.global_model.state_dict().keys()
        self.opt = Adam(self.global_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def update_student_model(self, model_state):
        """
        Reconstruct a student model from received parameters.

        Args:
            model_state (list[torch.Tensor]): Parameters of the student model.

        Returns:
            torch.nn.Module: Reconstructed student model on `device`.
        """
        student_model = copy.deepcopy(self.model)

        student_model.load_state_dict(dict(zip(self.state_dict_key, model_state)))
        return student_model.to(self.device)

    # ------------------ aggregate ------------------
    def aggregate(self, model_state):
        """
        Store the student model parameters and public-data indices for distillation.

        Args:
            model_state (tuple): (model_params, dataset, public_idx)
        """

        model_state, dataset, public_idx = model_state

        self.dataset = dataset
        self.__public_idx = public_idx
        self.__model_state_sum.append(model_state)


    def average(self):
        """
        Distill final model using aggregated pseudo labels.

        Returns:
            list of torch.Tensor: Parameters of the distilled global model.
        """

        self.dataloader = self.dataset[self.__public_idx]
        public_dataset = DataLoader(self.dataloader, batch_size=self.batch_size, drop_last=False)

        all_votes = []
        for model_state in self.__model_state_sum:
            stu_votes = []
            stu_model = self.update_student_model(model_state)

            stu_model.train()
            for step, batch in enumerate(public_dataset):
                if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                    mol1_batch, mol2_batch = extract_batch_data(public_dataset.dataset.mol_dataset, batch)
                    mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(
                        self.device)
                    logits = stu_model(mol1_batch, mol2_batch)

                elif self.dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                    mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                    logits = stu_model(mol_data)

                stu_pred_cls = (torch.sigmoid(logits) > 0.5).long()
                stu_votes.append(stu_pred_cls.cpu())

            all_votes.append(torch.cat(stu_votes, dim=0))

        all_votes = torch.stack(all_votes, dim=0)  # [t, N_public]
        pseudos, _ = torch.mode(all_votes, dim=0)  # [N_public]

        # Add DP noise if required
        if self.dp_level == 1:
            noise = torch.distributions.Laplace(0, 1 / self.gamma).sample(pseudos.shape)
            pseudos = (pseudos.float() + noise).clamp(0, self.num_vars - 1).round().long()

        # Rebuild dataset
        public_dataset = self.dataset[self.__public_idx]
        public_dataset.y = pseudos
        pseudo_loader = DataLoader(public_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.global_model.train()

            for batch in pseudo_loader:
                if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                    mol1, mol2 = extract_batch_data(pseudo_loader.dataset.mol_dataset, batch)
                    mol1, mol2, y = mol1.to(self.device), mol2.to(self.device), batch.y.to(self.device)
                    logits = self.global_model(mol1, mol2)

                else:
                    mol, y = batch.to(self.device), batch.y.to(self.device)
                    logits = self.global_model(mol)

                stu_loss = self.criterion(self.dataset)(logits, y)
                stu_loss.backward()
                self.opt.step()

        self.dataset = None
        self.__public_idx = None
        self.__model_state_sum = []

        mean_model_state = [weight.data for weight in self.global_model.state_dict().values()]

        return mean_model_state

    def criterion(self, dataset):

        # Define the loss criterion based on the dataset type
        criterion_ = None
        if dataset.dataset_name in dataset.dataset_names['regression']:
            criterion_ = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
        elif dataset.dataset_name in dataset.dataset_names['classification']:
            criterion_ = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for classification tasks

        return criterion_