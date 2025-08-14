# -*- coding: utf-8 -*-
# @Author : liang
# @File : zenoplusplus.py


import copy
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from fedlg.utils.chemutils import extract_batch_data


class ZenoPlusPlus:
    """
    Attributes:
        device (torch.device): Device to use for computations (e.g., 'cuda' or 'cpu').
        model (torch.nn.Module): The local/client model template.
        global_model (torch.nn.Module): The global model maintained by the server.
        batch_size (int): Batch size for validation on public data.
        gamma (float): Hyperparameter controlling the influence of the loss term in the score.
        rho (float): Regularization coefficient for the signed distance term.
        sco (float): Sampling ratio for the public batch used to compute scores.

        public_loader (torch.utils.data.DataLoader): DataLoader for the public validation dataset.
        __model_states (list): Internal storage for collected model states from clients.
        state_dict_key (list): Keys of the model state dictionary.
    """

    def __init__(self,
                 model,
                 batch_size=32,
                 gamma=1.0,
                 rho=1e-5,
                 device="cuda",
                 sco=0.1,
                 ):
        """
        Args:
            model (torch.nn.Module): The local/client model template.
            batch_size (int, optional): Batch size for validation on public data.
            gamma (float, optional): Loss-term weight in the score; larger values amplify loss impact.
            rho (float, optional): L2-regularization weight on the signed distance to mitigate noise.
            device (str or torch.device, optional): Device for computations.
            sco (float, optional): Fraction of public data to sample when computing client scores.
        """

        self.device = device
        self.model = model
        self.global_model = copy.deepcopy(model).to(self.device)
        self.batch_size = batch_size

        self.gamma = gamma
        self.rho = rho
        self.sco = sco

        self.public_loader = None
        self.__model_states = []
        self.state_dict_key = list(self.model.state_dict().keys())

    def aggregate(self, model_state):
        """
        Collects client model state and staleness for validation.

        Args:
            model_state: Tuple (state_list, dataset, indices)
            staleness: Time delay of the client update
        """
        state_list, dataset, indices = model_state

        # Only set public dataset once (use the first one)
        if self.public_loader is None:
            self.dataset = dataset
            self.indices = indices
            self.public_loader = DataLoader(
                dataset[indices],
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )

        self.__model_states.append(state_list)


    def average(self):
        """
        Perform Zeno++ filtering and update global model.

        Returns:
            list[torch.Tensor]: Updated global model parameters.
        """
        if not self.__model_states:
            raise ValueError("No models to aggregate.")

        grad = self._compute_gradient(self.global_model)  # server-side gradient

        for state in self.__model_states:
            client_model = self._reconstruct_model(state)

            # compute g = x_t - x_i
            g = [p1.data - p2.data for p1, p2 in zip(self.global_model.parameters(), client_model.parameters())]

            # compute Zeno++ score
            inner_product = sum((gi * gi_grad).sum() for gi, gi_grad in zip(g, grad))
            squared_norm = sum((gi ** 2).sum() for gi in g)

            score = self.gamma * inner_product - self.rho * squared_norm

            if score >= -self.gamma * self.sco:
                # Accept update: x_t = x_t - γ g
                with torch.no_grad():
                    for p, g_delta in zip(self.global_model.parameters(), g):
                        p.sub_(self.gamma * g_delta)

        self.__model_states = []
        self.public_loader = None

        mean_model_state = [weight.data for weight in self.global_model.state_dict().values()]
        return mean_model_state

    def _compute_gradient(self, model):
        """Compute gradient on a single batch of public data."""
        model.train()
        model.zero_grad()

        # use only one batch to approximate gradient
        for batch in self.public_loader:
            if self.dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                mol1, mol2 = extract_batch_data(self.dataset.mol_dataset, batch)
                mol1, mol2, y = mol1.to(self.device), mol2.to(self.device), batch.y.to(self.device)
                logits = model(mol1, mol2)
            else:
                x, y = batch.to(self.device), batch.y.to(self.device)
                logits = model(x)

            loss = self.criterion(self.dataset)(logits, y)
            loss.backward()
            break

        return [p.grad.detach() for p in model.parameters()]


    def _reconstruct_model(self, state):
        model = copy.deepcopy(self.model)
        model.load_state_dict(dict(zip(self.state_dict_key, state)))
        return model.to(self.device)


    def criterion(self, dataset):
        if dataset.dataset_name in dataset.dataset_names['regression']:
            return nn.MSELoss()
        elif dataset.dataset_name in dataset.dataset_names['classification']:
            return nn.BCEWithLogitsLoss()
        elif dataset.dataset_name in dataset.dataset_names['multi_classification']:
            return nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported dataset type for loss function.")
