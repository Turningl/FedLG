# -*- coding: utf-8 -*-
# @Author : liang
# @File : sageflow.py



import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.loader import DataLoader
from fedlg.utils.chemutils import extract_batch_data


class SageFlow:
    """
    Attributes:
        device (torch.device): Device to use for computations (e.g., 'cuda' or 'cpu').
        batch_size (int): Batch size for training.
        entropy_threshold (float): Entropy threshold for filtering high-uncertainty samples.
        loss_exponent (float): Exponent for scaling loss-based weighting.
        staleness_exponent (float): Exponent for scaling staleness-based weighting.
        gamma (float): Discount factor for staleness calculation.
        scorep (float): Proportion of samples to select based on score.

        model_template (torch.nn.Module): Template for local/client models.
        global_model (torch.nn.Module): The global model maintained by the server.
        optimizer (torch.optim.Optimizer): Optimizer for training the global model.
        state_dict_keys (list): Keys of the model state dictionary.

        __model_states (list): Internal storage for collected model states from clients.
        __staleness (list): Internal storage for staleness values of client updates.
        __sample_counts (list): Internal storage for sample counts from clients.

        public_loader (torch.utils.data.DataLoader): DataLoader for public dataset.
        loss_fn (torch.nn.Module): Loss function for training.
        dataset (torch.utils.data.Dataset): The dataset used for training.
        indices (list): Indices of selected samples from the public dataset.
        task_type (str): Type of the task (e.g., 'classification', 'regression').
        dataset_type (str): Type of the dataset
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float,
            weight_decay: float,
            batch_size: int,
            scorep,
            entropy_threshold,
            loss_exponent,
            staleness_exponent,
            gamma,
            device,
            max_entropy,
            min_entropy
    ):
        """
        Args:
            model (torch.nn.Module): The local/client model template.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            batch_size (int): Batch size for training.
            scorep (float): Proportion of samples to select based on score.
            entropy_threshold (float): Entropy threshold for filtering high-uncertainty samples.
            loss_exponent (float): Exponent for scaling loss-based weighting.
            staleness_exponent (float): Exponent for scaling staleness-based weighting.
            gamma (float): Discount factor for staleness calculation.
            device (torch.device): Device to use for computations.
        """

        self.device = device
        self.batch_size = batch_size
        self.entropy_threshold = entropy_threshold
        self.loss_exponent = loss_exponent
        self.staleness_exponent = staleness_exponent
        self.gamma = gamma
        self.scorep = scorep

        self.model_template = model
        self.global_model = copy.deepcopy(model).to(self.device)
        self.optimizer = Adam(self.global_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.state_dict_keys = list(model.state_dict().keys())

        self.__model_states = []
        self.__staleness = []
        self.__sample_counts = []

        # public data & loss function (to be initialized on first aggregate call)
        self.public_loader = None
        self.loss_fn = None
        self.dataset = None
        self.indices = None
        self.task_type = None
        self.dataset_type = None

        self.max_entropy = max_entropy
        self.min_entropy = min_entropy

    def aggregate(self, model_state, staleness):
        """
        Args:
            model_state: tuple (state_list, dataset, indices)
            staleness: int, how many rounds stale
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

            self.task_type = self._detect_task_type(dataset)
            self.loss_fn = self._build_criterion()
            self.dataset_type = dataset.related_title

        self.__sample_counts.append(len(indices))
        self.__model_states.append(state_list)
        self.__staleness.append(staleness)

    def average(self):
        if not self.__model_states:
            raise ValueError("No models received.")

        # Group models by staleness
        groups = {}
        for state, stale, count in zip(self.__model_states, self.__staleness, self.__sample_counts):
            groups.setdefault(stale, []).append((state, count))

        group_reps = []
        group_weights = []

        for staleness, group in groups.items():
            survivors = []
            for state_list, count in group:
                model = self._reconstruct_model(state_list)
                entropy = self._compute_entropy(model)

                if entropy:
                    loss = self._compute_loss(model)
                    survivors.append({"model": model, "count": count, "loss": loss})

            if not survivors:
                continue

            inverse_loss = [self.scorep / (s["loss"] ** self.loss_exponent + 1e-8) for s in survivors]
            total_weight = sum(s["count"] * w for s, w in zip(survivors, inverse_loss))

            rep_model = copy.deepcopy(self.model_template).to(self.device)

            with torch.no_grad():
                for name, param in rep_model.named_parameters():
                    weighted_sum = sum(
                        s["count"] * w * s["model"].state_dict()[name]
                        for s, w in zip(survivors, inverse_loss)
                    )

                    param.copy_(weighted_sum / total_weight)

            group_reps.append(rep_model)
            group_weights.append(
                sum(s["count"] for s in survivors) / ((staleness + 1) ** self.staleness_exponent)
            )

        if not group_reps:
            raise ValueError("No valid models after entropy filtering.")

        total_group_weight = sum(group_weights)
        group_weights = [w / total_group_weight for w in group_weights]

        with torch.no_grad():
            new_state = {}
            for name in self.global_model.state_dict():
                weighted_avg = sum(w * m.state_dict()[name] for w, m in zip(group_weights, group_reps))
                new_state[name] = (1 - self.gamma) * self.global_model.state_dict()[name] + self.gamma * weighted_avg
            self.global_model.load_state_dict(new_state)

        # Cleanup cache
        self.__model_states = []
        self.__staleness = []
        self.__sample_counts = []
        self.public_loader = None

        mean_model_state = [weight.data for weight in self.global_model.state_dict().values()]

        return mean_model_state

    def _reconstruct_model(self, flat_state):
        model = copy.deepcopy(self.model_template)
        model.load_state_dict(dict(zip(self.state_dict_keys, flat_state)))
        return model.to(self.device)

    def _detect_task_type(self, dataset):
        if dataset.dataset_name in dataset.dataset_names["regression"]:
            return "regression"
        elif dataset.dataset_name in dataset.dataset_names["classification"]:
            return "binary"
        elif dataset.dataset_name in dataset.dataset_names["multi_classification"]:
            return "multi"
        else:
            raise ValueError("Unknown task type")

    def _build_criterion(self):
        if self.task_type == "regression":
            return nn.MSELoss()
        elif self.task_type == "binary":
            return nn.BCEWithLogitsLoss()
        elif self.task_type == "multi":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported task type.")

    def _compute_entropy(self, model):
        model.eval()
        entropy_sum, total = 0.0, 0.0

        with torch.no_grad():
            for batch in self.public_loader:
                if self.dataset_type in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                    mol1, mol2 = extract_batch_data(self.public_loader.dataset.mol_dataset, batch)
                    mol1, mol2, y = mol1.to(self.device), mol2.to(self.device), batch.y.to(self.device)
                    logits = model(mol1, mol2)
                else:
                    x, y = batch.to(self.device), batch.y.to(self.device)
                    logits = model(x)

                if self.task_type == "binary":
                    # p = torch.sigmoid(logits).clamp(1e-8, 0.99 - 1e-8)
                    
                    p = torch.sigmoid(logits)
                    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).squeeze()
                    entropy = entropy.clamp(min=self.min_entropy, max=self.max_entropy)

                elif self.task_type == "multi":
                    # p = torch.softmax(logits, dim=1).clamp(1e-8, 0.99 - 1e-8)
                    
                    p = torch.softmax(logits, dim=1)
                    entropy = -(p * torch.log(p)).sum(dim=1).clamp(min=self.min_entropy, max=self.max_entropy)

                elif self.task_type == "regression":
                    # entropy = ((logits - y) ** 2).clamp(min=self.min_entropy, max=self.max_entropy)
                    entropy = ((logits - y) ** 2)

                entropy_sum += entropy.sum().item()
                total += logits.size(0)

        return entropy_sum / total


    def _compute_loss(self, model):
        model.eval()
        loss_sum, total = 0.0, 0

        with torch.no_grad():
            for batch in self.public_loader:
                if self.dataset_type in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                    mol1, mol2 = extract_batch_data(self.public_loader.dataset.mol_dataset, batch)
                    mol1, mol2, y = mol1.to(self.device), mol2.to(self.device), batch.y.to(self.device)
                    logits = model(mol1, mol2)
                else:
                    x, y = batch.to(self.device), batch.y.to(self.device)
                    logits = model(x)

                loss = self.loss_fn(logits, y)
                loss_sum += loss.item()
                total += logits.size(0)

        return loss_sum / total


