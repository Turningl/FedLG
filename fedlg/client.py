# -*- coding: utf-8 -*-
# @Author : liang
# @File : client.py


import ast
import copy
import torch
import random
import numpy as np
import scipy as sp
import configparser
import torch.nn as nn
import hyperopt as hy
from torch.optim import Adam, SGD, RMSprop
from torch_geometric.loader import DataLoader
from utils.chemutils import extract_batch_data
from utils.nnutils import bulid_estimator_client
from torch_geometric.nn import global_mean_pool
from utils.lanczos import Lanczos
from tqdm import tqdm

class SimulatedDatabase:
    """
    A simulated database class for federated learning.

    This class simulates a local database for federated learning, including local training,
    differential privacy, and communication optimization.
    """

    def __init__(self, train, indices, args):
        """
        Initialize the SimulatedDatabase.

        Args:
            train (list): Training dataset for local training.
            indices (list): Index values obtained by the splitting algorithm.
            args (object): Configuration arguments.
                - batch_size (int): Batch size for local training.
                - local_round (int): Number of local training rounds.
                - dp (bool): Whether to use differential privacy.
                - device (str): Device to use (e.g., 'cuda' or 'cpu').
                - lr (float): Learning rate for the optimizer.
                - weight_decay (float): Weight decay for the optimizer.
                - clip (float): Gradient clipping value.
                - seed (int): Random seed for reproducibility.
                - constant (float): Constant value for differential privacy.
                - alg (str): Federated learning algorithm.
                - proj_dims (int): Projection dimensions for communication optimization.
                - lanczos_iter (int): Lanczos iteration count for communication optimization.
                - init (bool): Initialization flag.
                - max_step (int): Maximum step count.
        """

        super(SimulatedDatabase, self).__init__()
        self.train = train  # train dataset for local training
        self.indices = indices  # index value obtained by splitting algorithm
        self.dataset_size = len(indices)  # training dataset size
        self.batch_size = args.batch_size  # batch size
        self.local_round = args.local_round  # local training round
        self.dp = args.dp  # differential privacy value
        # self.grad_norm = args.grad_norm  # grad norm value
        self.device = args.device  # device is cuda or cpu
        self.lr = args.lr  # learning rate size
        self.weight_decay = args.weight_decay # weight decay size
        self.clip = args.clip # clip value
        self.random_seed = args.seed  # random seed value
        self.constant = args.constant  # constant
        self.alg = args.alg  # fedavg, fedprox, fedsgd, fedlg, fedadam and flit algorithm can be chose
        # self.anti_noise_ability = args.anti_noise  # anti noise ability magnitude
        self.model = None  # The global model starts with None
        self.optimizer = None  # optimizer starts with None
        self.is_private = None  # None
        self.weight_denomaitor = None  # None
        self.mu = 0.1  # fedprox mu value
        self.warmup = True  # flit warm up value
        self.factor_ema = 0.8  # flit factor ema value
        self.tmp = 0.5  # flit tmp value

        self.proj_dims = args.proj_dims
        self.lanczos_iter = args.lanczos_iter

        self.state_dict_key = None
        self.shape_vars = None
        self.model_states = None
        self.means = None

        self.init = args.init
        self.max_step = args.max_step
        self.updates = None
        self.rstate = None
        self.train_dataset = None # train dataset
        self.valid_dataset = None # valid dataset

    # download model parameter
    def download(self, model):
        """
        Download the global model to the local device.

        Args:
            model (nn.Module): Global model to be downloaded.
        """
        if self.device:
            self.model = model.to(self.device)  # Move the model to the specified device
        else:
            self.model = model  # Use the model as is
        self.state_dict_key = self.model.state_dict().keys()  # Get the keys of the model state dictionary

    # update communication efficiency information value
    def update_comm_optimization(self, model_states=None, means=None, participant=None):
        """
        Update communication optimization information.

        Args:
            model_states (list, optional): Model states for communication optimization. Defaults to None.
            means (list, optional): Means for communication optimization. Defaults to None.
            participant (bool, optional): Whether this database is a participant. Defaults to None.
        """
        self.model_states = model_states  # Update model states
        self.means = means  # Update means
        self.is_participant = participant  # Update participant status

    def set_optimizer(self):
        pass

    def set_local_differential_privacy(self, epsilon):
        """
        Set the local differential privacy value.

        Args:
            epsilon (float): Differential privacy value.
        """
        self.epsilon = epsilon  # Set the differential privacy value

    def set_local_distribution(self, updates):
        """
        Add Laplace noise to the updates for differential privacy.

        Args:
            updates (Tensor): Updates to be modified.

        Returns:
            Tensor: Updates with added Laplace noise.
        """
        if self.dp:  # Check if differential privacy is enabled
            scale = torch.tensor(self.constant * self.lr * self.clip / np.sqrt(self.batch_size) / self.epsilon)
            # Compute the scale for Laplace noise

            if self.random_seed:  # Set random seed for reproducibility
                random.seed(self.random_seed)
                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)
                torch.cuda.manual_seed(self.random_seed)
                torch.cuda.manual_seed_all(self.random_seed)

            laplace_dist = torch.distributions.Laplace(0, scale)  # Define Laplace distribution
            laplace_noise = laplace_dist.sample(updates.shape)  # Sample Laplace noise

            if self.device:  # Move noise to the specified device
                laplace_noise = laplace_noise.to(self.device)

            updates += laplace_noise.type(updates.dtype)  # Add noise to updates
            return updates
        return updates  # Return updates without noise if differential privacy is disabled

    def __standardize(self, M):
        """
        Standardize the input matrix M by subtracting the mean of each row.

        Args:
            M (torch.Tensor): Input matrix of shape (n, m).

        Returns:
            torch.Tensor: Standardized matrix of shape (n, m).
            torch.Tensor: Mean vector of shape (n,) representing the mean of each row.
        """
        n, m = M.shape  # Get the dimensions of the input matrix M
        if m == 1:
            return M, torch.zeros(n, device=self.device)  # If the matrix has only one column, return the original matrix and a zero mean vector

        mean = torch.mul(M, torch.ones((m, 1), dtype=torch.float32, device=self.device)) / m  # Compute the mean of each row
        standardized_M = M - mean   # Subtract the mean from each row to standardize the matrix

        return standardized_M, mean.flatten()  # Return the standardized matrix and the mean vector

    def __eigen_by_lanczos(self, mat, lanczos_iter):
        Tri_Mat, Orth_Mat = Lanczos(mat, lanczos_iter)  # getting a tridiagonal matrix T and an orthogonal matrix V

        T_evals_, T_evecs_ = np.linalg.eig(Tri_Mat)  # calculating the eigenvalues and eigenvectors of a tridiagonal matrix
        T_evals, T_evecs = sp.sparse.linalg.eigsh(Tri_Mat, k=2, which='LM')

        idx = T_evals.argsort()[-1: -(self.proj_dims + 1): -1]  # finding the index of the largest element in the eigenvalue array T evals

        proj_eigenvecs = np.dot(Orth_Mat.T, T_evecs[:, idx])  # the eigenvector corresponding to the maximum eigenvalue is projected into the new eigenspace

        if proj_eigenvecs.size >= 2:
            proj_eigenvecs = torch.from_numpy(proj_eigenvecs).to(torch.float32).to(self.device).squeeze()
        else:
            proj_eigenvecs = torch.from_numpy(proj_eigenvecs).to(torch.float32).to(self.device).squeeze(0)

        return proj_eigenvecs

    # local model update function
    def local_update(self):
        """
        Perform local model updates for federated learning.

        This method trains the local model on the assigned dataset, validates the model,
        and optionally applies differential privacy and Bayesian optimization.
        """

        model = self.model.train()
        w_model = copy.deepcopy(model)  # Create a copy of the model for reference

        # Get the local dataset based on the assigned indices
        dataset = self.train[self.indices]

        # Initialize the optimizer
        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Shuffle the indices to randomize the training order
        np.random.shuffle(self.indices)

        # Split the dataset into training and validation sets
        self.train_size, self.val_size = int(0.9 * len(self.indices)), len(self.indices) - int(0.9 * len(self.indices))
        train_idx, valid_idx = self.indices[:self.train_size], self.indices[self.train_size:]

        # Create training and validation datasets
        self.train_dataset = self.train[train_idx]
        self.valid_dataset = self.train[valid_idx]

        # Create data loaders for training and validation
        train_dataset = DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=False)
        valid_dataset = DataLoader(self.valid_dataset, batch_size=self.batch_size, drop_last=False)

        # Define the loss criterion based on the dataset type
        criterion = None
        if dataset.dataset_name in dataset.dataset_names['regression']:
            criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
        elif dataset.dataset_name in dataset.dataset_names['classification']:
            criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for classification tasks

        # if self.dp:
        #     privacy_engine = PrivacyEngine(secure_mode=False)
        #     model, optimizer, train_loader = privacy_engine.make_private(module=model,
        #                                                                  optimizer=optimizer,
        #                                                                  data_loader=data_loader,
        #                                                                  noise_multiplier=self.budget_accountant.noise_multiplier,
        #                                                                  max_grad_norm=self.grad_norm)

        # local training process
        for epoch in range(self.local_round):
            train_losses, val_losses = 0, 0

            model = model.train()  # training
            for step, batch in enumerate(train_dataset):
                if dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                    mol1_batch, mol2_batch = extract_batch_data(train_dataset.dataset.mol_dataset, batch)
                    mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(
                        self.device)
                    y_pred = model(mol1_batch, mol2_batch)

                elif dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                    mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                    y_pred = model(mol_data)

                train_loss = criterion(y_pred, y_true)

                # Add proximal term for FedProx algorithm
                if self.alg == 'FedProx':
                    proximal_term = 0
                    for w, w_global in zip(model.parameters(), w_model.parameters()):
                        proximal_term += self.mu / 2 * torch.norm(w - w_global, 2)
                    train_loss += proximal_term

                # Apply FLIT algorithm
                elif self.alg == 'FLIT':
                    if self.warmup:
                        with torch.no_grad():
                            if dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                                y_pred_global = w_model(mol1_batch, mol2_batch)
                            elif dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                                y_pred_global = w_model(mol_data)

                        lossg_label = criterion(y_pred_global, y_true)
                        lossl_label = criterion(y_pred, y_true)
                        weightloss = lossl_label + torch.relu(lossl_label - lossg_label.detach())

                        if self.weight_denomaitor is None:
                            self.weight_denomaitor = weightloss.mean(dim=0, keepdim=True).detach()
                        else:
                            self.weight_denomaitor = self.factor_ema * self.weight_denomaitor + (
                                        1 - self.factor_ema) * weightloss.mean(dim=0, keepdim=True).detach()

                        train_loss = (1 - torch.exp(
                            -weightloss / (self.weight_denomaitor + 1e-7)) + 1e-7) ** self.tmp * lossl_label
                    else:
                        self.warmup += 1

                # Backpropagation and optimization
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_losses += train_loss.item() * len(batch)

            if self.alg in ['FedDF', 'AdaFedSemi', 'SelectiveFD']:
                print('Epoch is: %d, Train loss: %.4f' % ((epoch + 1), train_losses / int(0.9 * len(dataset))))

            else:
                model.eval()  # eval
                with torch.no_grad():
                    for step, batch in enumerate(valid_dataset):
                        if dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                            mol1_batch, mol2_batch = extract_batch_data(valid_dataset.dataset.mol_dataset, batch)
                            mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(self.device)
                            y_pred = model(mol1_batch, mol2_batch)

                        elif dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                            mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                            y_pred = model(mol_data)

                        val_loss = criterion(y_pred, y_true)
                        val_losses += val_loss.item() * len(batch)

                print('Epoch is: %d, Train loss: %.4f, Valid loss: %.4f' % ((epoch + 1), train_losses / int(0.9 * len(dataset)), val_losses / (len(dataset) - int(0.9 * len(dataset)))))

        # collecting the weights of local model
        updates = [weight.data for weight in model.state_dict().values()]

        # adding differently privacy by set laplace noise
        if self.dp:
            updates = [self.set_local_distribution(updates[i]) for i in range(len(updates))]

            self.updates = updates
            self.shape_vars = [var.shape for var in self.updates]

        # fetch Bayesian optimization
        if self.model_states and self.is_participant:

            # execute bayesian optimization process
            best_pbounds = self.bayesian_optimization()

            # get project updates with the best parameter from bayesian optimization
            _, proj_updates = self.fetch_model_update(best_pbounds)

            return proj_updates

        if self.alg in ['FedDF', 'AdaFedSemi']:
            return updates, self.train, valid_idx

        elif self.alg in ['SelectiveFD']:
            # Prepare data for density ratio estimation
            known_dataset = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), drop_last=False)
            estimated_dataset = DataLoader(self.valid_dataset, batch_size=len(self.valid_dataset), drop_last=False)

            known_mol_data, known_batch = None, None
            for step, known_batch in enumerate(known_dataset):
                if dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                    mol1_batch, mol2_batch = extract_batch_data(known_dataset.dataset.mol_dataset, known_batch)
                    known_mol_data = mol1_batch.x
                else:
                    known_mol_data = known_batch.x

            estimated_mol_data, estimated_batch = None, None
            for step, estimated_batch in enumerate(estimated_dataset):
                if dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                    mol1_batch, mol2_batch = extract_batch_data(estimated_dataset.dataset.mol_dataset, estimated_batch)
                    estimated_mol_data = mol1_batch.x
                else:
                    estimated_mol_data = estimated_batch.x

            known_data = global_mean_pool(known_mol_data, known_batch.batch).numpy()
            estimated_data = global_mean_pool(estimated_mol_data, estimated_batch.batch).numpy()

            # Build density ratio estimator
            density_ratio_estimator = bulid_estimator_client(dataset=known_data, Gaussian_kernel_width=5)

            # Estimate density ratios
            estimated_density_ratio = density_ratio_estimator.ratio_estimator(estimated_data)
            binary_classification_result = estimated_density_ratio > density_ratio_estimator.eval_first_quartile  # eval_median

            # Filter validation indices based on density ratio
            filtered_valid_idx = [idx for idx, mask in zip(valid_idx, binary_classification_result) if mask]

            return updates, self.train, filtered_valid_idx

        return updates

    # bayesian optimization process
    def bayesian_optimization(self):
        """
        Perform Bayesian optimization using Tree-structured Parzen Estimator (TPE).

        This method optimizes the hyperparameters defined in the `pbounds` dictionary
        using the TPE algorithm from the Hyperopt library.

        Returns:
            dict: The best hyperparameters found by the optimization process.
        """
        # Define the parameter bounds for Bayesian optimization
        pbounds = {
            'lanczos_iter': self.hy_parameter_setting(label='lanczos_iter'),  # Set Lanczos iteration number
        }

        # Print parameter bounds (optional)
        # print("Parameter bounds:")
        # for param, value in pbounds.items():
        #     print(f"{param}: {value}")

        print('Using Bayesian Optimization with TPE')

        # Set the Tree-structured Parzen Estimator (TPE) algorithm
        algo = hy.partial(hy.tpe.suggest, n_startup_jobs=self.init)

        # Set the random state for reproducibility
        if self.random_seed:
            self.rstate = np.random.RandomState(self.random_seed)

        # Initialize the Trials object to store the optimization history
        trials = hy.Trials()

        # Perform Bayesian optimization
        best = hy.fmin(
            fn=self.model_prediction,  # Objective function to minimize
            space=pbounds,  # Parameter space to search
            algo=algo,  # Optimization algorithm
            max_evals=self.max_step,  # Maximum number of evaluations
            trials=trials,  # Store optimization history
            rstate=self.rstate  # Random state for reproducibility
        )

        return best

    def fetch_model_update(self, params):
        """
        Fetch the model update based on the provided hyperparameters.

        This method projects the model updates using the Lanczos algorithm and
        returns the updated model.

        Args:
            params (dict): Hyperparameters obtained from Bayesian optimization.

        Returns:
            nn.Module: The updated model.
            list: The projected updates.
        """
        # Flatten the updates
        updates = [update.flatten() for update in self.updates]
        proj_updates = [0] * len(self.model_states)

        # Execute Lanczos algorithm and project updates
        for i in range(len(self.model_states)):
            proj_eigenvecs = self.__eigen_by_lanczos(self.model_states[i].T, lanczos_iter=params['lanczos_iter'])
            proj_updates[i] = (torch.mul(proj_eigenvecs, torch.dot(proj_eigenvecs.T, (updates[i] - self.means[i]))) +
                               self.means[i]).reshape(self.shape_vars[i])

        # Load the updated model state dictionary
        model = self.model.eval()
        model.load_state_dict(dict(zip(self.state_dict_key, proj_updates)))

        return model, proj_updates


    @torch.no_grad()
    def model_prediction(self, params):
        """
        Evaluate the model on the validation dataset using the provided parameters.

        Args:
            params (dict): Model parameters to be used for evaluation.

        Returns:
            float: Average loss on the validation dataset.
        """
        # Get the local dataset based on the assigned indices
        dataset = self.train[self.indices]

        # Create a data loader for the validation dataset
        val_dataset = DataLoader(self.valid_dataset, batch_size=self.batch_size, drop_last=True)

        # Define the loss criterion based on the dataset type
        criterion = None
        if dataset.dataset_name in dataset.dataset_names['regression']:
            criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
        elif dataset.dataset_name in dataset.dataset_names['classification']:
            criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for classification tasks

        # Fetch the model with the provided parameters
        model, _ = self.fetch_model_update(params)

        # Evaluation process
        losses = 0
        for step, batch in enumerate(val_dataset):
            # Extract batch data based on the dataset type
            if dataset.related_title in ['DrugBank', 'BIOSNAP', 'CoCrystal']:
                mol1_batch, mol2_batch = extract_batch_data(val_dataset.dataset.dataset.mol_dataset, batch)
                mol1_batch, mol2_batch, y_true = mol1_batch.to(self.device), mol2_batch.to(self.device), batch.y.to(
                    self.device)
                y_pred = model(mol1_batch, mol2_batch)
            elif dataset.related_title in ['MoleculeNet', 'LITPCBA']:
                mol_data, y_true = batch.to(self.device), batch.y.to(self.device)
                y_pred = model(mol_data)

            # Compute the loss
            loss = criterion(y_pred, y_true)
            losses += loss.item() * len(batch)

        # Return the average loss on the validation dataset
        return losses / len(self.valid_dataset)

    def hy_parameter_setting(self, label):
        """
        Set hyperparameters for Bayesian optimization using the provided label.

        Args:
            label (str): Label for the hyperparameter to be set.

        Returns:
            hyperopt.hp: Hyperparameter object for Bayesian optimization.
        """
        # Read the configuration file
        config = configparser.RawConfigParser()
        config.read('./parameters/{}_range.in'.format('optimization'))

        try:
            # Get the range for the specified label
            val = ast.literal_eval(config.get('range', label))
        except Exception as e:
            print(f"Error reading config for {label}: {e}")
            return None

        # Set hyperopt parameter
        if type(val) is list:
            if label == 'lanczos_iter':
                # Define a choice of values for Lanczos iteration
                return hy.hp.choice(label, np.arange(val[0], val[1], 2).tolist())
            else:
                print('Unsupported parameter. Check configuration or label name.')
        else:
            print('Invalid data format in config file. Expected a list of two values.')

        return None
