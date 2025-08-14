# -*- coding: utf-8 -*-
# @Author : liang
# @File : server.py


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from fedlg.utils.saveutils import print_init_alg_info, print_average_info
from fedlg.center import FedLG, FedSGD, FedDF, FedAvg, FedAdam, AdaFedSemi, SelectiveFD, FedKT, SageFlow, ZenoPlusPlus


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
        num_vars (int): Number of model variables (param).
        shape_vars (list): Shapes of the model variables.
        __alg (object): The federated learning algorithm instance.
        open_access (list): List of open-access database indices.
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
            args (object): Configuration arguments containing various param.
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

        self.stu_epoch = args.local_round
        self.gamma = args.gamma

        self.entropy_threshold = args.entropy_threshold
        self.loss_exponent = args.loss_exponent
        self.staleness_exponent = args.staleness_exponent

        self.scorep = args.scorep
        self.rho = args.rho
        self.min_entropy = args.min_entropy
        self.max_entropy = args.max_entropy

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
        if self.alg == 'FedLG':
            return self.__alg.open_access_model_states, self.__alg.means
        else:
            raise TypeError("\nError loading federated learning, must be FedLG!")

    @print_init_alg_info('Initialization')
    def init_alg(self, alg):
        """
        Initialize the federated learning algorithm.

        Args:
            alg (str): Name of the federated learning algorithm to use.
        """
        self.alg = alg

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
        elif alg == 'FedKT':
            self.__alg = FedKT(self.model, self.lr, self.weight_decay, self.stu_epoch, self.batch_size, self.gamma, self.device)
        elif alg == 'Sageflow':
            self.__alg = SageFlow(self.model, self.lr, self.weight_decay, self.batch_size, self.scorep,self.entropy_threshold,
                                  self.loss_exponent, self.staleness_exponent, self.gamma, self.device, self.min_entropy, self.max_entropy)
        elif alg == 'Zeno++':
            self.__alg = ZenoPlusPlus(self.model, self.batch_size, self.gamma, self.rho, self.device)
        else:
            raise TypeError('\nSelect an algorithm to get the aggregated model.\n')

        print('\n{} algorithm.\n'.format(str(alg)))

    def aggregate(self, participant, round, model_state, alg):
        """
        Aggregate the model state from a participant database.

        Args:
            participant (int): Index of the participant database.
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
        elif alg == 'FedKT':
            self.__alg.aggregate(model_state)
        elif alg == 'Sageflow':
            self.__alg.aggregate(model_state, round)
        elif alg == 'Zeno++':
            self.__alg.aggregate(model_state)
        else:
            raise TypeError('\nSelect an algorithm to get the aggregated model.\n')

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
