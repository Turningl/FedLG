# -*- coding: utf-8 -*-
# @Author : liang
# @File : fedadam.py


import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from fedlg.utils.nnutils import add_weights


class FedAdam:
    """
    Attributes:
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small value added for numerical stability.
        lr (float): Learning rate for the Adam optimizer.
        device (torch.device): Device to use for computations (e.g., 'cuda' or 'cpu').

        __model_state (list): Internal storage for aggregated model states.
        num_vars (int): Number of model variables (param).
        shape_vars (list): Shapes of the model variables.
        m (list): First moment estimates.
        v (list): Second moment estimates.
        t (int): Time step (iteration counter).
    """

    def __init__(self,
                 beta1,
                 beta2,
                 epsilon,
                 lr,
                 device
                 ):
        """
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
