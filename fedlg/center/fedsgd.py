# -*- coding: utf-8 -*-
# @Author : liang
# @File : fedsgd.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from fedlg.utils.nnutils import add_weights


class FedSGD:
    """
    Attributes:
        __model_state (list): Internal storage for aggregated model states.
        num_vars (int): Number of model variables (param).
        shape_vars (list): Shapes of the model variables.
        lr (float): Learning rate for the gradient descent step.
    """

    def __init__(self,
                 lr=0.01
                 ):
        """
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