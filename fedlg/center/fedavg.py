# -*- coding: utf-8 -*-
# @Author : liang
# @File : fedavg.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from fedlg.utils.nnutils import add_weights


class FedAvg:
    """
    Attributes:
        __model_state (list): Internal storage for aggregated model states.
        num_vars (int): Number of model variables (param).
        shape_vars (list): Shapes of the model variables.
    """

    def __init__(self):
        """
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