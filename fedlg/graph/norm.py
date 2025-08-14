# -*- coding: utf-8 -*-
# @Author : liang
# @File : norm.py


import torch.nn as nn
import torch.nn.functional as F


class Linear_BatchNorm1d(nn.Module):
    def __init__(self, node_dim, output_dim):
        """
        Initialize the Linear_BatchNorm1d module.

        Args:
            node_dim (int): The number of input feature channels.
            output_dim (int): The number of output feature channels.
        """
        super(Linear_BatchNorm1d, self).__init__()
        self.linear = nn.Linear(node_dim, output_dim)  # Linear layer
        self.batchnorm1d = nn.BatchNorm1d(output_dim, eps=1e-06, momentum=0.1)  # Batch normalization layer

    def forward(self, x):
        """
        Forward pass of the Linear_BatchNorm1d module.

        Args:
            x (Tensor): The input feature tensor.

        Returns:
            Tensor: The output feature tensor after linear transformation, batch normalization, and ReLU activation.
        """
        x = F.relu(self.linear(x))  # Apply linear transformation and ReLU activation
        x = self.batchnorm1d(x)  # Apply batch normalization

        return x