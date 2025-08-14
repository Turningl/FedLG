# -*- coding: utf-8 -*-
# @Author : liang
# @File : s2s.py


import torch.nn as nn
from torch_geometric.nn import Set2Set


class Set2set(nn.Module):
    def __init__(self, args):
        """
        Initialize the Set2set module.

        Args:
            args (object): An object containing configuration param.
                - node_size (int): The number of feature channels for nodes.
        """
        super(Set2set, self).__init__()
        self.s2s = Set2Set(args.node_size, processing_steps=3)  # Set2Set module with 3 processing steps

    def forward(self, x, batch):
        """
        Forward pass of the Set2set module.

        Args:
            x (Tensor): The node feature tensor.
            batch (Tensor): The batch index tensor.

        Returns:
            Tensor: The aggregated global feature vector.
        """
        x = self.s2s(x, batch)  # Apply Set2Set pooling
        return x