# -*- coding: utf-8 -*-
# @Author : liang
# @File : gcn.py

import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, node_size, bond_size, extend_dim, dropout):
        """
        Initialize the GCN model.

        Args:
            node_size (int): Size of the node features.
            bond_size (int): Size of the edge features (not used in this implementation).
            extend_dim (int): Extension dimension for the node features.
            dropout (float): Dropout rate for the normalization block.
        """

        super(GCN, self).__init__()

        hidden_dim = node_size * extend_dim

        self.gconv = GCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            improved=False,      # Use the original GCN normalization
            cached=False,        # Do not cache the normalized adjacency matrix
            add_self_loops=True, # Add self-loops to preserve central node info
            normalize=True,      # Apply symmetric normalization
            bias=True
        )

        self.norm_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout)
        )

        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, h, batch):
        """
        Forward pass of the GCN model.

        Args:
            x (torch.Tensor): Node feature tensor.
            edge_index (torch.Tensor): Edge index tensor.
            edge_attr (torch.Tensor): Edge attribute tensor (not used in this implementation).
            h (torch.Tensor): Hidden state tensor (not used in this implementation).
            batch (torch.Tensor): Batch index tensor (not used in this implementation).

        Returns:
            torch.Tensor: Output tensor after applying the GCN layer.
            torch.Tensor: Hidden state tensor (unchanged).
        """
        identity = x  # Save the input tensor for residual connection

        if hasattr(self, 'norm_block') and hasattr(self, 'gconv'):
            x = self.norm_block(x)  # Apply normalization and dropout
            x = self.gconv(x, edge_index)  # Apply graph convolution

        x = x + identity  # Add residual connection
        x = self.act(x)  # Apply activation function
        return x, h

    @property
    def name(self):
        """
        Property to return the name of the model.

        Returns:
            str: Name of the model.
        """
        return 'GCN'