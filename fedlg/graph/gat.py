# -*- coding: utf-8 -*-
# @Author : liang
# @File : gat.py


import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATLayer(nn.Module):
    """
    A single Graph Attention Network (GAT) layer wrapper.

    This module wraps `torch_geometric.nn.GATConv` to provide a clean interface
    for applying multi-head graph attention on node features.

    Parameters
    ----------
    embedding_size : int
        Input feature dimension for each node.
    output_size : int
        Output feature dimension for each node after attention.

    Attributes
    ----------
    attn : GATConv
        The underlying graph attention layer from PyTorch Geometric.
    """

    def __init__(self, embedding_size: int, output_size: int):
        super(GATLayer, self).__init__()
        self.attn = GATConv(
            in_channels=embedding_size,
            out_channels=output_size,
            heads=1,  # Single-head attention by default; increase for multi-head
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            bias=True,
            add_self_loops=True,
        )

    def forward(self, x, edge_index, edge_attr=None, h=None, batch=None):
        """
        Forward pass of the GAT layer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape `(num_nodes, embedding_size)`.
        edge_index : torch.Tensor
            Graph connectivity in COO format, shape `(2, num_edges)`.
        edge_attr : torch.Tensor, optional
            Edge feature matrix of shape `(num_edges, num_edge_features)`.
            (Not used by GATConv but kept for API consistency.)
        h : torch.Tensor, optional
            Hidden state from previous layers (unused here, placeholder for RNN variants).
        batch : torch.Tensor, optional
            Batch vector assigning each node to a specific graph in a batched input.
            (Not used by GATConv but kept for API consistency.)

        Returns
        -------
        torch.Tensor
            Output node features after attention, shape `(num_nodes, output_size)`.
        """
        return self.attn(x, edge_index)


class GAT(nn.Module):
    def __init__(self, node_size, bond_size, extend_dim, dropout):
        """
        Initialize the GAT model.

        Args:
            node_size (int): Size of the node features.
            bond_size (int): Size of the edge features (not used in this implementation).
            extend_dim (int): Extension dimension for the node features.
            dropout (float): Dropout rate for the normalization block.
        """

        super(GAT, self).__init__()
        self.gatconv = GATConv(
            in_channels=node_size * extend_dim,
            out_channels=node_size * extend_dim,
            heads=1,  # Single-head attention by default; increase for multi-head
            concat=True,
            negative_slope=0.2,
            dropout= dropout,
            bias=True,
            add_self_loops=True,
        )
        # self.gatconv = GATConv(node_size * extend_dim, node_size * extend_dim)

        self.norm_block = nn.Sequential(
            nn.LayerNorm(node_size * extend_dim),
            nn.Dropout(p=dropout)
        )
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, h, batch):
        """
        Forward pass of the GAT model.

        Args:
            x (torch.Tensor): Node feature tensor.
            edge_index (torch.Tensor): Edge index tensor.
            edge_attr (torch.Tensor): Edge attribute tensor (not used in this implementation).
            h (torch.Tensor): Hidden state tensor (not used in this implementation).
            batch (torch.Tensor): Batch index tensor (not used in this implementation).

        Returns:
            torch.Tensor: Output tensor after applying the GAT layer.
            torch.Tensor: Hidden state tensor (unchanged).
        """
        identity = x  # Save the input tensor for residual connection

        if hasattr(self, 'norm_block') and hasattr(self, 'gatconv'):
            x = self.norm_block(x)  # Apply normalization and dropout
            x = self.gatconv(x, edge_index)  # Apply graph attention

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
        return 'GAT'