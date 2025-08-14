# -*- coding: utf-8 -*-
# @Author : liang
# @File : mpnn.py


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter.scatter import *
from torch_geometric.nn.conv import MessagePassing
from torch.nn.init import kaiming_uniform_, zeros_


class TripletMessage(MessagePassing):
    def __init__(self, node_channels, edge_channels, heads=3, negative_slope=0.2, **kwargs):
        """
        Initialize the TripletMessage module.

        Args:
            node_channels (int): The number of feature channels for nodes.
            edge_channels (int): The number of feature channels for edges.
            heads (int, optional): The number of attention heads. Defaults to 3.
            negative_slope (float, optional): The negative slope for the LeakyReLU activation. Defaults to 0.2.
            **kwargs: Additional keyword arguments passed to the MessagePassing base class.
        """

        super(TripletMessage, self).__init__(aggr='add', node_dim=0, **kwargs)  # Use 'add' aggregation
        self.node_channels = node_channels
        self.heads = heads
        self.negative_slope = negative_slope

        # Initialize learnable param
        self.weight_node = Parameter(torch.Tensor(node_channels, heads * node_channels))
        self.weight_edge = Parameter(torch.Tensor(edge_channels, heads * node_channels))
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 3 * node_channels))
        self.weight_scale = Parameter(torch.Tensor(heads * node_channels, node_channels))
        self.bias = Parameter(torch.Tensor(node_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the learnable param using Kaiming uniform initialization.
        """

        kaiming_uniform_(self.weight_node)
        kaiming_uniform_(self.weight_edge)
        kaiming_uniform_(self.weight_triplet_att)
        kaiming_uniform_(self.weight_scale)
        zeros_(self.bias)


    def forward(self, x, edge_index, edge_attr, size=None):
        """
        The forward pass of the TripletMessage module.

        Args:
            x (Tensor): The node feature tensor.
            edge_index (Tensor): The edge index tensor.
            edge_attr (Tensor): The edge attribute tensor.
            size (tuple, optional): The size of the node feature tensor. Defaults to None.

        Returns:
            Tensor: The updated node feature tensor.
        """
        # Transform node and edge features
        x = torch.matmul(x, self.weight_node)
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        # Propagate messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        """
        Compute the message for each edge.

        Args:
            x_j (Tensor): The source node features.
            x_i (Tensor): The target node features.
            edge_index_i (Tensor): The edge index tensor for the target nodes.
            edge_attr (Tensor): The edge attributes.
            size_i (int): The number of target nodes.

        Returns:
            Tensor: The computed message tensor.
        """
        # Reshape node and edge features for multi-head attention
        x_j = x_j.view(-1, self.heads, self.node_channels)
        x_i = x_i.view(-1, self.heads, self.node_channels)
        e_ij = edge_attr.view(-1, self.heads, self.node_channels)

        # Concatenate source node, edge, and target node features
        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)

        # Compute attention coefficients
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=0)  # Apply softmax over the edges

        # Reshape attention coefficients
        alpha = alpha.view(-1, self.heads, 1)

        # Compute the message
        return alpha * e_ij * x_j

    def update(self, aggr_out):
        """
        Update the node features based on aggregated messages.

        Args:
            aggr_out (Tensor): The aggregated message tensor.

        Returns:
            Tensor: The updated node feature tensor.
        """
        # Reshape aggregated messages
        aggr_out = aggr_out.view(-1, self.heads * self.node_channels)

        # Apply scaling and bias
        aggr_out = torch.matmul(aggr_out, self.weight_scale)
        aggr_out = aggr_out + self.bias

        return aggr_out

    def extra_repr(self):
        """
        Return a string representation of the module.
        """
        return '{node_channels}, {node_channels}, heads={heads}'.format(**self.__dict__)


class MPNN(nn.Module):
    def __init__(self, node_size, bond_size, extend_dim, dropout):
        """
        Initialize the MPNN model.

        Args:
            node_size (int): Size of the node features.
            bond_size (int): Size of the edge features.
            extend_dim (int): Extension dimension for the node features.
            dropout (float): Dropout rate for the normalization block.
        """
        super(MPNN, self).__init__()
        self.gru = nn.GRU(node_size * extend_dim, node_size * extend_dim)  # GRU for node updates
        self.norm_block = nn.Sequential(
            nn.LayerNorm(node_size * extend_dim),  # Normalization layer
            nn.Dropout(p=dropout)  # Dropout layer
        )
        self.conv = TripletMessage(node_size * extend_dim, bond_size)  # Triplet-based message passing module
        self.act = nn.ReLU()  # Activation function

    def forward(self, x, edge_index, edge_attr, h=None, batch=None):
        """
        Forward pass of the MPNN model.

        Args:
            x (torch.Tensor): Node feature tensor.
            edge_index (torch.Tensor): Edge index tensor.
            edge_attr (torch.Tensor): Edge attribute tensor.
            h (torch.Tensor, optional): Hidden state tensor for the GRU. Defaults to None.
            batch (torch.Tensor, optional): Batch index tensor. Defaults to None.

        Returns:
            torch.Tensor: Updated node feature tensor.
            torch.Tensor: Updated hidden state tensor.
        """
        identity = x  # Save the input tensor for residual connection

        if h is None:
            h = x.unsqueeze(0)  # Initialize hidden state if not provided

        x = self.norm_block(x)  # Apply normalization and dropout

        # Message passing using the TripletMessage module
        x = self.conv(x, edge_index, edge_attr)

        if hasattr(self, 'gru'):
            x = torch.celu(x)  # Apply celu activation before GRU
            out, h = self.gru(x.unsqueeze(0), h)  # Update node features using GRU
            x = out.squeeze(0)  # Remove the sequence dimension

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
        return 'MPNN'