# -*- coding: utf-8 -*-
# @Author : liang
# @File : gnn.py


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv as gcnconv, GATConv as gatconv, SAGEConv as sageconv, MessagePassing as messagepassing
from torch_geometric.nn import Set2Set
from torch_geometric.nn import global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops
from torch_scatter.scatter import *
from torch_geometric.nn.conv import MessagePassing
from torch.nn.init import kaiming_uniform_, zeros_
from torch_geometric.nn.inits import glorot_orthogonal


class Mol_architecture(nn.Module):
    def __init__(self, args):
        """
        Initialize the Mol_architecture model.

        Args:
            args (object): An object containing configuration parameters.
                - node_size (int): Size of the input node features.
                - hidden_size (int): Size of the hidden layers.
                - extend_dim (int): Extension dimension for the neural network.
                - output_size (int): Size of the output layer.
                - model (str): Name of the custom GNN model to use.
                - bond_size (int): Size of the edge features.
                - dropout (float): Dropout rate for the GNN model.
                - message_steps (int): Number of message passing steps in the GNN model.
        """

        super(Mol_architecture, self).__init__()

        # Initial linear transformation for node features
        self.mol_lin = nn.Sequential(
            nn.Linear(args.node_size, args.hidden_size * args.extend_dim, bias=True),  # Linear layer
            nn.RReLU()  # Randomized ReLU activation function
        )

        # Dynamically load the specified GNN model
        self.model = globals()[args.model](args.hidden_size, args.bond_size, args.extend_dim, args.dropout)

        # Global pooling layer to aggregate node features
        self.mol_readout = GlobalPool(args)

        # Number of message passing steps
        self.message_steps = args.message_steps

        # Linear transformation after pooling
        self.mol_flat = nn.Sequential(
            nn.Linear(args.hidden_size * args.extend_dim * 5, args.hidden_size * args.extend_dim, bias=True),
            nn.RReLU()
        )

        # Final output layer
        self.mol_out = nn.Sequential(
            nn.Linear(args.hidden_size * args.extend_dim, args.output_size, bias=True),
            nn.RReLU()
        )

        # Initialize model parameters
        self.reset_parameters()


    def reset_parameters(self):
        """
        Reset the model parameters using Glorot orthogonal initialization.
        """
        # Initialize parameters of the initial linear layer
        for layer in self.mol_lin:
            if hasattr(layer, 'weight') and layer.weight is not None:
                glorot_orthogonal(layer.weight, scale=2.0)  # Glorot orthogonal initialization
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.fill_(0)  # Initialize bias to zero

        # Initialize parameters of the final linear layers
        for sequential_layer in [self.mol_flat, self.mol_out]:
            for layer in sequential_layer:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()  # Use built-in reset function if available
                elif hasattr(layer, 'weight') and layer.weight is not None:
                    glorot_orthogonal(layer.weight, scale=2.0)  # Glorot orthogonal initialization
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data.fill_(0)  # Initialize bias to zero


    def forward(self, dataset):
        """
        Forward pass of the model.

        Args:
            dataset (object): A dataset object containing the following attributes:
                - x (torch.Tensor): Node feature tensor.
                - edge_index (torch.Tensor): Edge index tensor.
                - edge_attr (torch.Tensor): Edge attribute tensor.
                - batch (torch.Tensor): Batch index tensor.

        Returns:
            torch.Tensor: Output tensor of the model.
        """

        # Extract dataset attributes
        x, edge_index, edge_attr, batch = dataset.x, dataset.edge_index, dataset.edge_attr, dataset.batch

        # Apply initial linear transformation
        x = self.mol_lin(x)

        # Perform message passing steps using the custom GNN model
        if hasattr(self, 'model'):
            hmol = None
            for i in range(self.message_steps):
                x, hmol = self.model(x, edge_index, edge_attr, h=hmol, batch=batch)

        # Apply global pooling to aggregate node features
        x = self.mol_readout(x, batch)

        # Apply final linear transformations
        x = self.mol_flat(x)
        x = self.mol_out(x)

        return x


class DMol_architecture(nn.Module):
    def __init__(self, args):
        """
        Initialize the DMol_architecture model.

        Args:
            args (object): An object containing configuration parameters.
                - node_size (int): Size of the input node features.
                - hidden_size (int): Size of the hidden layers.
                - extend_dim (int): Extension dimension for the neural network.
                - output_size (int): Size of the output layer.
                - model (str): Name of the custom GNN model to use.
                - bond_size (int): Size of the edge features.
                - dropout (float): Dropout rate for the GNN model.
        """

        super(DMol_architecture, self).__init__()
        self.message_steps = 3  # Number of message passing steps

        # Initial linear transformations for the first molecule
        self.mol1_lin0 = nn.Sequential(
            nn.Linear(args.node_size, args.node_size * args.extend_dim, bias=True),  # Linear layer
            nn.RReLU()  # Randomized ReLU activation function
        )

        # Initial linear transformations for the second molecule
        self.mol2_lin0 = nn.Sequential(
            nn.Linear(args.node_size, args.node_size * args.extend_dim, bias=True),  # Linear layer
            nn.RReLU()  # Randomized ReLU activation function
        )

        # GNN models for the first and second molecules
        self.mol1_conv = globals()[args.model](args.hidden_size, args.bond_size, args.extend_dim, args.dropout)
        self.mol2_conv = globals()[args.model](args.hidden_size, args.bond_size, args.extend_dim, args.dropout)

        # Global pooling layers for the first and second molecules
        self.mol1_readout, self.mol2_readout = GlobalPool(args), GlobalPool(args)

        # Linear transformations after pooling for the first and second molecules
        self.mol1_flat = nn.Linear(args.node_size * args.extend_dim * 5, args.node_size * args.extend_dim)
        self.mol2_flat = nn.Linear(args.node_size * args.extend_dim * 5, args.node_size * args.extend_dim)

        # Output layers
        self.lin_out1 = nn.Linear(args.node_size * args.extend_dim * 2 + self.message_steps * 2, args.hidden_size)
        self.lin_out2 = nn.Linear(args.hidden_size, args.output_size)

        # Initialize model parameters
        self.reset_parameters()


    def reset_parameters(self):
        """
        Reset the model parameters using Glorot orthogonal initialization.
        """

        # Initialize parameters of the initial linear layers for the first molecule
        for layer in self.mol1_lin0:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=2.0)  # Glorot orthogonal initialization
            if hasattr(layer, 'bias'):
                layer.bias.data.fill_(0)  # Initialize bias to zero

        # Initialize parameters of the initial linear layers for the second molecule
        for layer in self.mol2_lin0:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=2.0)  # Glorot orthogonal initialization
            if hasattr(layer, 'bias'):
                layer.bias.data.fill_(0)  # Initialize bias to zero

        # Reset parameters of the GNN models if they have a reset_parameters method
        if hasattr(self, 'mol1_conv') and hasattr(self.mol1_conv, 'reset_parameters'):
            self.mol1_conv.reset_parameters()
        if hasattr(self, 'mol2_conv') and hasattr(self.mol2_conv, 'reset_parameters'):
            self.mol2_conv.reset_parameters()

        # Initialize parameters of the linear layers after pooling
        glorot_orthogonal(self.mol1_flat.weight, scale=2.0)
        self.mol1_flat.bias.data.fill_(0)
        glorot_orthogonal(self.mol2_flat.weight, scale=2.0)
        self.mol2_flat.bias.data.fill_(0)

        # Initialize parameters of the output layers
        glorot_orthogonal(self.lin_out1.weight, scale=2.0)
        self.lin_out1.bias.data.fill_(0)
        glorot_orthogonal(self.lin_out2.weight, scale=2.0)
        self.lin_out2.bias.data.fill_(0)

    def forward(self, mol1, mol2):
        """
        Forward pass of the model.

        Args:
            mol1 (object): Dataset object for the first molecule.
            mol2 (object): Dataset object for the second molecule.

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        # Apply initial linear transformations
        xm1 = self.mol1_lin0(mol1.x)
        xm2 = self.mol2_lin0(mol2.x)

        # List to store fusion features
        fusion = []

        # Perform message passing steps for both molecules
        if hasattr(self, 'mol1_conv') and hasattr(self, 'mol2_conv'):
            hmol1, hmol2 = None, None
            for i in range(self.message_steps):
                xm1, hmol1 = self.mol1_conv(xm1, mol1.edge_index, mol1.edge_attr, h=hmol1, batch=mol1.batch)
                xm2, hmol2 = self.mol2_conv(xm2, mol2.edge_index, mol2.edge_attr, h=hmol2, batch=mol2.batch)
                # Compute fusion features by dot product and global pooling
                fusion.append(dot_and_global_pool(xm1, xm2, mol1.batch, mol2.batch))

        # Apply global pooling to aggregate node features for both molecules
        outm1 = self.mol1_readout(xm1, mol1.batch)
        outm2 = self.mol2_readout(xm2, mol2.batch)

        # Apply linear transformations after pooling
        outm1 = self.mol1_flat(outm1)
        outm2 = self.mol2_flat(outm2)

        # Concatenate the features of both molecules and fusion features
        out = self.lin_out1(torch.cat([outm1, outm2, torch.cat(fusion, dim=-1)], dim=-1))

        # Apply the final output layer
        out = self.lin_out2(out)
        return out


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
        self.gconv = gcnconv(node_size * extend_dim, node_size * extend_dim)
        self.norm_block = nn.Sequential(
            nn.LayerNorm(node_size * extend_dim),
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


class GATLayer(nn.Module):
    def __init__(self, embedding_size, output_size):
        super(GATLayer, self).__init__()
        self.attn = gatconv(embedding_size, output_size)

    def forward(self, x, edge_index, edge_attr, h, batch):
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
        self.gatconv = gatconv(node_size * extend_dim, node_size * extend_dim)
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


# class SAGE(nn.Module):
#     def __init__(self, node_size, bond_size, extend_dim, dropout):
#         super(SAGE, self).__init__()
#         self.sgconv = sageconv(node_size,
#                                node_size * extend_dim)
#
#     def forward(self, x, edge_index, edge_attr, h, batch):
#         identity = x
#
#         if hasattr(self, 'norm_block') and hasattr(self, 'sgconv'):
#             x = self.norm_block(x)
#             x = self.sgconv(x, edge_index)
#
#         x = x + identity
#         x = self.act(x)
#         return x, h
#
#     @property
#     def name(self):
#         return 'SAGE'


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

        # Initialize learnable parameters
        self.weight_node = Parameter(torch.Tensor(node_channels, heads * node_channels))
        self.weight_edge = Parameter(torch.Tensor(edge_channels, heads * node_channels))
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 3 * node_channels))
        self.weight_scale = Parameter(torch.Tensor(heads * node_channels, node_channels))
        self.bias = Parameter(torch.Tensor(node_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the learnable parameters using Kaiming uniform initialization.
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


class GlobalPool(torch.nn.Module):
    def __init__(self, args):
        """
        Initialize the GlobalPool module.

        Args:
            args (object): An object containing configuration parameters.
        """
        super(GlobalPool, self).__init__()
        self.args = args

    def forward(self, x, batch):
        """
        Forward pass of the GlobalPool module.

        Args:
            x (Tensor): The node feature tensor.
            batch (Tensor): The batch index tensor.

        Returns:
            Tensor: The aggregated global feature vector.
        """
        # Determine the pooling strategy based on the model name

        if self.args.model == 'AttentiveFP':
            # For AttentiveFP, use unique batch indices for pooling
            mean = global_mean_pool(x, batch.unique())  # Mean pooling
            sum = global_add_pool(x, batch.unique())    # Sum pooling
            topk = global_sort_pool(x, batch.unique(), k=3)  # Top-k pooling (k=3)

        else:
            # For other models, use the original batch indices for pooling
            mean = global_mean_pool(x, batch)  # Mean pooling
            sum = global_add_pool(x, batch)    # Sum pooling
            topk = global_sort_pool(x, batch, k=3)  # Top-k pooling (k=3)

        # Concatenate the results of the different pooling strategies
        return torch.cat([mean, sum, topk], dim=-1)

    @property
    def name(self):
        """
        Property to return the name of the module.

        Returns:
            str: Name of the module.
        """
        return 'GlobalPool'


class Set2set(nn.Module):
    def __init__(self, args):
        """
        Initialize the Set2set module.

        Args:
            args (object): An object containing configuration parameters.
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

    @property
    def name(self):
        """
        Property to return the name of the module.

        Returns:
            str: Name of the module.
        """
        return 'Set2set'


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


# class attentiveFP(nn.Module):
#     def __init__(self, args):
#         super(attentiveFP, self).__init__()
#         self.node_linear = nn.Sequential(Linear_BatchNorm1d(args.node_size, args.hidden_size),
#                                          nn.ReLU(),
#                                          nn.Dropout(p=args.dropout))
#
#         self.edge_linear = nn.Sequential(Linear_BatchNorm1d(args.bond_size, args.hidden_size),
#                                          nn.ReLU(),
#                                          nn.Dropout(p=args.dropout))
#
#         self.norm1 = nn.Sequential(Linear_BatchNorm1d(4 * args.hidden_size, 1),
#                                    nn.Dropout(p=args.dropout))
#
#         self.norm2 = nn.Sequential(Linear_BatchNorm1d(2 * args.hidden_size, 1),
#                                    nn.Dropout(p=args.dropout))
#
#         self.gru1 = nn.GRUCell(3 * args.hidden_size, args.hidden_size)
#         self.gru2 = nn.GRUCell(args.hidden_size, args.hidden_size)
#         self.attention = gatlayer(args.hidden_size, args.hidden_size)
#
#         self.output = nn.Sequential(Linear_BatchNorm1d(args.hidden_size, 16),
#                                     nn.ReLU(inplace=True),
#                                     nn.Dropout(p=args.dropout),
#                                     nn.Linear(16, args.output_size))
#
#     @property
#     def name(self):
#         return 'attentiveFP'
#
#     def forward(self, x, edge_index, edge_attr, batch):
#         node = self.node_linear(x)
#         bond = self.edge_linear(edge_attr)
#
#         # neighour_node = node[edge_index[1]]
#         neighour = torch.cat([bond, node[edge_index[1]], node[edge_index[1]] + bond - node[edge_index[1]] * bond], dim=-1)
#         feature_concat = torch.cat([node[edge_index[0]], neighour], dim=-1)
#         attention_weight = nn.functional.softmax(nn.functional.leaky_relu(self.norm1(feature_concat)), dim=1)
#         context = scatter_add(torch.mul(attention_weight, neighour), index=edge_index[0], dim=0, dim_size=x.shape[0])
#         node = self.gru1(nn.functional.elu(context), node)
#         node = self.attention(node, edge_index, edge_attr, batch)
#
#         # supernode_num = batch.max() + 1
#         supernode = scatter_add(node, batch, dim=0, dim_size=batch.max() + 1)
#         e1 = nn.functional.leaky_relu(self.norm2(torch.cat([supernode[batch], node], dim=-1)))
#         attention_weight_sa = nn.functional.softmax(e1, dim=-1)
#         context_sa = scatter_add(torch.mul(attention_weight_sa, node), index=batch, dim=0, dim_size=batch.max() + 1)
#         update = self.output(self.gru2(nn.functional.elu(context_sa), supernode))
#
#         return update


def dot_and_global_pool(mol_out1, mol_out2, mol_batch1, mol_batch2):
    """
    Compute the dot product between node features of two molecular graphs and apply global pooling.

    This function calculates the dot product between the node features of two molecular graphs,
    and then applies global pooling to aggregate the results for each batch.

    Args:
        mol_out1 (Tensor): Node features of the first molecular graph.
        mol_out2 (Tensor): Node features of the second molecular graph.
        mol_batch1 (Tensor): Batch index tensor for the first molecular graph.
        mol_batch2 (Tensor): Batch index tensor for the second molecular graph.

    Returns:
        Tensor: Aggregated results for each batch, containing the maximum and mean values of the dot products.
    """
    # Compute the cumulative sum of node counts for each batch in mol_batch1 and mol_batch2
    mol_node_slice1 = torch.cumsum(torch.from_numpy(np.bincount(mol_batch1.cpu())), 0)
    mol_node_slice2 = torch.cumsum(torch.from_numpy(np.bincount(mol_batch2.cpu())), 0)

    # Determine the batch size
    batch_size = mol_batch1.max() + 1

    # Initialize the output tensor with zeros
    out = mol_out1.new_zeros([batch_size, 2])

    # Iterate over each batch
    for i in range(batch_size):
        # Compute the start and end indices for the current batch in mol_out1
        mol_start1 = mol_node_slice1[i - 1].item() if i != 0 else 0
        mol_end1 = mol_node_slice1[i].item()

        # Compute the start and end indices for the current batch in mol_out2
        mol_start2 = mol_node_slice2[i - 1].item() if i != 0 else 0
        mol_end2 = mol_node_slice2[i].item()

        # Compute the dot product between the node features of the current batch
        item = torch.matmul(mol_out1[mol_start1:mol_end1], mol_out2[mol_start2:mol_end2].T)

        # Aggregate the results by computing the maximum and mean values of the dot products
        out[i] = torch.stack([item.max(), item.mean()])

    return out
