# -*- coding: utf-8 -*-
# @Author : liang
# @File : gnn.py


import torch
import torch.nn as nn
from torch_scatter.scatter import *
from torch_geometric.nn.inits import glorot_orthogonal

from fedlg.graph import (
    MPNN, GAT, GCN,
    Set2Set, GlobalPool,
    dot_and_global_pool,
    Linear_BatchNorm1d
)


class Mol_architecture(nn.Module):
    def __init__(self, args):
        """
        Initialize the Mol_architecture model.

        Args:
            args (object): An object containing configuration param.
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

        # Initialize model param
        self.reset_parameters()


    def reset_parameters(self):
        """
        Reset the model param using Glorot orthogonal initialization.
        """
        # Initialize param of the initial linear layer
        for layer in self.mol_lin:
            if hasattr(layer, 'weight') and layer.weight is not None:
                glorot_orthogonal(layer.weight, scale=2.0)  # Glorot orthogonal initialization
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.fill_(0)  # Initialize bias to zero

        # Initialize param of the final linear layers
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
            args (object): An object containing configuration param.
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

        # Initialize model param
        self.reset_parameters()


    def reset_parameters(self):
        """
        Reset the model param using Glorot orthogonal initialization.
        """

        # Initialize param of the initial linear layers for the first molecule
        for layer in self.mol1_lin0:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=2.0)  # Glorot orthogonal initialization
            if hasattr(layer, 'bias'):
                layer.bias.data.fill_(0)  # Initialize bias to zero

        # Initialize param of the initial linear layers for the second molecule
        for layer in self.mol2_lin0:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=2.0)  # Glorot orthogonal initialization
            if hasattr(layer, 'bias'):
                layer.bias.data.fill_(0)  # Initialize bias to zero

        # Reset param of the GNN models if they have a reset_parameters method
        if hasattr(self, 'mol1_conv') and hasattr(self.mol1_conv, 'reset_parameters'):
            self.mol1_conv.reset_parameters()
        if hasattr(self, 'mol2_conv') and hasattr(self.mol2_conv, 'reset_parameters'):
            self.mol2_conv.reset_parameters()

        # Initialize param of the linear layers after pooling
        glorot_orthogonal(self.mol1_flat.weight, scale=2.0)
        self.mol1_flat.bias.data.fill_(0)
        glorot_orthogonal(self.mol2_flat.weight, scale=2.0)
        self.mol2_flat.bias.data.fill_(0)

        # Initialize param of the output layers
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
