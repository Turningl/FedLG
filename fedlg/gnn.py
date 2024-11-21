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
        super(Mol_architecture, self).__init__()

        self.mol_lin = nn.Sequential(
            nn.Linear(args.node_size, args.hidden_size * args.extend_dim, bias=True),
            nn.RReLU())

        # getattr
        self.model = globals()[args.model](args.hidden_size, args.bond_size, args.extend_dim, args.dropout)

        self.mol_readout = GlobalPool(args)

        self.message_steps = args.message_steps

        self.mol_flat = nn.Sequential(
            nn.Linear(args.hidden_size * args.extend_dim * 5, args.hidden_size * args.extend_dim, bias=True),
            nn.RReLU())

        self.mol_out = nn.Sequential(
            nn.Linear(args.hidden_size * args.extend_dim, args.output_size, bias=True),
            nn.RReLU())

        self.reset_parameters()


    def reset_parameters(self):
        for layer in self.mol_lin:
            if hasattr(layer, 'weight') and layer.weight is not None:
                glorot_orthogonal(layer.weight, scale=2.0)

            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.fill_(0)

        for sequential_layer in [self.mol_flat, self.mol_out]:
            for layer in sequential_layer:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

                elif hasattr(layer, 'weight') and layer.weight is not None:
                    glorot_orthogonal(layer.weight, scale=2.0)

                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, dataset):
        x, edge_index, edge_attr, batch = dataset.x, dataset.edge_index, dataset.edge_attr, dataset.batch

        x = self.mol_lin(x)

        if hasattr(self, 'model'):
            hmol = None
            for i in range(self.message_steps):
                x, hmol = self.model(x, edge_index, edge_attr, h=hmol, batch=batch)

        x = self.mol_readout(x, batch)
        x = self.mol_flat(x)
        x = self.mol_out(x)

        return x


class DMol_architecture(nn.Module):
    def __init__(self, args):
        super(DMol_architecture, self).__init__()
        self.message_steps = 3

        self.mol1_lin0 = nn.Sequential(
            nn.Linear(args.node_size, args.node_size * args.extend_dim, bias=True),
            nn.RReLU())

        self.mol2_lin0 = nn.Sequential(
            nn.Linear(args.node_size,args.node_size * args.extend_dim, bias=True),
            nn.RReLU())

        self.mol1_conv = globals()[args.model](args.hidden_size, args.bond_size, args.extend_dim, args.dropout)

        self.mol2_conv = globals()[args.model](args.hidden_size, args.bond_size, args.extend_dim, args.dropout)

        self.mol1_readout, self.mol2_readout = GlobalPool(args), GlobalPool(args)

        self.mol1_flat = nn.Linear(args.node_size * args.extend_dim * 5,
                                   args.node_size * args.extend_dim)
        self.mol2_flat = nn.Linear(args.node_size * args.extend_dim * 5,
                                   args.node_size * args.extend_dim)

        self.lin_out1 = nn.Linear(args.node_size * args.extend_dim * 2 + self.message_steps * 2,
                                  args.hidden_size)
        self.lin_out2 = nn.Linear(args.hidden_size,
                                  args.output_size)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mol1_lin0:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=2.0)
            if hasattr(layer, 'bias'):
                layer.bias.data.fill_(0)

        for layer in self.mol2_lin0:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=2.0)
            if hasattr(layer, 'bias'):
                layer.bias.data.fill_(0)

        if hasattr(self, 'mol1_conv') and hasattr(self.mol1_conv, 'reset_parameters'):
            self.mol1_conv.reset_parameters()
        if hasattr(self, 'mol2_conv') and hasattr(self.mol2_conv, 'reset_parameters'):
            self.mol2_conv.reset_parameters()

        glorot_orthogonal(self.mol1_flat.weight, scale=2.0)
        self.mol1_flat.bias.data.fill_(0)
        glorot_orthogonal(self.mol2_flat.weight, scale=2.0)
        self.mol2_flat.bias.data.fill_(0)

        glorot_orthogonal(self.lin_out1.weight, scale=2.0)
        self.lin_out1.bias.data.fill_(0)
        glorot_orthogonal(self.lin_out2.weight, scale=2.0)
        self.lin_out2.bias.data.fill_(0)

    def forward(self, mol1, mol2):
        xm1 = self.mol1_lin0(mol1.x)
        xm2 = self.mol2_lin0(mol2.x)

        fusion = []
        if hasattr(self, 'mol1_conv') and hasattr(self, 'mol2_conv'):
            hmol1, hmol2 = None, None
            for i in range(self.message_steps):
                xm1, hmol1 = self.mol1_conv(xm1, mol1.edge_index, mol1.edge_attr, h=hmol1, batch=mol1.batch)
                xm2, hmol2 = self.mol2_conv(xm2, mol2.edge_index, mol2.edge_attr, h=hmol2, batch=mol2.batch)
                fusion.append(dot_and_global_pool(xm1, xm2, mol1.batch, mol2.batch))

        outm1 = self.mol1_readout(xm1, mol1.batch)
        outm2 = self.mol2_readout(xm2, mol2.batch)
        outm1 = self.mol1_flat(outm1)
        outm2 = self.mol2_flat(outm2)

        out = self.lin_out1(torch.cat([outm1, outm2,
                                       torch.cat(fusion, dim=-1)],
                            dim=-1))
        out = self.lin_out2(out)
        return out


class GCN(nn.Module):
    def __init__(self, node_size, bond_size, extend_dim, dropout):
        super(GCN, self).__init__()
        self.gconv = gcnconv(node_size * extend_dim,
                             node_size * extend_dim)

        self.norm_block = nn.Sequential(
            nn.LayerNorm(node_size * extend_dim),
            nn.Dropout(p=dropout))

        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, h, batch):
        identity = x

        if hasattr(self, 'norm_block') and hasattr(self, 'gconv'):
            x = self.norm_block(x)
            x = self.gconv(x, edge_index)

        x = x + identity
        x = self.act(x)
        return x, h

    @property
    def name(self):
        return 'GCN'


class GATLayer(nn.Module):
    def __init__(self, embedding_size, output_size):
        super(GATLayer, self).__init__()
        self.attn = gatconv(embedding_size, output_size)

    def forward(self, x, edge_index, edge_attr, h, batch):
        return self.attn(x, edge_index)


class GAT(nn.Module):
    def __init__(self, node_size, bond_size, extend_dim, dropout):
        super(GAT, self).__init__()
        self.gatconv = gatconv(node_size * extend_dim,
                               node_size * extend_dim)

        self.norm_block = nn.Sequential(
            nn.LayerNorm(node_size * extend_dim),
            nn.Dropout(p=dropout))

        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, h, batch):
        identity = x

        if hasattr(self, 'norm_block') and hasattr(self, 'gatconv'):
            x = self.norm_block(x)
            x = self.gatconv(x, edge_index)

        x = x + identity
        x = self.act(x)
        return x, h

    @property
    def name(self):
        return 'GAT'


class SAGE(nn.Module):
    def __init__(self, node_size, bond_size, extend_dim, dropout):
        super(SAGE, self).__init__()
        self.sgconv = sageconv(node_size,
                               node_size * extend_dim)

    def forward(self, x, edge_index, edge_attr, h, batch):
        identity = x

        if hasattr(self, 'norm_block') and hasattr(self, 'sgconv'):
            x = self.norm_block(x)
            x = self.sgconv(x, edge_index)

        x = x + identity
        x = self.act(x)
        return x, h

    @property
    def name(self):
        return 'SAGE'


class TripletMessage(MessagePassing):
    def __init__(self, node_channels, edge_channels, heads=3, negative_slope=0.2, **kwargs):
        super(TripletMessage, self).__init__(aggr='add', node_dim=0, **kwargs)  # aggr='mean'
        # node_dim = 0 for multi-head aggr support
        self.node_channels = node_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_node = Parameter(torch.Tensor(node_channels, heads * node_channels))
        self.weight_edge = Parameter(torch.Tensor(edge_channels, heads * node_channels))
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 3 * node_channels))
        self.weight_scale = Parameter(torch.Tensor(heads * node_channels, node_channels))
        self.bias = Parameter(torch.Tensor(node_channels))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight_node)
        kaiming_uniform_(self.weight_edge)
        kaiming_uniform_(self.weight_triplet_att)
        kaiming_uniform_(self.weight_scale)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = torch.matmul(x, self.weight_node)
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.node_channels)
        x_i = x_i.view(-1, self.heads, self.node_channels)
        e_ij = edge_attr.view(-1, self.heads, self.node_channels)

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)
        # return x_j * alpha
        # return self.prelu(alpha * e_ij * x_j)
        return alpha * e_ij * x_j

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.node_channels)
        aggr_out = torch.matmul(aggr_out, self.weight_scale)
        aggr_out = aggr_out + self.bias
        return aggr_out

    def extra_repr(self):
        return '{node_channels}, {node_channels}, heads={heads}'.format(**self.__dict__)


class MPNN(nn.Module):
    def __init__(self, node_size, bond_size, extend_dim, dropout):
        super(MPNN, self).__init__()
        self.gru = nn.GRU(node_size * extend_dim,
                          node_size * extend_dim)

        self.norm_block = nn.Sequential(
            nn.LayerNorm(node_size * extend_dim),
            nn.Dropout(p=dropout))

        self.conv = TripletMessage(node_size * extend_dim,
                                   bond_size)

        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, h=None, batch=None):
        identity = x
        if h is None: h = x.unsqueeze(0)
        x = self.norm_block(x)

        # message passing and update
        x = self.conv(x, edge_index, edge_attr)

        if hasattr(self, 'gru'):
            x = torch.celu(x)
            out, h = self.gru(x.unsqueeze(0), h)
            x = out.squeeze(0)

        x = x + identity
        x = self.act(x)
        return x, h

    @property
    def name(self):
        return 'MPNN'


class GlobalPool(torch.nn.Module):
    def __init__(self, args):
        super(GlobalPool, self).__init__()
        self.args = args

    def forward(self, x, batch):
        if self.args.model == 'AttentiveFP':
            mean, sum, topk = (global_mean_pool(x, batch.unique()),
                               global_add_pool(x, batch.unique()),
                               global_sort_pool(x, batch.unique(),
                                                k=3))
        else:
            mean, sum, topk = (global_mean_pool(x, batch),
                               global_add_pool(x, batch),
                               global_sort_pool(x, batch, k=3))

        return torch.cat([mean, sum, topk], -1)

    @property
    def name(self):
        return 'GlobalPool'


class Set2set(nn.Module):
    def __init__(self, args):
        super(Set2set, self).__init__()
        self.s2s = Set2Set(args.node_size, processing_steps=3)

    def forward(self, x, batch):
        x = self.s2s(x, batch)
        return x

    @property
    def name(self):
        return 'Set2set'


class Linear_BatchNorm1d(nn.Module):
    def __init__(self, node_dim, output_dim):
        super(Linear_BatchNorm1d, self).__init__()
        self.linear = nn.Linear(node_dim, output_dim)
        self.batchnorm1d = nn.BatchNorm1d(output_dim, eps=1e-06, momentum=0.1)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.batchnorm1d(x)

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
#         self.norm1 = nn.Sequential(Linear_BatchNorm1d(4 * args.hidden_size, s2y4.png),
#                                    nn.Dropout(p=args.dropout))
#
#         self.norm2 = nn.Sequential(Linear_BatchNorm1d(2 * args.hidden_size, s2y4.png),
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
#         # neighour_node = node[edge_index[s2y4.png]]
#         neighour = torch.cat([bond, node[edge_index[s2y4.png]], node[edge_index[s2y4.png]] + bond - node[edge_index[s2y4.png]] * bond], dim=-s2y4.png)
#         feature_concat = torch.cat([node[edge_index[0]], neighour], dim=-s2y4.png)
#         attention_weight = nn.functional.softmax(nn.functional.leaky_relu(self.norm1(feature_concat)), dim=s2y4.png)
#         context = scatter_add(torch.mul(attention_weight, neighour), index=edge_index[0], dim=0, dim_size=x.shape[0])
#         node = self.gru1(nn.functional.elu(context), node)
#         node = self.attention(node, edge_index, edge_attr, batch)
#
#         # supernode_num = batch.max() + s2y4.png
#         supernode = scatter_add(node, batch, dim=0, dim_size=batch.max() + s2y4.png)
#         e1 = nn.functional.leaky_relu(self.norm2(torch.cat([supernode[batch], node], dim=-s2y4.png)))
#         attention_weight_sa = nn.functional.softmax(e1, dim=-s2y4.png)
#         context_sa = scatter_add(torch.mul(attention_weight_sa, node), index=batch, dim=0, dim_size=batch.max() + s2y4.png)
#         update = self.output(self.gru2(nn.functional.elu(context_sa), supernode))
#
#         return update


def dot_and_global_pool(mol_out1, mol_out2, mol_batch1, mol_batch2):
    mol_node_slice1 = torch.cumsum(torch.from_numpy(np.bincount(mol_batch1.cpu())), 0)
    mol_node_slice2 = torch.cumsum(torch.from_numpy(np.bincount(mol_batch2.cpu())), 0)
    batch_size = mol_batch1.max() + 1
    out = mol_out1.new_zeros([batch_size, 2])

    for i in range(batch_size):
        mol_start1 = mol_node_slice1[i - 1].item() if i != 0 else 0
        mol_start2 = mol_node_slice2[i - 1].item() if i != 0 else 0
        mol_end1 = mol_node_slice1[i].item()
        mol_end2 = mol_node_slice2[i].item()
        item = torch.matmul(mol_out1[mol_start1:mol_end1], mol_out2[mol_start2:mol_end2].T)
        out[i] = torch.stack([item.max(), item.mean()])
    return out
