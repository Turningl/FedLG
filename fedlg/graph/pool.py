# -*- coding: utf-8 -*-
# @Author : liang
# @File : pool.py


import numpy as np
from torch_geometric.nn import global_mean_pool, global_add_pool, global_sort_pool
from torch_scatter.scatter import *


class GlobalPool(torch.nn.Module):
    def __init__(self, args):
        """
        Initialize the GlobalPool module.

        Args:
            args (object): An object containing configuration param.
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