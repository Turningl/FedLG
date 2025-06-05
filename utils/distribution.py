# -*- coding: utf-8 -*-
# @Author : liang
# @File : distribution.py


import os
import torch
import numpy as np
from utils.chemutils import scaffold_split
from collections import defaultdict


def molecule_dirichlet_distribution(args, dataset, sample_clients, alpha, null_value=-1, seed=1234):
    """
    Args:
        args (object): Configuration arguments.
        dataset (torch.utils.data.Dataset): The dataset to split.
        sample_clients (int): Number of clients to sample.
        alpha (float): Dirichlet distribution parameter.
        null_value (int, optional): Null value for scaffolds. Defaults to -1.
        seed (int, optional): Random seed. Defaults to 1234.

    Returns:
        dict: A dictionary mapping client indices to their respective data indices.
    """
    storage_path = os.path.join(os.path.dirname('./dataset/' + args.root + '/processed/'), '{}_scaffold_split_map_{}.pt'.format(args.dataset, seed))

    if os.path.exists(storage_path):
        scaffold_split_map = torch.load(storage_path)
        return scaffold_split_map

    if args.split == 'smi':
        scaffolds, scaffolds_label = scaffold_split(dataset, dataset.data.smi, null_value=null_value, seed=seed)
    elif args.split == 'smi1':
        scaffolds, scaffolds_label = scaffold_split(dataset, dataset.data.smi1, null_value=null_value, seed=seed)
    elif args.split == 'smi2':
        scaffolds, scaffolds_label = scaffold_split(dataset, dataset.data.smi2, null_value=null_value, seed=seed)
    else:
        return 'please choose a split style!'

    # Identify unique scaffolds and ensure the number of unique scaffolds matches the expected count.
    scaffold_unique = np.unique(scaffolds_label)
    assert len(scaffold_unique) == len(scaffolds.keys()), f'The scaffold number is incorrect, please check source code.'

    # Initialize a dictionary to store the scaffold split map.
    scaffold_split_map = defaultdict(list)

    # Initialize a list of empty lists, one for each client.
    index_batch = list([] for _ in range(sample_clients))

    # Iterate over each unique scaffold index.
    for idx in range(len(scaffold_unique) + 1):
        # Get the indices of the current scaffold.
        idx_label = np.where(scaffolds_label == idx)[0]

        # Shuffle the indices to ensure randomness.
        np.random.seed(seed)
        np.random.shuffle(idx_label)

        # Compute the Dirichlet distribution proportions for each client.
        proportions1 = np.random.dirichlet(np.repeat(alpha, sample_clients))

        # Adjust proportions based on the current size of each client's batch.
        proportions2 = np.array(
            [p * (len(idx_j) < len(scaffolds_label) / sample_clients) for p, idx_j in zip(proportions1, index_batch)])

        # Normalize the proportions and compute the cumulative sum to get the split points.
        proportions = (np.cumsum(proportions2 / proportions2.sum()) * len(idx_label)).astype(int)[:-1]

        # Split the indices according to the computed proportions and add them to each client's batch.
        index_batch = [idx_j + idx.tolist() for idx_j, idx in zip(index_batch, np.split(idx_label, proportions))]

    # Shuffle the indices in each client's batch to ensure randomness.
    for j in range(sample_clients):
        np.random.shuffle(index_batch[j])
        scaffold_split_map[j] = index_batch[j]

    # Save the scaffold split map to the storage path for future use.
    torch.save(scaffold_split_map, storage_path)

    return scaffold_split_map


def random_distribution(args, dataset, sample_clients, alpha, null_value=-1, seed=1234):
    """
    Args:
        args (object): Configuration arguments.
        dataset (torch.utils.data.Dataset): The dataset to split.
        sample_clients (int): Number of clients to sample.
        alpha (float): Not used in this function.
        null_value (int, optional): Not used in this function. Defaults to -1.
        seed (int, optional): Random seed. Defaults to 1234.

    Returns:
        dict: A dictionary mapping client indices to their respective data indices.
    """
    storage_path = os.path.join(os.path.dirname(args.root + '/processed/'), '{}_scaffold_split_map_{}.pt'.format(args.dataset, seed))

    if os.path.exists(storage_path):
        # If the storage path exists, load the scaffold split map from the file.
        scaffold_split_map = torch.load(storage_path)
        # Return the loaded scaffold split map.
        return scaffold_split_map

    # If the storage path does not exist, proceed with the random distribution.
    total_num = len(dataset)  # Get the total number of data points in the dataset.
    idxs = np.random.permutation(total_num)  # Randomly permute the indices of the dataset.
    batch_idxs = np.array_split(idxs, sample_clients)  # Split the permuted indices into batches for each client.
    scaffold_split_map = {i: batch_idxs[i] for i in range(
        sample_clients)}  # Create a dictionary mapping client indices to their respective data indices.

    # Save the scaffold split map to the storage path for future use.
    torch.save(scaffold_split_map, storage_path)
    # Return the scaffold split map.
    return scaffold_split_map
