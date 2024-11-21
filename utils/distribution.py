# -*- coding: utf-8 -*-
# @Author : liang
# @File : distribution.py


import os
import torch
import numpy as np
from utils.chemutils import scaffold_split
from collections import defaultdict


def molecule_dirichlet_distribution(args, dataset, sample_clients, alpha, null_value=-1, seed=1234):
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

    # min_size, minNumperCleint = 0, 3 or 4
    scaffold_unique = np.unique(scaffolds_label)
    assert len(scaffold_unique) == len(scaffolds.keys()), f'The scaffold number is incorrect, please check source code.'
    scaffold_split_map = defaultdict(list)

    index_batch = list([] for _ in range(sample_clients))

    for idx in range(len(scaffold_unique) + 1):
        idx_label = np.where(scaffolds_label == idx)[0]
        np.random.seed(seed)
        np.random.shuffle(idx_label)

        proportions1 = np.random.dirichlet(np.repeat(alpha, sample_clients))
        proportions2 = np.array([p * (len(idx_j) < len(scaffolds_label) / sample_clients) for p, idx_j in zip(proportions1, index_batch)])
        proportions = (np.cumsum(proportions2 / proportions2.sum()) * len(idx_label)).astype(int)[:-1]

        index_batch = [idx_j + idx.tolist() for idx_j, idx in zip(index_batch, np.split(idx_label, proportions))]

    for j in range(sample_clients):
        np.random.shuffle(index_batch[j])
        scaffold_split_map[j] = index_batch[j]

    torch.save(scaffold_split_map, storage_path)
    return scaffold_split_map


def random_distribution(args, dataset, sample_clients, alpha, null_value=-1, seed=1234):
    storage_path = os.path.join(os.path.dirname(args.root + '/processed/'), '{}_scaffold_split_map_{}.pt'.format(args.dataset, seed))

    if os.path.exists(storage_path):
        scaffold_split_map = torch.load(storage_path)
        return scaffold_split_map

    total_num = len(dataset)
    idxs = np.random.permutation(total_num)
    batch_idxs = np.array_split(idxs, sample_clients)
    scaffold_split_map = {i: batch_idxs[i] for i in range(sample_clients)}

    torch.save(scaffold_split_map, storage_path)
    return scaffold_split_map
