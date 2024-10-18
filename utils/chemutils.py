# -*- coding: utf-8 -*-
# @Author : liang
# @File : chemutils.py


from collections import defaultdict
from itertools import compress
import numpy as np
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Batch
from torch_scatter import scatter

try:
    import rdkit
    from rdkit import Chem, RDConfig
    from rdkit.Chem import ChemicalFeatures, MolFromSmiles
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit.Chem.rdchem import BondType
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')

except:
    rdkit, Chem, RDConfig, MolFromSmiles, ChemicalFeatures, HybridizationType, BondType = 7 * [None]
    print('Please install rdkit for data processing')

res_type_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', ]
# 'X' for unkown?
res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
res_aromatic_table = ['F', 'W', 'Y']

res_non_polar_table = ['A', 'G', 'P', 'V', 'L', 'I', 'M']
res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
res_acidic_charged_table = ['D', 'E']  # res_polar_negatively_table = ['D', 'E']
res_basic_charged_table = ['H', 'K', 'R']  # res_polar_positively_table = ['R', 'K', 'H']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
meiler_feature_table = {
    'A': [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
    'C': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    'D': [1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
    'E': [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
    'F': [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
    'G': [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
    'H': [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
    'I': [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
    'K': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
    'L': [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
    'M': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
    'N': [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
    'P': [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
    'Q': [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
    'R': [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
    'S': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
    'T': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
    'V': [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
    'W': [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
    'Y': [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
}
kidera_feature_table = {
    'A': [-1.56, -1.67, -0.97, -0.27, -0.93, -0.78, -0.2, -0.08, 0.21, -0.48],
    'C': [0.12, -0.89, 0.45, -1.05, -0.71, 2.41, 1.52, -0.69, 1.13, 1.1],
    'E': [-1.45, 0.19, -1.61, 1.17, -1.31, 0.4, 0.04, 0.38, -0.35, -0.12],
    'D': [0.58, -0.22, -1.58, 0.81, -0.92, 0.15, -1.52, 0.47, 0.76, 0.7],
    'G': [1.46, -1.96, -0.23, -0.16, 0.1, -0.11, 1.32, 2.36, -1.66, 0.46],
    'F': [-0.21, 0.98, -0.36, -1.43, 0.22, -0.81, 0.67, 1.1, 1.71, -0.44],
    'I': [-0.73, -0.16, 1.79, -0.77, -0.54, 0.03, -0.83, 0.51, 0.66, -1.78],
    'H': [-0.41, 0.52, -0.28, 0.28, 1.61, 1.01, -1.85, 0.47, 1.13, 1.63],
    'K': [-0.34, 0.82, -0.23, 1.7, 1.54, -1.62, 1.15, -0.08, -0.48, 0.6],
    'M': [-1.4, 0.18, -0.42, -0.73, 2.0, 1.52, 0.26, 0.11, -1.27, 0.27],
    'L': [-1.04, 0.0, -0.24, -1.1, -0.55, -2.05, 0.96, -0.76, 0.45, 0.93],
    'N': [1.14, -0.07, -0.12, 0.81, 0.18, 0.37, -0.09, 1.23, 1.1, -1.73],
    'Q': [-0.47, 0.24, 0.07, 1.1, 1.1, 0.59, 0.84, -0.71, -0.03, -2.33],
    'P': [2.06, -0.33, -1.15, -0.75, 0.88, -0.45, 0.3, -2.3, 0.74, -0.28],
    'S': [0.81, -1.08, 0.16, 0.42, -0.21, -0.43, -1.89, -1.15, -0.97, -0.23],
    'R': [0.22, 1.27, 1.37, 1.87, -1.7, 0.46, 0.92, -0.39, 0.23, 0.93],
    'T': [0.26, -0.7, 1.21, 0.63, -0.1, 0.21, 0.24, -1.15, -0.56, 0.19],
    'W': [0.3, 2.1, -0.72, -1.57, -1.16, 0.57, -0.48, -0.4, -2.3, -0.6],
    'V': [-0.74, -0.71, 2.04, -0.4, 0.5, -0.81, -1.07, 0.06, -0.46, 0.65],
    'Y': [1.38, 1.48, 0.8, -0.56, -0.0, -0.68, -0.31, 1.03, -0.05, 0.53],
    # 'UNKNOWN': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
}


def one_of_k_encoding(x, allowable_set):
    """
    One-hot encoding for the input `x` based on the `allowable_set`.

    Args:
    - x: Input element to be encoded.
    - allowable_set: List of allowable elements to encode from.

    Returns:
    - A list of boolean values representing the one-hot encoding.
    """
    if x not in allowable_set:
        pass

    return list(map(lambda s: x == s, allowable_set))


def generate_scaffold(smiles, include_chirality=False):
    """
    Generates a scaffold from the input SMILES string.

    Args:
    - smiles: Input SMILES string.
    - include_chirality: Whether to include chirality information in the scaffold.

    Returns:
    - Scaffold SMILES string.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold


def is_valid_smiles(smi):
    """
    Checks if the input SMILES string is valid.

    Args:
    - smi: Input SMILES string.

    Returns:
    - True if valid, False otherwise.
    """
    try:
        Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
    except:
        print("not successfully processed smiles: ", smi)
        return False
    return True


def canonical(smi):
    """
    Converts the input SMILES to its canonical form.

    Args:
    - smi: Input SMILES string.

    Returns:
    - Canonical SMILES string if successful, None otherwise.
    """
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
    except:
        print("failed smiles: ", smi, end='\t')
        return None


def smiles_from_MolBlock(num, mol_block):
    """
    Generates a SMILES string from a MolBlock representation.

    Args:
    - num: Identifier or index for the MolBlock.
    - mol_block: Input MolBlock string representation.

    Returns:
    - SMILES string if successful, None otherwise.
    """
    try:
        mol = Chem.MolFromMolBlock(mol_block, removeHs=True)
        smiles = Chem.MolToSmiles(mol, canonical=True)
    except:
        print("number {} failed mol_block".format(num), end='\t')
        return None
    return smiles


def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0, frac_train=0.8, frac_test=0.2, seed=0):
    np.testing.assert_almost_equal(frac_train + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[:, task_idx].item() for data in dataset])
        # boolean array that correspond to non-null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset.data.y)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    # rng = np.random.RandomState(seed)
    # scaffold_sets = rng.permutation(list(scaffolds.values()))  new version of numpy will raise error
    scaffold_sets = list(scaffolds.values())
    np.random.shuffle(scaffold_sets)

    n_total_test = int(np.floor(frac_test * len(dataset.data.y)))

    train_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_dataset = dataset[torch.tensor(train_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, test_dataset


def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0, frac_train=0.8, frac_test=0.2, seed=0):
    np.testing.assert_almost_equal(frac_train + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[:, task_idx].item() for data in dataset])
        # boolean array that correspond to non-null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_label = np.zeros(len(dataset), dtype=int)
    for idx, indices in enumerate(scaffolds.values()):
        scaffold_label[indices] = idx

    return scaffolds, scaffold_label


def make_smi_attribute(smi):
    mol = Chem.MolFromSmiles(smi)
    N = mol.GetNumAtoms()
    atom_type = []
    atomic_type = []
    atomic_number = []
    aromatic = []
    hybridization = []

    for atom in mol.GetAtoms():
        atom_type.append(atom.GetSymbol())
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization.append(atom.GetHybridization())

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row.extend([start, end])
        col.extend([end, start])
        edge_type.extend(2 * [bond.GetBondType()])
    edge_index = torch.LongTensor([row, col])
    edge_type = [one_of_k_encoding(edge, [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]) for
                 edge in edge_type]
    edge_attr = torch.FloatTensor(edge_type)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    row, col = edge_index

    hs = (torch.tensor(atomic_number, dtype=torch.long) == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()
    x_atom_type = [one_of_k_encoding(t, ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']) for t in atom_type]
    x_hybridization = [one_of_k_encoding(h, [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3]) for h
                       in hybridization]
    x2 = torch.tensor([atomic_number, aromatic, num_hs], dtype=torch.float).t().contiguous()
    x = torch.cat([torch.FloatTensor(x_atom_type), torch.FloatTensor(x_hybridization), x2], dim=-1)

    return x, edge_index, edge_attr


def extract_batch_data(mol_data, id_batch):
    smis1, smis2, ys = id_batch.smi1, id_batch.smi2, id_batch.y
    mol1_batch_list = [mol_data[smi] for smi in smis1]  # smi[0] for unpack smi from [['smi1'], ['smi2']...]
    mol1_batch = Batch().from_data_list(mol1_batch_list)
    mol2_batch_list = [mol_data[smi] for smi in smis2]
    mol2_batch = Batch().from_data_list(mol2_batch_list)
    return mol1_batch, mol2_batch


def get_pro_nodes_edges(protein_seq, contact_map):
    # add node information
    feat = []
    for residue in protein_seq:
        residue_features = get_residue_features(residue)
        feat.append(residue_features)
    node_attr = torch.FloatTensor(feat)

    # add main_chain information
    m_index_row, m_index_col, m_edge_attr = [], [], []
    for i in range(len(protein_seq) - 1):
        m_index_row += [i, i + 1]
        m_index_col += [i + 1, i]
        m_edge_attr.append([1, 1, 0, 0, 0, 0, 0, 1])  # read the code below about edge feature extract
        m_edge_attr.append([1, 1, 0, 0, 0, 0, 0, 1])

    # read edge features from contactmap.txt
    edge_attr = []
    index_row, index_col = np.where(contact_map > 0)
    index_row, index_col = index_row.tolist(), index_col.tolist()
    for i, j in zip(index_row, index_col):
        main_chain = 0  # int(np.abs(i - j) == s2y4.png)
        prob = contact_map[i, j]
        reversed_prob = 1 - prob
        # prob level range
        l1 = int(0 <= prob < 0.3)
        l2 = int(0.3 <= prob < 0.5)
        l3 = int(0.5 <= prob < 0.7)
        l4 = int(0.5 <= prob < 0.9)
        l5 = int(0.9 <= prob <= 1)
        edge_attr.append([main_chain, prob, reversed_prob, l1, l2, l3, l4, l5])

    edge_index = torch.LongTensor([m_index_row + index_row, m_index_col + index_col])
    edge_attr = torch.FloatTensor(m_edge_attr + edge_attr)
    # print(node_attr.shape, edge_index.shape, edge_attr.shape)
    # assert edge_index.shape[1] == edge_attr.shape[0]
    return node_attr, edge_index, edge_attr


def get_residue_features(residue):
    res_type = one_of_k_encoding(residue, res_type_table)
    res_type = [int(x) for x in res_type]
    res_property1 = [1 if residue in res_aliphatic_table else 0, 1 if residue in res_aromatic_table else 0,
                     1 if residue in res_polar_neutral_table else 0, 1 if residue in res_acidic_charged_table else 0,
                     1 if residue in res_basic_charged_table else 0, ]
    res_property2 = [res_weight_table[residue], res_pka_table[residue],
                     res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue],
                     res_hydrophobic_ph7_table[residue], ]
    res_property3 = meiler_feature_table[residue] + kidera_feature_table[residue]
    return res_type + res_property1 + res_property2 + res_property3
