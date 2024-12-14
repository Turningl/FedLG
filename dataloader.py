# -*- coding: utf-8 -*-
# @Author : liang
# @File : dataloader.py


import argparse
import rdkit

import warnings
warnings.filterwarnings('ignore')

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, choices=['MoleculeNet, DrugBank, BIOSNAP, LITPCBA, CoCrystal'],
                        help='choose the dataset, start with the path to dataset dir.')
parser.add_argument('--dataset', type=str,
                    choices=['MoleculeNet: bbbp', 'MoleculeNet: bace', 'MoleculeNet: sider', 'MoleculeNet: tox21',
                             'MoleculeNet: toxcast','MoleculeNet: esol', 'MoleculeNet: lipo', 'MoleculeNet: freesolv',
                             'LIT-PCBA: ALDH1', 'LIT-PCBA: FEN1', 'LIT-PCBA: GBA', 'LIT-PCBA: KAT2A',
                             'LIT-PCBA: MAPK1', 'LIT-PCBA: PKM2', 'LIT-PCBA: VDR',
                             'DrugBank: DrugBank', 'CoCrystal: CoCrystal', 'BIOSNAP: BIOSNAP'],
                    help='dataset is directly related to root.')
parser.add_argument('--split', type=str, choices=['smi, smi1, smi2, random'],
                    help='Choose a data splitting method.')  #  MoleculeNet and LITPCBA we use smi, DrugBank, BIOSNAP and CoCrystal we use smi1, smi2
parser.add_argument('--seed', default=4567, type=int, choices=[1234, 4567, 7890],
                    help='Initialize random number seeds for model training and data splitting.')
parser.add_argument('--device', default='cuda', type=str,
                    choices=['cuda', 'cpu'])
args = parser.parse_args()
print(args)

exec("dataset = {}Dataset('./dataset/{}', '{}', '{}', {})".format(args.root, args.root, args.dataset, args.split, args.seed))
# dataset = MolDataset(root='../dataset/' + args.root, name=args.dataset, split_seed=args.seed)
print('the dataset name: {}, the mol size: {}.'.format(args.dataset, len(dataset)))
