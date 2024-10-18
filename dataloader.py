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
parser.add_argument('--root', type=str, default='MoleculeNet', help='MoleculeNet, DrugBank, BIOSNAP, LITPCBA, CoCrystal')
parser.add_argument('--dataset', default='tox21', type=str)
parser.add_argument('--split', default='smi', type=str, help='smi, smi1, smi2, random') #  MoleculeNet and LITPCBA we use smi, DrugBank, BIOSNAP and CoCrystal we use smi1, smi1
parser.add_argument('--seed', default=4567, type=int, help='1234, 4567, 7890')
parser.add_argument('--device', default='cuda', type=str)
args = parser.parse_args()
print(args)

exec("dataset = {}Dataset('./dataset/{}', '{}', '{}', {})".format(args.root, args.root, args.dataset, args.split, args.seed))
# dataset = MolDataset(root='../dataset/' + args.root, name=args.dataset, split_seed=args.seed)
print('the dataset name: {}, the mol size: {}.'.format(args.dataset, len(dataset)))

