# -*- coding: utf-8 -*-
# @Author : liang
# @File : nnutils.py


import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from utils.chemutils import scaffold_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error
from collections import defaultdict
from utils.chemutils import extract_batch_data


def model_test_classifier(args, model, test):
    model.eval().to(args.device)
    test_dataset = DataLoader(test, batch_size=args.batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    test_loss = []
    test_acc = []

    with torch.no_grad():
        print(' ')
        for step, test_batch in tqdm(enumerate(test_dataset), total=len(test_dataset)):
            if test.related_title == 'DrugBank' or test.related_title == 'BIOSNAP' or test.related_title == 'CoCrystal':
                mol1_test, mol2_test = extract_batch_data(test_dataset.dataset.mol_dataset, test_batch)
                mol1_test, mol2_test, y_test = mol1_test.to(args.device), mol2_test.to(args.device), test_batch.y.to(args.device)
                y_pred = model(mol1_test, mol2_test)

            elif test.related_title == 'MoleculeNet' or test.related_title == 'LITPCBA':
                x_test, y_test = test_batch.to(args.device), test_batch.y.to(args.device)
                y_pred = model(x_test)

            loss = criterion(y_pred, y_test)

            accuracy = []
            for k in range(y_pred.shape[1]):
                try:
                    accuracy.append(
                        roc_auc_score(y_test[:, k].cpu().detach().numpy(), y_pred[:, k].cpu().detach().numpy()))

                    test_loss.append(loss.item())
                    test_acc.append(np.mean(accuracy))

                except Exception as e:
                    # print(e)
                    continue

            # _, test_pred = torch.max(output, 1)
            # correct = (test_pred == y_test).sum()
            # test_acc += correct.item()

    print()
    return np.mean(test_acc), np.mean(test_loss)


def model_test_regression(args, model, test):
    model.eval().to(args.device)
    test_dataset = DataLoader(test, batch_size=args.batch_size, shuffle=True)

    criterion = nn.MSELoss()

    test_loss = []
    test_rmse = []

    with torch.no_grad():
        print(' ')
        for i, test_batch in enumerate(test_dataset):
            try:
                if test.related_title == 'DrugBank' or test.related_title == 'BIOSNAP' or test.related_title == 'CoCrystal':
                    mol1_test, mol2_test = extract_batch_data(test_dataset.dataset.mol_dataset, test_batch)
                    mol1_test, mol2_test, y_test = mol1_test.to(args.device), mol2_test.to(
                        args.device), test_batch.y.to(args.device)
                    y_pred = model(mol1_test, mol2_test)

                elif test.related_title == 'MoleculeNet' or test.related_title == 'LITPCBA':
                    x_test, y_test = test_batch.to(args.device), test_batch.y.to(args.device)
                    y_pred = model(x_test)

                loss = criterion(y_pred, y_test)
                rmse = np.sqrt(mean_squared_error(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))

                test_loss.append(loss.item())
                test_rmse.append(rmse)

            except Exception as e:
                # test_loss.append(0)
                # test_rmse.append(0)
                continue

    print()
    return np.mean(test_rmse), np.mean(test_loss)