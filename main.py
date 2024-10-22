# -*- coding: utf-8 -*-
# @Author : liang
# @File : main.py


import os
import copy
import torch
import argparse
import numpy as np
import pandas as pd
from fedlg.server import GlobalServer
from fedlg.client import SimulatedDatabase
from fedlg.gnn import Mol_architecture, DMol_architecture
from utils.dataset import MoleculeNetDataset, DrugBankDataset, LITPCBADataset, BIOSNAPDataset, CoCrystalDataset
from utils.distribution import molecule_dirichlet_distribution, random_distribution, prepare_local_differential_privacy
from utils.nnutils import model_test_classifier, model_test_regression
from utils.save_func import save_progress, print_rmse_accoutant, print_accuracy_accoutant
import warnings

warnings.filterwarnings('ignore')


def main(args, dataset, model):
    train, test = dataset.train, dataset.test
    print('Using differential privacy!\n') if args.dp else print('No differential privacy!\n')

    # prepare local dataset follow by dirichlet distribution
    local_indices = molecule_dirichlet_distribution(args, train, args.num_clients, args.alpha, args.null_value, args.seed)

    # set privacy preference
    privacy_preferences = prepare_local_differential_privacy(args, args.num_clients)
    print('privacy preferences: \n', privacy_preferences, '\n')

    # set simulated databases
    simulated_databases = []
    for i in range(args.num_clients):
        simulated_database = SimulatedDatabase(train=train, indices=local_indices[i], args=args)

        # set noise multiplier
        if args.dp:
            epsilon = privacy_preferences[i]
            simulated_database.set_local_differential_privacy(epsilon)
            print('the %d simulated database noise epsilon is %.4f' % ((i + 1), epsilon))

        simulated_databases.append(simulated_database)

    # set server
    server = GlobalServer(model=model, args=args)

    # set open access database
    server.set_open_access_database(privacy_preferences) if args.dp else None

    # init server algorithm
    server.init_alg(alg=args.alg)

    # init global model
    server_model = server.init_global_model()

    # set communication round
    communication_round = args.global_round // args.local_round
    print('the communication_round is %d' % communication_round)

    accuracy_accountant, rmse_accoutant = [], []

    # start communication
    for r in range(communication_round):
        print()
        print('the %d communication round. \n' % (r + 1))

        # local update and aggregate
        for idx, participant in enumerate(simulated_databases):
            print("the %dth participant local update." % (idx + 1))

            # delivery model
            participant.download(copy.deepcopy(server_model))

            # local update
            model_state = participant.local_update()

            # aggregate
            server.aggregate(idx, model_state, args.alg)

        # load average weight
        global_model = server.update()

        # regression
        if dataset.dataset_name in dataset.dataset_names['regression']:
            test_rmse, test_loss = model_test_regression(args, global_model, test)
            print('current global model has test rmse: %.4f  test loss: %.4f' % (test_rmse, test_loss))

            rmse_accoutant.append(test_rmse)

        # classification
        elif dataset.dataset_name in dataset.dataset_names['classification']:
            test_acc, test_loss = model_test_classifier(args, global_model, test)
            print('current global model has test acc: %.4f  test loss: %.4f' % (test_acc, test_loss))

            accuracy_accountant.append(test_acc)
        # print('current global model has test loss: %.4f' % test_loss)
        # accuracy_accountant.append(test_accuracy)

        # torch.save(accuracy_accountant, 'accuracy_accountant.pt')

    if rmse_accoutant:
        optimal_result = print_rmse_accoutant(rmse_accoutant)
        save_progress(args, rmse_accoutant, optimal_result)
        return np.min(rmse_accoutant)

    elif accuracy_accountant:
        optimal_result = print_accuracy_accoutant(accuracy_accountant)
        save_progress(args, accuracy_accountant, optimal_result)
        return np.max(accuracy_accountant)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning Lanczos Graph')
    parser.add_argument('--alg', type=str, choices=['FedAvg, FedProx, FedSGD, FedLG, FedAdam, FedChem'],
                        help='algorithm options, start with the choosed algorithm.')
    parser.add_argument('--root', type=str, choices=['MoleculeNet, DrugBank, BIOSNAP, LITPCBA, CoCrystal'],
                        help='choose the dataset, start with the path to dataset dir.')
    parser.add_argument('--dataset', default='bbbp', type=str)

    parser.add_argument('--node_size', default=16, type=int,
                        help='number of atom size.')
    parser.add_argument('--bond_size', default=16, type=int,
                        help='number of bond size.')
    parser.add_argument('--hidden_size', default=15, type=int,
                        help='initial hidden size.')
    parser.add_argument('--extend_dim', default=4, type=float)
    parser.add_argument('--output_size', default=1, type=int,
                        help='initial output size.')
    parser.add_argument('--model', default='MPNN', type=str, choices=['MPNN, GCN, GAT'],
                        help='Graph model algorithm of MPNN, GCN and GAT.')
    parser.add_argument('--split', default='smi', type=str, choices=['smi, smi1, smi2, random'],
                        help='Choose a data splitting method.')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--message_steps', default=3, type=int)

    parser.add_argument('--num_clients', default=4, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--null_value', default=-1, type=float)
    parser.add_argument('--seed', default=4567, type=int, choices=[1234, 4567, 7890],
                        help='Initialize random number seeds for model training and data splitting.')
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    parser.add_argument('--eps', type=str, default='mixgauss1',
                        help='epsilon file name')
    parser.add_argument('--constant', type=float, default=2000)
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='differential privacy parameter')
    parser.add_argument('--dp', default=True, type=bool, choices=[True, False],
                        help='if True, use differential privacy')

    parser.add_argument('--batch_size', default=32, type=int,
                        choices=[32, 64, 128])
    parser.add_argument('--device', default='cuda', type=str,
                        choices=['cuda', 'cpu'])
    parser.add_argument('--save_dir', default='results', type=str)

    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--local_round', default=10, type=int)
    parser.add_argument('--proj_dims', default=1, type=int)
    parser.add_argument('--lanczos_iter', default=8, type=int)
    parser.add_argument('--global_round', default=200, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        choices=[0.1, 0.001, 0.0001])
    parser.add_argument('--clip', default=0.5, type=float,
                        choices=[1.0, 1.5, 2.0])
    # parser.add_argument('--anti_noise', default=0, type=float, help='0.1, 0.15, 0.2')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    exec("dataset = {}Dataset('./dataset/{}', '{}', '{}', {})".format(args.root, args.root, args.dataset, args.split, args.seed))
    # dataset = MolDataset(root='../dataset/' + args.root, name=args.dataset, split_seed=args.seed)
    print('the dataset name: {}, the mol size: {}.'.format(args.dataset, len(dataset)))

    args.num_clients = 3 if len(dataset) <= 2000 else 4
    args.node_size, args.bond_size = dataset.node_features, dataset.edge_features
    args.output_size = dataset.num_tasks
    print(args)

    # accountants = []
    architecture = Mol_architecture(args) if args.root in ['MoleculeNet', 'LITPCBA'] else DMol_architecture(args)
    main(args, dataset, architecture)
