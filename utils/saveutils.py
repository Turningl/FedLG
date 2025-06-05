# -*- coding: utf-8 -*-
# @Author : liang
# @File : save_func.py


import os
import csv
import numpy as np

np.random.seed(42)


def save_progress(args, performance_accountant, optimal_accountant):
    """
    Save the progress of a federated learning experiment to a CSV file.

    Args:
        args (object): Configuration arguments.
        performance_accountant (list): List of performance metrics.
        optimal_accountant (float): Optimal result (e.g., best validation accuracy or loss).

    Returns:
        None
    """
    # Determine the save directory based on whether communication optimization is used.
    if args.comm_optimization:
        save_dir = os.path.join(os.getcwd(), args.save_dir,
                                (args.root if args.root else 'root'),
                                (args.dataset if args.dataset else 'dataset'),
                                'Bayesian Optimization' if args.comm_optimization else 'comm_optimization')
    else:
        save_dir = os.path.join(os.getcwd(), args.save_dir,
                                (args.root if args.root else 'root'),
                                (args.dataset if args.dataset else 'dataset'))

    # Create the save directory if it does not exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Construct the file name based on the experiment configuration.
    file_name = '{}_{}_{}_{}_{}_{}'.format(('federated algorithm-' + args.alg if args.alg else 'alg'),
                                           ('graph model-' + args.model if args.model else 'model'),
                                           ('num clients-' + str(args.num_clients) if args.num_clients else 'num_clients'),
                                           ('seed-' + str(args.seed) if args.seed else 'split_seed'),
                                           ('global round-' + str(args.global_round) if args.global_round else 'global_round'),
                                           ('local round-' + str(args.local_round) if args.local_round else 'local_round'))


    # Write the performance metrics and optimal result to a CSV file.
    with open(os.path.join(save_dir, file_name + '.csv'), 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(performance_accountant)  # Write the performance metrics.
        writer.writerow(['optimal result is:' + str(optimal_accountant)])  # Write the optimal result.


def print_rmse_accoutant(test_results):
    print('-------------------------------------------------------------------------------------')
    print('the min rmse is: %.4f\n' % np.min(test_results))
    print('-------------------------------------------------------------------------------------')
    return np.min(test_results)


def print_accuracy_accoutant(test_results):
    print('-------------------------------------------------------------------------------------')
    print('the max acc is:%.4f\n' % np.max(test_results))
    print('-------------------------------------------------------------------------------------')
    return np.max(test_results)


def print_init_alg_info(res):
    def _print_init_alg_info(func):
        def wrapper(*args, **kwargs):
            print('')
            print('-------------------------------------------------------------------------------------')
            print(res + '!!!')
            print('')
            print('Execute ' + 'Federated learning algorithm. Using the following algorithm:')
            return func(*args, **kwargs)

        return wrapper

    return _print_init_alg_info


def print_average_info(func):
    def wrapper(*args, **kwargs):
        print('')
        print('-------------------------------------------------------------------------------------')
        print('')
        print('Execute ' + 'model state dict loading!')
        return func(*args, **kwargs)

    return wrapper
