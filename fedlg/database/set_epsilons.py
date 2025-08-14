# -*- coding: utf-8 -*-
# @Author : liang
# @File : set_epsilons.py


import configparser
import numpy as np


def set_epsilons_from_distributions(args, N):
    """
    Args:
        args (object): Configuration arguments.
        N (int): Total number of clients.

    Returns:
        numpy.ndarray: Array of epsilon values for each database.
    """
    np.random.seed(args.seed + 1)  # Set the random seed for reproducibility.
    public_num, private_num = 1, N - 1  # Define the number of public and private clients.
    print('Set epsilons')

    # Read the configuration file for epsilon distributions.
    config = configparser.RawConfigParser()
    config.read('./fedlg/param/{}.in'.format(args.eps))

    # Parse the distributions from the configuration file.
    dists = []
    for section in config.sections()[:2]:  # Consider only the first two sections.
        mean = float(config.get(section, 'mean'))  # Get the mean of the distribution.
        std = float(config.get(section, 'std'))  # Get the standard deviation of the distribution.
        dist = {'mean': mean, 'std': std}  # Store the distribution param.
        dists.append(dist)

    # Generate samples from the distributions.
    samples1 = np.random.normal(
        loc=dists[0]['mean'],  # Generate samples for private clients.
        scale=dists[0]['std'],
        size=private_num).tolist()

    samples2 = np.random.normal(
        loc=dists[1]['mean'],  # Generate samples for public clients.
        scale=dists[1]['std'],
        size=public_num).tolist()

    epsilons = np.concatenate([samples1, samples2])  # Combine the samples into a single array.

    # Optionally parse 'prob' and 'threshold' sections if needed.
    # prob = [float(p) for p in config.get('prob', 'a').splitlines() if p.strip()]
    # threshold = [float(t) for t in config.get('threshold', 'threshold').splitlines() if t.strip()]

    # print("Probabilities:", prob)
    # print("Thresholds:", threshold)

    # np.random.shuffle(epsilons)  # Shuffle the epsilon values to ensure randomness.
    return epsilons


def prepare_local_differential_privacy(args, num_clients):
    """
    Prepare epsilon values for local differential privacy.

    Args:
        args (object): Configuration arguments.
        num_clients (int): Total number of clients.

    Returns:
        numpy.ndarray or None: Array of epsilon values for each database if differential privacy is enabled, otherwise None.
    """
    epsilons = None  # Initialize epsilon values to None.
    if args.dp:  # Check if differential privacy is enabled.
        epsilons = set_epsilons_from_distributions(args, num_clients)  # Generate epsilon values.
    return epsilons  # Return the epsilon values or None.


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dp', default=True)
#     parser.add_argument('--seed', default=12)
#     parser.add_argument('--eps', default='mixgauss1')
#
#     args = parser.parse_args()
#
#     prepare_local_differential_privacy(args=args, num_clients=4)
