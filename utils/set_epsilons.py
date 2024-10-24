# -*- coding: utf-8 -*-
# @Author : liang
# @File : set_epsilons.py

import configparser
import numpy as np

def set_epsilons_from_distributions(args, N):
    np.random.seed(args.seed + 1)
    public_num, private_num = 1, N - 1
    print('Set epsilons')

    config = configparser.RawConfigParser()
    config.read('../epsilons/{}.in'.format(args.eps))

    # Parse distributions
    dists = []
    for section in config.sections()[:2]:
        mean = float(config.get(section, 'mean'))
        std = float(config.get(section, 'std'))
        dist = {'mean': mean, 'std': std}
        dists.append(dist)

    # Generate samples from the distributions
    samples1 = np.random.normal(loc=dists[0]['mean'],
                                scale=dists[0]['std'],
                                size=private_num).tolist()

    samples2 = np.random.normal(loc=dists[1]['mean'],
                                scale=dists[1]['std'],
                                size=public_num).tolist()

    epsilons = np.concatenate([samples1, samples2])

    # Optionally parse 'prob' and 'threshold' sections if needed
    # prob = [float(p) for p in config.get('prob', 'a').splitlines() if p.strip()]
    #threshold = [float(t) for t in config.get('threshold', 'threshold').splitlines() if t.strip()]

    # print("Probabilities:", prob)
    # print("Thresholds:", threshold)

    # np.random.shuffle(epsilons)
    return epsilons

def prepare_local_differential_privacy(args, num_clients):
    epsilons = None
    if args.dp:
        epsilons = set_epsilons_from_distributions(args, num_clients)
    return epsilons


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
