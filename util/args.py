#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import argparse
import medmnist
from medmnist import INFO

def args_parser():
    parser = argparse.ArgumentParser()
    
    # basic arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--custom_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_dir', type=str, default='.', help='when to start saving models')    
    
    # federated learning arguments
    parser.add_argument('--rounds', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--test_bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--weight_decay', type=float, default=0.00001, help="weight decay (default: 0.00001)")
    parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay ratio")
    parser.add_argument('--reset', type=str, default='random_init', help='doing FL with queried dataset or not')
    parser.add_argument('--fl_algo', type=str, default='fedavg', help='federated learning algorithm')
    parser.add_argument('--mu', type=float, default=0.01, help='weight of FedProx regularization term')
    
    # dataset arguments
    parser.add_argument('--data_dir', type=str, default='./data', help='data path')
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--partition', type=str, default="dir_balance", help="methods for Non-IID")
    parser.add_argument('--dd_beta', type=float, default=0.1, help="beta for dirichlet distribution")
    parser.add_argument('--num_classes_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--imb_ratio', type=float, default=1.0, help="imbalance ratio for long tail dataset")
    
    # model arguments
    parser.add_argument('--model', type=str, default='cnn4conv', help='model name')
    
    # active learning arguments
    parser.add_argument('--resume_ratio', type=float, default=0., help="ratio of data examples for resume")
    parser.add_argument('--query_ratio', type=float, default=0.05, help="ratio of data examples per one query")
    parser.add_argument('--end_ratio', type=float, default=0.0, help="ratio for stopping query")
    parser.add_argument('--query_model_mode', type=str, default="global")
    parser.add_argument('--al_method', type=str, default=None)
    parser.add_argument(
        '--budget_mode',
        type=str,
        default='fixed',
        choices=[
            'fixed', 'oracle_r', 'oracle_u', 'oracle_ru', 'estimated_r',
            'estimated_r_soft', 'estimated_r_soft_blend',
            'estimated_r_anchor_blend', 'oracle_r_blend',
            'shuffled_r_soft_blend'
        ],
        help='per-client active-query budget allocation mode'
    )
    parser.add_argument(
        '--budget_min_ratio',
        type=float,
        default=0.5,
        help='minimum per-client budget as a fraction of the fixed budget'
    )
    parser.add_argument(
        '--reliability_budget_blend',
        type=float,
        default=0.5,
        help='fixed-to-reliability budget blend strength for reliability blend modes'
    )
    parser.add_argument(
        '--reliability_threshold',
        type=float,
        default=0.7,
        help='minimum global-model confidence used by Estimated-R'
    )
    parser.add_argument(
        '--reliability_ema_beta',
        type=float,
        default=0.8,
        help='EMA coefficient for Estimated-R reliability updates'
    )
    parser.add_argument(
        '--reliability_min_confident',
        type=int,
        default=5,
        help='minimum high-confidence queried samples for an Estimated-R update'
    )
    # queried-label annotation noise
    parser.add_argument(
        '--noise_mode',
        type=str,
        default='clean',
        choices=['clean', 'uniform', 'hetero', 'custom'],
        help='annotation noise mode for newly queried labels'
    )

    parser.add_argument(
        '--custom_noise_rates',
        type=str,
        default=None,
        help='comma-separated per-client noise rates under custom noise'
    )

    parser.add_argument(
        '--uniform_noise_rate',
        type=float,
        default=0.19,
        help='queried-label error rate under uniform noise'
    )
    
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()

    if args.budget_min_ratio < 0:
        raise ValueError('--budget_min_ratio must be non-negative')

    if not 0.0 <= args.reliability_budget_blend <= 1.0:
        raise ValueError('--reliability_budget_blend must be within [0, 1]')

    if not 0.0 <= args.reliability_threshold <= 1.0:
        raise ValueError('--reliability_threshold must be within [0, 1]')

    if not 0.0 <= args.reliability_ema_beta < 1.0:
        raise ValueError('--reliability_ema_beta must be within [0, 1)')

    if args.reliability_min_confident < 1:
        raise ValueError('--reliability_min_confident must be at least 1')

    if args.noise_mode == 'custom':
        if args.custom_noise_rates is None:
            raise ValueError(
                '--custom_noise_rates is required when noise_mode=custom'
            )

        try:
            custom_noise_rates = [
                float(value.strip())
                for value in args.custom_noise_rates.split(',')
            ]
        except ValueError as exc:
            raise ValueError(
                '--custom_noise_rates must be a comma-separated list of numbers'
            ) from exc

        if len(custom_noise_rates) != args.num_users:
            raise ValueError(
                '--custom_noise_rates must contain exactly {} values; got {}'.format(
                    args.num_users,
                    len(custom_noise_rates)
                )
            )

        if any(not 0.0 <= rate <= 1.0 for rate in custom_noise_rates):
            raise ValueError(
                'Each value in --custom_noise_rates must be within [0, 1]'
            )

        args.custom_noise_rates = custom_noise_rates
    
    # popular benchmark
    if args.dataset in ['fmnist', 'fmnist_lt']:
        args.num_classes = 10
        args.in_channels = 1
        args.img_size = 28
        if not args.end_ratio: args.end_ratio = 0.45
    elif args.dataset in ['cifar10', 'cifar10_lt']:
        args.num_classes = 10
        args.in_channels = 3
        args.img_size = 32
        if not args.end_ratio: args.end_ratio = 0.45
    elif args.dataset in ['cifar100', 'cifar100_lt']:
        args.num_classes = 100
        args.in_channels = 3
        args.img_size = 32
        if not args.end_ratio: args.end_ratio = 0.45
    
    # medical benchmark
    elif args.dataset == 'octmnist': # 109,309 samples, 97,477 train
        info = INFO[args.dataset]
        args.num_classes = len(info['label']) # 4
        args.in_channels = info['n_channels'] # 1
        args.img_size = 28
        if not args.end_ratio: args.end_ratio = 0.45
    elif args.dataset == 'dermamnist': # 10,015 samples, 7,007 train
        info = INFO[args.dataset]
        args.num_classes = len(info['label']) # 7
        args.in_channels = info['n_channels'] # 3
        args.img_size = 28
        if not args.end_ratio: args.end_ratio = 0.45
        
    # for init
    if not args.resume_ratio:
        args.current_ratio = args.query_ratio
    else:
        args.current_ratio = args.resume_ratio
        
    if args.dataset in ['cifar10_lt', 'cifar100_lt']:
        args.data_dir += '/{}_{}/'.format(args.dataset, args.imb_ratio)
    else:
        args.data_dir += '/{}/'.format(args.dataset)
        
    return args
