#!/usr/bin/env bash
# CIFAR10-LT FairFAL coverage-preserving-interpolation ablation (lambda = 1.0).
# Run from the repository root. These commands intentionally differ from the
# final FairFAL configuration only in --reliability_budget_blend.

set -euo pipefail

python main.py --gpu 0 --seed 2 --custom_name c10_lambda1_seed2 --al_method fairfal --noise_mode hetero --model cnn4conv --dataset cifar10_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 0.1 --num_users 10 --frac 1.0 --num_classes 10 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05 --end_ratio 0.45 --budget_mode estimated_r_soft_blend --budget_min_ratio 0.5 --reliability_ema_beta 0.5 --reliability_budget_blend 1.0

python main.py --gpu 0 --seed 3 --custom_name c10_lambda1_seed3 --al_method fairfal --noise_mode hetero --model cnn4conv --dataset cifar10_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 0.1 --num_users 10 --frac 1.0 --num_classes 10 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05 --end_ratio 0.45 --budget_mode estimated_r_soft_blend --budget_min_ratio 0.5 --reliability_ema_beta 0.5 --reliability_budget_blend 1.0

python main.py --gpu 0 --seed 4 --custom_name c10_lambda1_seed4 --al_method fairfal --noise_mode hetero --model cnn4conv --dataset cifar10_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 0.1 --num_users 10 --frac 1.0 --num_classes 10 --rounds 100 --local_ep 5 --reset random --query_ratio 0.05 --end_ratio 0.45 --budget_mode estimated_r_soft_blend --budget_min_ratio 0.5 --reliability_ema_beta 0.5 --reliability_budget_blend 1.0
