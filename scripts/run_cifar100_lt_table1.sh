#!/usr/bin/env bash
# CIFAR100-LT Table 1 primary controlled allocator comparison (seeds 2-6).
# Run from the repository root.
#
# FedRBA (c100_soft790_seedN) and uniform (c100_softfix_seedN) share the same
# FairFAL within-client selector, FedAvg backend, data partitions, and
# queried-label noise; they differ ONLY in client-level budgets.
#
# Budget note: the raw nominal 5% query count on CIFAR100-LT is 795, but equal
# client-wise integer allocation yields 79 queries per client, so every method
# uses the exactly matched effective budget B = 790 (10 x 79). No extra flag is
# needed: the allocator floors to 79 per client automatically.
#
# Training schedule: 300 FedAvg rounds per acquisition stage, lr x0.1 after
# round 225 (implemented as rounds*3/4), matching Section 4.1 of the paper.
#
# Override the interpreter if needed: PYTHON_BIN=/path/to/python bash scripts/run_cifar100_lt_table1.sh

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LOGDIR="${LOGDIR:-logs_c100_r300}"
mkdir -p "$LOGDIR"

COMMON_ARGS="--gpu 0 --al_method fairfal --noise_mode hetero --model cnn4conv \
  --dataset cifar100_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 0.1 \
  --num_users 10 --frac 1.0 --num_classes 100 --rounds 300 --local_ep 5 \
  --reset random --query_ratio 0.05 --end_ratio 0.45"

for SEED in 2 3 4 5 6; do
  # FedRBA: reliability-aware budgets (beta = gamma = lambda = 0.5)
  "$PYTHON_BIN" main.py --seed $SEED --custom_name c100_soft790_seed$SEED \
    $COMMON_ARGS \
    --budget_mode estimated_r_soft_blend --budget_min_ratio 0.5 \
    --reliability_ema_beta 0.5 --reliability_budget_blend 0.5 \
    2>&1 | tee "$LOGDIR/c100_soft790_seed${SEED}.log"

  # Uniform allocation (identical except the client budgets)
  "$PYTHON_BIN" main.py --seed $SEED --custom_name c100_softfix_seed$SEED \
    $COMMON_ARGS \
    --budget_mode fixed \
    2>&1 | tee "$LOGDIR/c100_softfix_seed${SEED}.log"
done
