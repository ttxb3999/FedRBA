#!/usr/bin/env bash
# Table 2 ablations of the FedRBA allocator (seeds 2-6), both CIFAR10-LT and
# CIFAR100-LT. Run from the repository root.
#
# Three allocator variants are compared against the full FedRBA (beta = gamma =
# lambda = 0.5, reproduced by scripts/run_cifar{10,100}_lt_table1.sh):
#
#   * w/o EMA       : --reliability_ema_beta 0      (Eq.5 memory disabled;
#                                                     updated reliability = this
#                                                     cycle's point estimate)
#   * w/o Floor     : --budget_min_ratio 0          (Eq.6 floor gamma disabled;
#                                                     pure reliability-weighted split)
#   * lambda = 1    : --reliability_budget_blend 1.0(blend collapses to the raw
#                                                     reliability budget, no fixed
#                                                     uniform share)
#
# All variants share the FairFAL selector, FedAvg backend, partitions and
# queried-label noise with the main table; only the allocator knobs differ.
#
# Training schedule: 300 FedAvg rounds per stage, lr x0.1 after round 225
# (rounds*3/4), matching Section 4.1. The effective per-cycle budget is exactly
# matched across datasets (B = 850 for CIFAR10-LT, B = 790 for CIFAR100-LT) by
# the allocator's per-client floor.
#
# Override the interpreter if needed:
#   PYTHON_BIN=/path/to/python bash scripts/run_table2_ablations.sh

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LOGDIR="${LOGDIR:-logs_table2_r300}"
mkdir -p "$LOGDIR"

for DS in cifar10_lt cifar100_lt; do
  NC=$( [ "$DS" = "cifar10_lt" ] && echo 10 || echo 100 )
  for SEED in 2 3 4 5 6; do
    COMMON_ARGS="--gpu 0 --al_method fairfal --noise_mode hetero --model cnn4conv \
      --dataset $DS --imb_ratio 0.05 --partition dir_balance --dd_beta 0.1 \
      --num_users 10 --frac 1.0 --num_classes $NC --rounds 300 --local_ep 5 \
      --reset random --query_ratio 0.05 --end_ratio 0.45"

    # w/o EMA  (beta = 0)
    "$PYTHON_BIN" main.py --seed $SEED \
      --custom_name ${DS%_lt}_noema_seed$SEED \
      $COMMON_ARGS \
      --budget_mode estimated_r_soft_blend --budget_min_ratio 0.5 \
      --reliability_ema_beta 0 --reliability_budget_blend 0.5 \
      2>&1 | tee "$LOGDIR/${DS%_lt}_noema_seed${SEED}.log"

    # w/o Floor  (gamma = 0)
    "$PYTHON_BIN" main.py --seed $SEED \
      --custom_name ${DS%_lt}_nofloor_seed$SEED \
      $COMMON_ARGS \
      --budget_mode estimated_r_soft_blend --budget_min_ratio 0 \
      --reliability_ema_beta 0.5 --reliability_budget_blend 0.5 \
      2>&1 | tee "$LOGDIR/${DS%_lt}_nofloor_seed${SEED}.log"

    # lambda = 1  (blend collapses to raw reliability budget)
    "$PYTHON_BIN" main.py --seed $SEED \
      --custom_name ${DS%_lt}_lambda1_seed$SEED \
      $COMMON_ARGS \
      --budget_mode estimated_r_soft_blend --budget_min_ratio 0.5 \
      --reliability_ema_beta 0.5 --reliability_budget_blend 1.0 \
      2>&1 | tee "$LOGDIR/${DS%_lt}_lambda1_seed${SEED}.log"
  done
done
