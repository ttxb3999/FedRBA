# FedRBA

Official code for **"FedRBA: A Coverage-Preserving Budget Allocation Method via Online Client Reliability Estimation for Federated Active Learning"**.

FedRBA treats client-level annotation budgeting as an explicit decision at acquisition time in federated active learning (FAL). A closed reliability feedback loop estimates soft client reliability from pre-update model probabilities and returned annotations, then maps the prior reliability state to coverage-preserving client budgets via reliability weighting, a minimum floor (γ), and uniform blending (λ). FedRBA decides **how many** samples each client may query; the underlying FAL selector (FairFAL / KAFAL-KSAS) decides **which** samples are selected.

## Repository layout

```
main.py                  # entry point: federated active-learning loop with budget allocation
fl_methods/              # federated backends: FedAvg / FedProx / SCAFFOLD
models/                  # cnn4conv (used in the paper), resnet, mobilenet
query_strategies/        # within-client selectors (FairFAL, KAFAL-KSAS) + client budget allocator
util/                    # args, non-IID data simulator, long-tailed datasets, misc
scripts/                 # reproduction scripts (main runs, controls, multi-GPU)
analysis/                # significance tests and final-result analysis
requirements.txt         # minimal python dependencies
environment_server.txt   # full pip freeze of the GPU server used for the paper runs
```

## Setup

```bash
pip install -r requirements.txt
```

Datasets (CIFAR-10/100) are downloaded automatically into `./data/` on first run.

## Reproducing the paper

All paper runs use: CIFAR10-LT / CIFAR100-LT, imbalance ratio 0.05, K = 10 Dirichlet-balanced non-IID clients (α = 0.1), cnn4conv, 300 FedAvg rounds per acquisition stage, 5 local epochs, SGD lr 0.01 / momentum 0.9 / wd 1e-5 / batch 64, lr ×0.1 after round 225. The initial 5% labeled set is clean; only newly queried labels are corrupted with client error rates `[0, 0, 0, 0.1, 0.1, 0.1, 0.3, 0.3, 0.5, 0.5]` (`--noise_mode hetero`). FedRBA hyper-parameters are β = γ = λ = 0.5 for both datasets (no dataset-specific tuning).

### Table 1 — primary controlled allocator comparison (five seeds)

FedRBA (FairFAL selector + reliability-aware budgets):

```bash
python main.py --gpu 0 --seed 2 --custom_name c10_fedrba_seed2 \
  --al_method fairfal --noise_mode hetero --model cnn4conv \
  --dataset cifar10_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 0.1 \
  --num_users 10 --frac 1.0 --num_classes 10 --rounds 100 --local_ep 5 \
  --reset random --query_ratio 0.05 --end_ratio 0.45 \
  --budget_mode estimated_r_soft_blend --budget_min_ratio 0.5 \
  --reliability_ema_beta 0.5 --reliability_budget_blend 0.5
```

Uniform allocation (identical except the client budgets):

```bash
python main.py --gpu 0 --seed 2 --custom_name c10_uniform_seed2 \
  --al_method fairfal --noise_mode hetero --model cnn4conv \
  --dataset cifar10_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 0.1 \
  --num_users 10 --frac 1.0 --num_classes 10 --rounds 100 --local_ep 5 \
  --reset random --query_ratio 0.05 --end_ratio 0.45 \
  --budget_mode fixed
```

Repeat with `--seed 2 3 4 5 6`. For CIFAR100-LT use `scripts/run_cifar100_lt_table1.sh`, which runs both arms (FedRBA as `c100_soft790_seedN`, uniform as `c100_softfix_seedN`, matching the original run names) for seeds 2–6. The raw nominal 5% count is 795, but equal client-wise integer allocation yields 79 per client, so every method uses the exactly matched effective budget B = 790; no extra flag is needed.

### Table 2 — component ablation (seeds 2–4)

| Variant      | Change relative to FedRBA                          |
|--------------|-----------------------------------------------------|
| w/o EMA      | `--reliability_ema_beta 0.0`                        |
| w/o Floor    | `--budget_min_ratio 0.0`                            |
| w/o Blend    | `--reliability_budget_blend 1.0` (see `scripts/run_cifar10_lt_lambda1.sh`) |

### Reliability controls (Section 4.4)

- Oracle-R: `--budget_mode oracle_r_blend`
- Shuffled-R: `--budget_mode shuffled_r_soft_blend`
- Clean-anchor probe: `--budget_mode estimated_r_anchor_blend`
- Control-run scripts: `scripts/run_seed34_controls.ps1`, `scripts/prepare_seed2_control_caches.ps1`

### Matched FAL baselines (Figure 2)

- FairFAL selector with uniform budgets: `--al_method fairfal --budget_mode fixed`
- KAFAL-KSAS selector with uniform budgets: `--al_method kafal --budget_mode fixed`
- Multi-GPU launcher: `scripts/run_sota_multi_gpu.py`

## Notes

- Reliability evidence from cycle *t* is used only for cycle *t+1* budgets; no queried label affects the budget that caused it to be queried (see Algorithm 1 in the paper).
- Integer budgets are obtained by flooring plus largest-remainder assignment, preserving both the exact global budget and the per-client floor.
