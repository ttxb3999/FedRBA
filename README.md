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
scripts/                 # reproduction scripts: run_cifar{10,100}_lt_table1.sh (Table 1),
                        #   run_table2_ablations.sh (Table 2), run_cifar10_lt_lambda1.sh (λ=1),
                        #   run_seed34_controls.ps1 + prepare_seed2_control_caches.ps1 (controls),
                        #   run_sota_multi_gpu.py (multi-GPU baselines)
analysis/                # diagnostics_section4_4.py (Section 4.4) + c100_final_analysis.py (Table 1)
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

### Table 1 — primary controlled allocator comparison (five seeds, 300 rounds/stage)

One-shot scripts cover both arms for both datasets:

```bash
bash scripts/run_cifar10_lt_table1.sh   # CIFAR10-LT: FedRBA c10_soft850_seedN + uniform c10_softfix_seedN, seeds 2-6
bash scripts/run_cifar100_lt_table1.sh  # CIFAR100-LT: FedRBA c100_soft790_seedN + uniform c100_softfix_seedN, seeds 2-6
```

Or run a single arm by hand (FedRBA):

```bash
python main.py --gpu 0 --seed 2 --custom_name c10_soft850_seed2 \
  --al_method fairfal --noise_mode hetero --model cnn4conv \
  --dataset cifar10_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 0.1 \
  --num_users 10 --frac 1.0 --num_classes 10 --rounds 300 --local_ep 5 \
  --reset random --query_ratio 0.05 --end_ratio 0.45 \
  --budget_mode estimated_r_soft_blend --budget_min_ratio 0.5 \
  --reliability_ema_beta 0.5 --reliability_budget_blend 0.5
```

Uniform allocation (identical except the client budgets):

```bash
python main.py --gpu 0 --seed 2 --custom_name c10_softfix_seed2 \
  --al_method fairfal --noise_mode hetero --model cnn4conv \
  --dataset cifar10_lt --imb_ratio 0.05 --partition dir_balance --dd_beta 0.1 \
  --num_users 10 --frac 1.0 --num_classes 10 --rounds 300 --local_ep 5 \
  --reset random --query_ratio 0.05 --end_ratio 0.45 \
  --budget_mode fixed
```

The raw nominal 5% count is 850 on CIFAR10-LT (79 per client × 10 = 850) and 795 on CIFAR100-LT (79 per client × 10 = 790); the allocator floors to 79 per client automatically, so every method uses the exactly matched effective budget with no extra flag.

### Table 2 — component ablation (seeds 2–6, 300 rounds/stage)

| Variant      | Change relative to FedRBA                          | Script |
|--------------|-----------------------------------------------------|--------|
| w/o EMA      | `--reliability_ema_beta 0.0`                        | `scripts/run_table2_ablations.sh` |
| w/o Floor    | `--budget_min_ratio 0.0`                            | `scripts/run_table2_ablations.sh` |
| w/o Blend    | `--reliability_budget_blend 1.0`                    | `scripts/run_table2_ablations.sh` (also `scripts/run_cifar10_lt_lambda1.sh`) |

All three variants are launched for both CIFAR10-LT and CIFAR100-LT, seeds 2–6, by `scripts/run_table2_ablations.sh`. Custom names: `<ds>_noema_seedN`, `<ds>_nofloor_seedN`, `<ds>_lambda1_seedN` (e.g. `cifar10_noema_seed2`).

### Reliability controls (Section 4.4)

- Oracle-R: `--budget_mode oracle_r_blend`
- Shuffled-R: `--budget_mode shuffled_r_soft_blend`
- Clean-anchor probe: `--budget_mode estimated_r_anchor_blend`
- Control-run scripts: `scripts/run_seed34_controls.ps1`, `scripts/prepare_seed2_control_caches.ps1`

**Reproducing the Section 4.4 calibration metrics** (Spearman ρ, JSD, corrupted-query count, anchor MAE): `main.py` writes per-client / per-stage CSVs (`reliability_soft_estimation.csv`, `budget_allocation.csv`, `annotation_noise.csv`, `reliability_anchor_estimation.csv`) into each `label_ratio*/<custom_name>/` directory. Aggregate them with:

```bash
python analysis/diagnostics_section4_4.py \
  --run_dir <path/to/.../qmode_global/fairfal> \
  --custom_name c100_soft790_seed2 \
  --compare_dir <same/fairfal/path> --compare_custom_name c100_softfix_seed2
```

The script implements Spearman/JSD in pure numpy (no scipy) and reads the `*_debug` ground-truth columns the training loop logs. Targets quoted in the paper: ρ ≈ 0.893, JSD ≈ 0.0875 → 0.0916, corrupted queries 1280 → 1224, anchor MAE 0.215 → 0.133.

### Final-result analysis (Table 1)

`analysis/c100_final_analysis.py` recomputes mean±std curves and paired significance tests (t-test / Wilcoxon) from the run accuracies. It currently hard-codes the five-seed accuracy matrix; point it at your `save/` tree (or paste the numbers) to regenerate `cifar100_curve.png/.pdf` and the summary CSVs.

### Matched FAL baselines (Figure 2)

- FairFAL selector with uniform budgets: `--al_method fairfal --budget_mode fixed`
- KAFAL-KSAS selector with uniform budgets: `--al_method kafal --budget_mode fixed`
- Multi-GPU launcher: `scripts/run_sota_multi_gpu.py`

## Notes

- Reliability evidence from cycle *t* is used only for cycle *t+1* budgets; no queried label affects the budget that caused it to be queried (see Algorithm 1 in the paper).
- Integer budgets are obtained by flooring plus largest-remainder assignment, preserving both the exact global budget and the per-client floor.
- **Reproducibility / repository hygiene:** every run writes large `save/` trees and `*.log` files; these are git-ignored (`save/`, `logs/`, `logs_*/`, `*.log`, `*.pkl`, `data/`). Only source, scripts, analysis, `requirements.txt`, and this README are tracked. Re-run any `scripts/run_*.sh` (override the interpreter with `PYTHON_BIN=...`) to rebuild results locally.
- `requirements.txt` lists the minimal runtime dependencies (validated on the authors' RTX 5060 / RTX 3090, Python 3.11). Install the PyTorch wheel matching your CUDA version first; `environment_server.txt` is the full `pip freeze` of the paper's GPU server.
- The allocator's flat-likelihood guard (skips a client's EMA update when its softmax log-likelihood range ≤ 1e-6) and the per-client capacity cap on allocated budgets are implemented but not described in the paper; they only affect numerical edge cases and never change the reported numbers.
