#!/usr/bin/env python3
"""Reproduce the Section 4.4 reliability-calibration diagnostics from run CSVs.

main.py writes, into every ``label_ratio<X>/<custom_name>/`` directory, the
per-client / per-stage CSVs that this script aggregates:

  * reliability_soft_estimation.csv   (Estimated-R soft/MLE mode)
  * budget_allocation.csv             (final per-client budgets)
  * annotation_noise.csv              (per-client corrupted-query counts)
  * reliability_anchor_estimation.csv (only for anchor-calibrated runs)

Metrics reproduced (targets quoted from the paper):
  (a) Spearman rho(updated_reliability, true_reliability_debug)  -> 0.893
  (b) JSD-to-uniform: per-stage JSD between the allocated-budget  -> 0.0875 / 0.0916
      distribution and a uniform reference (1/N per client)
  (c) total corrupted queries  (FedRBA vs uniform baseline)      -> 1280 -> 1224
  (d) anchor MAE(updated_reliability, true_reliability_debug)    -> 0.215 -> 0.133

Spearman/JSD are computed WITHOUT scipy (pure numpy) so the script runs in the
same minimal env as main.py.

Usage:
  python analysis/diagnostics_section4_4.py --run_dir <fairfal_root>
  python analysis/diagnostics_section4_4.py --run_dir <fedrba_root> \
         --compare_dir <uniform_root>
"""

import argparse
import os

import numpy as np
import pandas as pd


# ----------------------------- helpers --------------------------------------
def _rankdata_average(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2:
        return float("nan")
    lm = left - left.mean()
    rm = right - right.mean()
    denom = np.sqrt((lm ** 2).sum() * (rm ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((lm * rm).sum() / denom)


def _spearman(a, b):
    return _pearson(_rankdata_average(a), _rankdata_average(b))


def _jsd(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _load_all(run_dir, name, custom_name=None):
    files = []
    for root, _, fnames in os.walk(run_dir):
        for fn in fnames:
            if fn != name:
                continue
            path = os.path.join(root, fn)
            if custom_name is not None and (
                os.sep + custom_name + os.sep not in path.replace("/", os.sep)
            ):
                continue
            files.append(path)
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _summary_noise_total(noise_df):
    if noise_df is None:
        return None
    s = noise_df.copy()
    s["sample_idx"] = s["sample_idx"].astype(str).str.strip()
    summ = s[(s["sample_idx"] == "") | (s["sample_idx"].str.lower() == "nan")]
    return int(summ["n_corrupt"].sum())


# ----------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        required=True,
        help="FedRBA run root; the directory that contains label_ratio*/<custom_name>/. "
        "In practice this is the .../qmode_global/fairfal/ folder of a seed.",
    )
    ap.add_argument(
        "--compare_dir",
        default=None,
        help="uniform/fixed baseline run root, for delta metrics (corrupted queries, JSD).",
    )
    ap.add_argument(
        "--custom_name",
        default=None,
        help="isolate a single run when several live under the same run_dir "
        "(e.g. c100_soft790_seed2). Matches the label_ratio*/<custom_name>/ path.",
    )
    ap.add_argument(
        "--compare_custom_name",
        default=None,
        help="custom_name filter for --compare_dir (e.g. c100_softfix_seed2).",
    )
    args = ap.parse_args()

    rel = _load_all(args.run_dir, "reliability_soft_estimation.csv", args.custom_name)
    bud = _load_all(args.run_dir, "budget_allocation.csv", args.custom_name)
    noise = _load_all(args.run_dir, "annotation_noise.csv", args.custom_name)
    anchor = _load_all(args.run_dir, "reliability_anchor_estimation.csv", args.custom_name)

    print("=" * 64)
    print(f"Run root: {args.run_dir}")
    print("=" * 64)

    # (a) Spearman rho(updated_reliability, true_reliability_debug)
    if rel is not None and "true_reliability_debug" in rel:
        d = rel.dropna(subset=["true_reliability_debug", "updated_reliability"])
        d = d[d["updated_reliability"].notna()]
        if len(d) >= 2:
            rho = _spearman(
                d["updated_reliability"].values, d["true_reliability_debug"].values
            )
            print(
                f"[a] Spearman rho (pooled, n={len(d)}) = {rho:.3f}"
            )
            # headline: final acquisition stage only
            final_ratio = d["current_ratio"].max()
            df = d[d["current_ratio"] == final_ratio]
            if len(df) >= 2:
                rho_f = _spearman(
                    df["updated_reliability"].values,
                    df["true_reliability_debug"].values,
                )
                print(
                    f"    Spearman rho @ final stage (ratio={final_ratio:.3f}, "
                    f"n={len(df)}) = {rho_f:.3f}"
                )
        else:
            print("[a] Spearman: not enough valid rows")
    else:
        print("[a] Spearman: reliability_soft_estimation.csv not found")

    # (b) JSD-to-uniform (paper Sec. 4.4): per-stage JSD between the
    #     allocated-budget distribution and a uniform reference (1/N per client).
    #     The paper reports this as "JSD-to-uniform changes only from 0.0875
    #     (uniform allocation) to 0.0916 (FedRBA)" -- i.e. the coverage-preserving
    #     constraints keep the allocation close to broad client participation.
    if bud is not None:
        n_clients = int(bud["user_idx"].nunique())
        uniform_ref = np.ones(n_clients) / n_clients
        stages = sorted(bud["current_ratio"].unique())
        jsds = []
        for s in stages:
            b = bud[bud["current_ratio"] == s]
            if len(b) == 0:
                continue
            alloc = b["allocated_budget"].values.astype(float)
            if alloc.sum() > 0:
                jsds.append(_jsd(alloc, uniform_ref))
        if jsds:
            jsds = np.array(jsds)
            print(
                f"[b] JSD-to-uniform (alloc || uniform): mean={jsds.mean():.4f} "
                f"min={jsds.min():.4f} max={jsds.max():.4f} (n_stages={len(jsds)})"
            )
    else:
        print("[b] JSD-to-uniform: budget_allocation.csv not found")

    # (c) corrupted queries
    total = _summary_noise_total(noise)
    if total is not None:
        print(f"[c] total corrupted queries (this run) = {total}")
        if args.compare_dir:
            noise2 = _load_all(args.compare_dir, "annotation_noise.csv", args.compare_custom_name)
            t2 = _summary_noise_total(noise2)
            if t2 is not None:
                print(
                    f"    total corrupted queries (compare) = {t2}  "
                    f"-> delta = {total - t2}"
                )
    else:
        print("[c] corrupted queries: annotation_noise.csv not found")

    # (d) anchor MAE
    if anchor is not None and "true_reliability_debug" in anchor:
        a = anchor.dropna(subset=["true_reliability_debug", "updated_reliability"])
        if len(a) >= 2:
            mae = np.abs(
                a["updated_reliability"].values.astype(float)
                - a["true_reliability_debug"].values.astype(float)
            ).mean()
            print(f"[d] anchor MAE(updated_reliability, true_reliability) = {mae:.3f}")
        else:
            print("[d] anchor MAE: not enough rows")
    else:
        print(
            "[d] anchor MAE: N/A (no reliability_anchor_estimation.csv; "
            "run an anchor-calibrated mode to produce it)"
        )


if __name__ == "__main__":
    main()
