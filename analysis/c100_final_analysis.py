import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ratios = np.array([0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45])

fixed = {
    2: [10.62,14.16,15.47,16.18,18.05,17.97,18.93,18.57,19.14],
    3: [11.34,13.90,15.05,16.07,18.13,17.68,18.17,17.89,19.02],
    4: [11.18,13.51,14.12,15.49,17.75,17.45,17.77,17.50,18.95],
    5: [11.09,13.03,14.53,15.60,17.21,17.53,17.25,17.54,19.00],
    6: [11.32,14.03,15.20,17.00,18.16,18.01,18.30,18.55,19.05],
}

soft = {
    2: [10.62,14.16,15.40,17.43,17.90,18.21,18.57,18.97,19.31],
    3: [11.34,13.90,15.85,18.29,16.67,17.54,18.32,17.95,18.45],
    4: [11.18,13.51,15.00,16.33,17.60,17.97,18.25,18.80,18.34],
    5: [11.09,13.03,14.64,17.08,17.52,17.55,18.24,17.82,17.58],
    6: [11.32,14.03,15.44,18.75,17.18,17.53,19.33,20.12,19.31],
}

seeds = sorted(fixed)

F = np.array([fixed[s] for s in seeds])
S = np.array([soft[s] for s in seeds])

# -----------------------------
# 1. Long-form raw results
# -----------------------------
rows = []

for method, data in [("Fixed", F), ("FedRAL-Soft", S)]:
    for i, seed in enumerate(seeds):
        for j, ratio in enumerate(ratios):
            rows.append({
                "method": method,
                "seed": seed,
                "ratio": ratio,
                "accuracy": data[i, j],
            })

raw_df = pd.DataFrame(rows)
raw_df.to_csv("cifar100_raw_results.csv", index=False)

# -----------------------------
# 2. Mean/std curve
# -----------------------------
curve_rows = []

for j, ratio in enumerate(ratios):
    f = F[:, j]
    s = S[:, j]

    curve_rows.append({
        "ratio": ratio,
        "fixed_mean": f.mean(),
        "fixed_std": f.std(ddof=1),
        "fedral_mean": s.mean(),
        "fedral_std": s.std(ddof=1),
        "fedral_minus_fixed": (s-f).mean(),
    })

curve_df = pd.DataFrame(curve_rows)
curve_df.to_csv("cifar100_curve_summary.csv", index=False)

# -----------------------------
# 3. Final@45 and Avg@10-45
# -----------------------------
fixed_final = F[:, -1]
soft_final = S[:, -1]

fixed_avg = F[:, 1:].mean(axis=1)
soft_avg = S[:, 1:].mean(axis=1)

seed_df = pd.DataFrame({
    "seed": seeds,
    "fixed_final45": fixed_final,
    "fedral_final45": soft_final,
    "final_diff": soft_final-fixed_final,
    "fixed_avg10_45": fixed_avg,
    "fedral_avg10_45": soft_avg,
    "avg_diff": soft_avg-fixed_avg,
})

seed_df.to_csv("cifar100_seed_metrics.csv", index=False)

summary = pd.DataFrame([
    {
        "metric": "Final@45",
        "fixed_mean": fixed_final.mean(),
        "fixed_std": fixed_final.std(ddof=1),
        "fedral_mean": soft_final.mean(),
        "fedral_std": soft_final.std(ddof=1),
        "mean_difference": (soft_final-fixed_final).mean(),
        "paired_t_p": stats.ttest_rel(soft_final, fixed_final).pvalue,
        "wilcoxon_p": stats.wilcoxon(soft_final, fixed_final).pvalue,
    },
    {
        "metric": "Avg@10-45",
        "fixed_mean": fixed_avg.mean(),
        "fixed_std": fixed_avg.std(ddof=1),
        "fedral_mean": soft_avg.mean(),
        "fedral_std": soft_avg.std(ddof=1),
        "mean_difference": (soft_avg-fixed_avg).mean(),
        "paired_t_p": stats.ttest_rel(soft_avg, fixed_avg).pvalue,
        "wilcoxon_p": stats.wilcoxon(soft_avg, fixed_avg).pvalue,
    }
])

summary.to_csv("cifar100_final_summary.csv", index=False)

# -----------------------------
# 4. Plot
# -----------------------------
fixed_mean = F.mean(axis=0)
fixed_std = F.std(axis=0, ddof=1)
soft_mean = S.mean(axis=0)
soft_std = S.std(axis=0, ddof=1)

plt.figure(figsize=(7,5))

plt.errorbar(
    ratios * 100,
    fixed_mean,
    yerr=fixed_std,
    marker="o",
    capsize=3,
    label="Fixed"
)

plt.errorbar(
    ratios * 100,
    soft_mean,
    yerr=soft_std,
    marker="s",
    capsize=3,
    label="FedRAL-Soft"
)

plt.xlabel("Labeled Data Ratio (%)")
plt.ylabel("Test Accuracy (%)")
plt.title("CIFAR100-LT")
plt.legend()
plt.tight_layout()

plt.savefig("cifar100_curve.png", dpi=300)
plt.savefig("cifar100_curve.pdf")

# -----------------------------
# 5. Print human-readable summary
# -----------------------------
print("\n===== PER-SEED METRICS =====")
print(seed_df.to_string(index=False))

print("\n===== FINAL SUMMARY =====")
print(summary.to_string(index=False))

print("\n===== MEAN CURVE =====")
print(curve_df.to_string(index=False))

print("\nFiles generated:")
for f in [
    "cifar100_raw_results.csv",
    "cifar100_curve_summary.csv",
    "cifar100_seed_metrics.csv",
    "cifar100_final_summary.csv",
    "cifar100_curve.png",
    "cifar100_curve.pdf",
]:
    print(" -", f)
