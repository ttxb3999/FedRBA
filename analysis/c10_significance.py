import numpy as np
from scipy import stats

fixed = np.array([
    [38.89,41.84,43.05,42.82,43.22,43.05,43.27,41.90,41.67],
    [39.31,41.03,40.77,40.54,42.39,42.20,41.46,41.99,41.00],
    [39.84,42.77,43.96,42.62,43.62,42.70,41.23,42.39,41.98],
    [37.83,40.19,42.31,43.43,43.63,42.82,43.56,42.73,41.99],
    [39.32,41.00,42.12,43.38,42.88,43.01,43.13,42.65,42.51],
])

fedral = np.array([
    [38.89,41.84,43.37,42.28,42.51,43.46,43.25,42.42,42.32],
    [39.31,41.03,42.05,41.21,42.69,41.28,41.73,41.29,41.16],
    [39.84,42.77,46.20,42.76,43.95,44.06,43.28,43.17,43.35],
    [37.83,40.19,44.15,42.80,42.32,43.11,43.11,43.03,42.53],
    [39.32,41.00,43.86,43.23,43.60,43.45,42.91,43.06,43.98],
])

def report(name, fixed_values, fedral_values):
    diff = fedral_values - fixed_values

    print(f"\n===== {name} =====")
    print("Fixed per seed :", fixed_values)
    print("FedRAL per seed:", fedral_values)
    print("Diff per seed  :", diff)

    print()
    print(
        "Fixed  = {:.4f} ± {:.4f}".format(
            fixed_values.mean(),
            fixed_values.std(ddof=1)
        )
    )
    print(
        "FedRAL = {:.4f} ± {:.4f}".format(
            fedral_values.mean(),
            fedral_values.std(ddof=1)
        )
    )
    print("Mean gain = {:.4f} pp".format(diff.mean()))

    t = stats.ttest_rel(fedral_values, fixed_values)
    w = stats.wilcoxon(fedral_values, fixed_values)

    print()
    print("Paired t-test:")
    print("  t =", t.statistic)
    print("  p =", t.pvalue)

    print("Wilcoxon signed-rank:")
    print("  statistic =", w.statistic)
    print("  p =", w.pvalue)


fixed_final = fixed[:, -1]
fedral_final = fedral[:, -1]

fixed_avg = fixed[:, 1:].mean(axis=1)
fedral_avg = fedral[:, 1:].mean(axis=1)

report("Final@45", fixed_final, fedral_final)
report("Avg@10-45", fixed_avg, fedral_avg)
