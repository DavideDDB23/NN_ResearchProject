# Phase A1 of the plan v3 (14 November 2026), notebook entry X.04.
# Question: M8b is above eps on the eval rows in 10.7% to 12.1% of the cases. Is this a problem of M8b, or the noise
# of the 3,000 eval rows?
# For M2 the answer came from the 7,000 rows outside the calibration (scripts/6_block3_review/m2_validity.py). For
# M8b this is not possible: the 4,000 adapt rows fit the heads (3,000) and choose L2 and the score (1,000), so the
# only clean rows are the 3,000 eval rows, and the result files keep only the risk on them. So here M8b is compared
# with M2 on the same cases (network, corruption, severity, split 0-2, eps 0.05, same score): both risks are measured
# on the same eval rows, so the noise of the eval sample is shared.
# It reads outputs/savings_under_shift_v2/runs/*.csv (it also works while the M8b job is not finished: only the
# networks with M8b rows are used) and writes only outputs/fable_review/fase_a/m8b_violations.csv.
# Usage (from the repository root): python outputs/fable_review/fase_a/m8b_violations.py

import glob

import numpy as np
import pandas as pd

EPS, N_EVAL = 0.05, 3000
CASE = ["dataset", "regime", "seed", "corruption", "severity", "split"]
pd.set_option("display.width", 220)


def two_way_bootstrap(cell8, cell2, reps=2000):
    # cell8, cell2: violation rate of M8b and M2 for every (network, corruption), as networks x corruptions arrays.
    # Networks and corruptions are drawn again with replacement, independently (the cases of one network, or of one
    # corruption, are not independent). Returns the 95% intervals of the rate of M8b, of M2 and of the difference.
    rng = np.random.default_rng(0)
    n, c = cell8.shape
    out = np.empty((reps, 3))
    for r in range(reps):
        rows = np.ix_(rng.integers(n, size=n), rng.integers(c, size=c))
        out[r] = cell8[rows].mean(), cell2[rows].mean(), cell8[rows].mean() - cell2[rows].mean()
    return np.percentile(out, [2.5, 97.5], axis=0).T


files = sorted(glob.glob("outputs/savings_under_shift_v2/runs/*.csv"))
df = pd.concat([pd.read_csv(f, dtype={"l2": str}) for f in files], ignore_index=True)
df = df[np.isclose(df.eps, EPS) & df.split.isin([0, 1, 2]) & df.method.isin(["M2", "M8b"])]
se = np.sqrt(EPS * (1 - EPS) / N_EVAL)  # same definition as scripts/6_block3_review/intervals.py
print(f"eps {EPS}, standard error of a risk of eps on {N_EVAL} rows {se:.4f}, eps + 2 se {EPS + 2 * se:.4f}")

rows = []
for score in ["max_softmax", "agreement"]:
    risk = df[df.score == score].pivot_table(index=CASE, columns="method", values="risk").dropna().reset_index()
    risk["network"] = risk.dataset + "_" + risk.regime + "_seed" + risk.seed.astype(str)
    print(f"\n=== score {score}: {risk.network.nunique()} networks with M8b rows, {len(risk)} paired cases")
    for name, group in [(f"seed {s}", risk[risk.seed == s]) for s in sorted(risk.seed.unique())] + [("all seeds", risk)]:
        above8, above2 = group.M8b > EPS, group.M2 > EPS
        cell8 = (group.M8b > EPS).groupby([group.network, group.corruption]).mean().unstack().to_numpy()
        cell2 = (group.M2 > EPS).groupby([group.network, group.corruption]).mean().unstack().to_numpy()
        (lo8, hi8), (lo2, hi2), (lod, hid) = two_way_bootstrap(cell8, cell2)
        row = {"score": score, "group": name, "networks": group.network.nunique(), "cases": len(group),
               "above_M8b": above8.mean(), "above_M8b_low": lo8, "above_M8b_high": hi8,
               "above_M2": above2.mean(), "above_M2_low": lo2, "above_M2_high": hi2,
               "difference": above8.mean() - above2.mean(), "difference_low": lod, "difference_high": hid,
               "only_M8b_above": (above8 & ~above2).mean(), "only_M2_above": (above2 & ~above8).mean(),
               "above_2se_M8b": (group.M8b > EPS + 2 * se).mean(), "above_2se_M2": (group.M2 > EPS + 2 * se).mean(),
               "mean_risk_M8b": group.M8b.mean(), "mean_risk_M2": group.M2.mean(),
               "worst_risk_M8b": group.M8b.max(), "worst_risk_M2": group.M2.max()}
        rows.append(row)
        print(f"   {name}: above eps M8b {row['above_M8b']:.3f} [{lo8:.3f}, {hi8:.3f}], M2 {row['above_M2']:.3f} "
              f"[{lo2:.3f}, {hi2:.3f}], difference {row['difference']:+.3f} [{lod:+.3f}, {hid:+.3f}]; "
              f"above eps + 2 se: M8b {row['above_2se_M8b']:.3f}, M2 {row['above_2se_M2']:.3f}; "
              f"mean risk M8b {row['mean_risk_M8b']:.4f}, M2 {row['mean_risk_M2']:.4f}; "
              f"worst risk M8b {row['worst_risk_M8b']:.3f}, M2 {row['worst_risk_M2']:.3f}")
    by = risk.assign(a8=risk.M8b > EPS, a2=risk.M2 > EPS)
    print("   by dataset and severity, above eps (M8b, M2):")
    print(by.groupby(["dataset", "severity"])[["a8", "a2"]].mean().unstack("severity").round(3).to_string())

table = pd.DataFrame(rows)
table.to_csv("outputs/fable_review/fase_a/m8b_violations.csv", index=False)

# the predictions of notebook entry X.04, written before the job ended
ms = table[(table.score == "max_softmax") & table.group.str.startswith("seed")]
print("\n=== predictions of X.04 (max softmax, by seed)")
print("difference M8b - M2 below 0.03 on every seed:", bool((ms.difference < 0.03).all()),
      ms.set_index("group").difference.round(3).to_dict())
print("cases above eps + 2 se at most 2% on every seed (M8b):", bool((ms.above_2se_M8b <= 0.02).all()),
      ms.set_index("group").above_2se_M8b.round(3).to_dict())
last = ms[ms.group == "seed 2"]
if len(last):
    print("95% interval of the rate of seed 2 contains 0.10:",
          bool((last.above_M8b_low.iloc[0] <= 0.10) and (0.10 <= last.above_M8b_high.iloc[0])))
print("alarm (difference >= 0.03 on a seed, or more than 3% of the cases above eps + 2 se):",
      bool((ms.difference >= 0.03).any() or (ms.above_2se_M8b > 0.03).any()))
