# Phase A1 of the plan v3 (14 November 2026). Checks on outputs/savings_under_shift_v2/results.csv, the file the M8b
# job writes when it ends (M8b on seeds 1 and 2, finding R05). Nothing is written: it only prints.
#   1. rows: the 230,040 rows of outputs/savings_under_shift/results.csv plus 16 x 450 new M8b rows;
#   2. every row of the original file is in the new one with the same values;
#   3. the new rows: 450 per network, no missing values, lambda in (0, 1], L2 strengths from the grid;
#   4. the gain of M8b over M2 by network, computed here in another way (from results.csv with a join, not from
#      runs/*.csv with a pivot) against outputs/block3_review/m8b_three_seeds_week4.csv, and that file against
#      outputs/block3_review/figures_v2/adaptation_test.csv (written by savings_figures.py).
# Usage (from the repository root, after the job, m8b_three_seeds.py and savings_figures.py):
#   python outputs/fable_review/fase_a/check_a1.py

import os

import numpy as np
import pandas as pd

OLD, NEW = "outputs/savings_under_shift/results.csv", "outputs/savings_under_shift_v2/results.csv"
KEY = ["dataset", "regime", "seed", "score", "eps", "method", "corruption", "severity", "split"]
L2_GRID = {"0.0001", "0.001", "0.01", "0.1", "1"}
pd.set_option("display.width", 220)

old, new = pd.read_csv(OLD, dtype={"l2": str}), pd.read_csv(NEW, dtype={"l2": str})
print(f"1. rows: original {len(old):,}, new {len(new):,}, difference {len(new) - len(old):,} (expected 7,200)")
assert not old.duplicated(KEY).any() and not new.duplicated(KEY).any(), "duplicated keys"

both = old.merge(new, on=KEY, how="left", suffixes=("_old", "_new"), indicator=True)
print("2. original rows found in the new file:", int((both._merge == "both").sum()), "of", len(old))
worst = 0.0
for col in [c for c in old.columns if c not in KEY]:
    a, b = both[col + "_old"], both[col + "_new"]
    if col == "l2":
        same = (a.fillna("") == b.fillna("")).all()
        print("   l2 identical:", bool(same))
    else:
        assert np.array_equal(a.isna(), b.isna()), col
        worst = max(worst, float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float)))))
print(f"   largest numeric difference on the original rows: {worst:.3g}")

added = new.merge(old[KEY], on=KEY, how="left", indicator=True)
added = added[added._merge == "left_only"]
print("3. new rows:", len(added), "; methods", added.method.unique().tolist(), "; seeds", sorted(added.seed.unique()),
      "; eps", added.eps.unique().tolist(), "; splits", sorted(added.split.unique()))
per_network = added.groupby(["dataset", "regime", "seed"]).size()
print("   networks:", len(per_network), "; rows per network min", per_network.min(), "max", per_network.max())
print("   missing values in lambda, risk, savings:", int(added[["lambda", "risk", "savings"]].isna().sum().sum()),
      "; lambda in (0, 1]:", bool(((added["lambda"] > 0) & (added["lambda"] <= 1)).all()))
m8b = new[new.method == "M8b"]
for seed, rows in m8b.groupby("seed"):
    values = pd.Series(" ".join(rows.l2).split())
    edge = values.isin(["0.0001", "1"]).mean()
    print(f"   seed {seed}: L2 strengths all in the grid {set(values.unique()) <= L2_GRID}; "
          f"share at an edge of the grid {edge:.3f}; most chosen {values.value_counts().head(3).to_dict()}")

print("4. gain of M8b over M2 at severity >= 3 (max softmax, eps 0.05, splits 0-2), by network")
case = ["dataset", "regime", "seed", "corruption", "severity", "split"]
d = new[np.isclose(new.eps, 0.05) & new.split.isin([0, 1, 2]) & (new.score == "max_softmax") & (new.severity >= 3)]
gain = d[d.method == "M8b"].set_index(case).savings - d[d.method == "M2"].set_index(case).savings
gain = gain.dropna().groupby(["dataset", "regime", "seed"]).mean().rename("gain_here").reset_index()
week4 = pd.read_csv("outputs/block3_review/m8b_three_seeds_week4.csv")
w = week4[week4.score == "max_softmax"].merge(gain, on=["dataset", "regime", "seed"], how="outer")
print(f"   networks {len(w)}; cases below 45 in {int((w.cases < 45).sum())} networks; "
      f"largest difference with m8b_three_seeds_week4.csv: {np.abs(w.gain - w.gain_here).max():.3g}")
print("   mean gain by seed:", w.groupby("seed").gain_here.mean().round(4).to_dict(),
      "; by dataset and seed:", w.groupby(["dataset", "seed"]).gain_here.mean().round(4).to_dict())

figures = "outputs/block3_review/figures_v2/adaptation_test.csv"
if os.path.exists(figures):
    f = pd.read_csv(figures, dtype={"seed": str})
    f = f[f.seed.isin(["0", "1", "2"])].astype({"seed": int})
    x = week4.merge(f, on=["score", "dataset", "regime", "seed"], how="outer", indicator=True)
    print("   m8b_three_seeds_week4.csv against figures_v2/adaptation_test.csv: rows", len(x), "; in both",
          int((x._merge == "both").sum()), "; largest differences: gain",
          f"{np.abs(x.gain - x.gain_M8b).max():.3g}, median recovery "
          f"{np.abs(x.median_recovery - x.median_recovery_M8b).max():.3g}, share "
          f"{np.abs(x.share_quarter - x.share_recovering_a_quarter).max():.3g}; same verdicts",
          bool((x.passes_x == x.passes_y).all()))
else:
    print("   ", figures, "not there yet")
