# Phase A2 of the plan v3 (14 November 2026), notebook entry X.07. Check of the sentence of finding R23, "with M8a the
# traffic moves from exits 1 to 4 to exits 5 to 8", which had no log behind it. M8a (agreement score re-fitted on the
# adapt rows of the shifted domain) against M2 with the same agreement score fitted on clean val: eps 0.05, severity
# 3 to 5, the 24 networks, 5 splits, paired cases. pi_1..pi_9 are the shares of the eval inputs leaving at each exit.
# It reads outputs/savings_under_shift/results.csv and only prints.
# Usage (from the repository root): python outputs/fable_review/fase_a/m8a_traffic.py

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
df = pd.read_csv("outputs/savings_under_shift/results.csv", dtype={"l2": str})
d = df[np.isclose(df.eps, 0.05) & (df.score == "agreement") & df.method.isin(["M2", "M8a"]) & (df.severity >= 3)]
pis = [f"pi_{i}" for i in range(1, 10)]
key = ["dataset", "regime", "seed", "corruption", "severity", "split"]
print("cases per method:", d.groupby("method").size().to_dict(), "; paired:", bool(d.groupby(key).size().eq(2).all()))
print(d.groupby(["dataset", "method"])[pis + ["savings", "lambda"]].mean().round(3).to_string())

g = d.assign(early=d[pis[:4]].sum(axis=1), late=d[pis[4:8]].sum(axis=1), last=d.pi_9)
print("\nshare of the inputs leaving at exits 1-4, 5-8 and 9, by dataset and method:")
print(g.groupby(["dataset", "method"])[["early", "late", "last", "savings"]].mean().round(3).to_string())

print("\nM8a minus M2, by dataset and regime:")
p = g.groupby(["dataset", "regime", "method"])[["early", "late", "last", "savings"]].mean().unstack("method")
print(pd.DataFrame({c: p[c]["M8a"] - p[c]["M2"] for c in ["early", "late", "last", "savings"]}).round(3).to_string())

net = g.groupby(["dataset", "regime", "seed", "method"])[["early", "late"]].mean().unstack("method")
print("\nnetworks (of 24) where M8a has less traffic at exits 1-4:", int((net["early"]["M8a"] < net["early"]["M2"]).sum()),
      "; more at exits 5-8:", int((net["late"]["M8a"] > net["late"]["M2"]).sum()))
