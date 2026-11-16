# Review of block 3: sensitivity of M2 (max softmax) to the calibration budget and to the guarantee. From the pilot
# outputs/fable_review/parte1/sensitivity.py (same logic):
#   n_cal in 100, 250, 500, 1000, 2000, 3000   at delta 0.1 and eps in 0.02, 0.05, 0.1   (UCB with WSR)
#   delta in 0.01, 0.05, 0.2                    at n_cal 3000 and eps 0.05                 (UCB with WSR)
#   CRC (risk below eps on average, no delta)   at eps 0.05 and every n_cal
# The calibration rows are the first n_cal rows of the cal split (a random subset), the eval rows are always the
# same 3,000, so every configuration is measured on the same rows. Domains: clean and the 15 standard corruptions at
# severity 1, 3 and 5; all 24 networks; the 5 splits of results.csv.
# Output: outputs/block3_review/calibration_budget.csv and calibration_budget.log.
#
# Usage: python scripts/6_block3_review/calibration_budget.py [--workers 3]

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from eesavings.analysis import LAMBDAS, consistency_losses, make_splits, ucb_index_fast
from eesavings.data.cifar import CORRUPTIONS
from eesavings.fys import crc_lam
from eesavings.scores import exit_scores
from review_common import OUT, all_runs, fractions_of, load_logits, start_log

N_CALS = [100, 250, 500, 1000, 2000, 3000]
CONFIGS = [("ucb", n, 0.1, eps) for n in N_CALS for eps in (0.02, 0.05, 0.1)]
CONFIGS += [("ucb", 3000, delta, 0.05) for delta in (0.01, 0.05, 0.2)]
CONFIGS += [("crc", n, np.nan, 0.05) for n in N_CALS]


def one_network(job):
    dataset, regime, seed = job
    fractions = fractions_of(dataset)
    splits = make_splits(5)
    rows = []
    for corruption, severity in [("none", 0)] + [(c, s) for c in CORRUPTIONS for s in (1, 3, 5)]:
        logits = load_logits(dataset, regime, seed, None if severity == 0 else corruption, severity)
        losses, exits = consistency_losses(logits, exit_scores(logits, "max_softmax"))
        for t, (_, cal, ev) in enumerate(splits):
            for method, n_cal, delta, eps in CONFIGS:
                cal_losses = losses[:, cal[:n_cal]]
                k = ucb_index_fast(cal_losses, eps, delta) if method == "ucb" else crc_lam(cal_losses, eps)
                rows.append({"dataset": dataset, "regime": regime, "seed": seed, "corruption": corruption,
                             "severity": severity, "split": t, "method": method, "n_cal": n_cal, "delta": delta,
                             "eps": eps, "lambda": LAMBDAS[k], "risk": losses[k, ev].mean(),
                             "savings": 1.0 - fractions[exits[k, ev] - 1].mean()})
    print(dataset, regime, seed, "done", flush=True)
    return rows


def summary(df):
    pd.set_option("display.width", 220)
    df["viol"] = df.risk > df.eps
    base = df[(df.method == "ucb") & (df.delta == 0.1)]
    for eps in (0.02, 0.05, 0.1):
        d = base[np.isclose(base.eps, eps)]
        s = d.groupby(["dataset", "n_cal", "severity"]).savings.mean().unstack("severity")
        s["left_3"], s["left_5"] = s[3] / s[0], s[5] / s[0]
        print(f"\nUCB, delta 0.1, eps {eps}: savings by n_cal and severity, and share of the clean savings left\n",
              s.round(3).to_string())
    d = df[(df.method == "ucb") & (df.n_cal == 3000) & np.isclose(df.eps, 0.05)]
    s = d.groupby(["dataset", "delta", "severity"]).savings.mean().unstack("severity")
    s["left_3"] = s[3] / s[0]
    print("\nUCB, n_cal 3000, eps 0.05: savings by delta\n", s.round(3).to_string())
    print("cases above eps by delta:", d.groupby("delta").viol.mean().round(3).to_dict())
    d = df[np.isclose(df.eps, 0.05) & ((df.method == "crc") | (df.delta == 0.1))]
    s = d.groupby(["dataset", "method", "n_cal"]).agg(
        savings_clean=("savings", lambda x: x[d.loc[x.index, "severity"] == 0].mean()),
        savings_sev3=("savings", lambda x: x[d.loc[x.index, "severity"] == 3].mean()),
        risk=("risk", "mean"), above_eps=("viol", "mean"))
    s["left_3"] = s.savings_sev3 / s.savings_clean
    print("\nCRC against UCB (delta 0.1) at eps 0.05: savings, mean eval risk and cases above eps\n",
          s.round(3).to_string())
    worst = d[d.method == "crc"].groupby(["dataset", "regime", "seed", "corruption", "severity", "n_cal"]).risk.mean()
    print("\nCRC: mean over the 5 splits of the eval risk, per domain; share of domains with mean above eps, by n_cal:",
          (worst > 0.05).groupby("n_cal").mean().round(3).to_dict(), "; largest domain mean by n_cal:",
          worst.groupby("n_cal").max().round(3).to_dict())
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out-dir", default=OUT)
    args = parser.parse_args()
    start_log("calibration_budget", args.out_dir)
    with Pool(args.workers) as pool:
        results = pool.map(one_network, all_runs(), chunksize=1)
    df = pd.DataFrame([row for rows in results for row in rows])
    df.to_csv(os.path.join(args.out_dir, "calibration_budget.csv"), index=False)
    print("saved", len(df), "rows")
    summary(df).to_csv(os.path.join(args.out_dir, "calibration_budget_crc_against_ucb.csv"))


if __name__ == "__main__":
    main()
