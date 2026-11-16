# Review of block 3: the README splits the drop of the savings from clean to M2 into "shift at the clean lambda"
# and "change of lambda" (savings_figures.decomposition). That is one of the two possible orders of a sequential
# decomposition. From the pilot outputs/fable_review/parte1/decomposition_orders.py (same logic).
# With S(D, lam) the savings on the eval rows of domain D at threshold lam, P clean and Q shifted:
#   order 1 (README):  shift = S(P, lam_P) - S(Q, lam_P),   lambda = S(Q, lam_P) - S(Q, lam_Q)
#   order 2:           lambda = S(P, lam_P) - S(P, lam_Q),   shift = S(P, lam_Q) - S(Q, lam_Q)
#   Shapley:           the mean of the two orders
# In both orders the two parts add up to the drop S(P, lam_P) - S(Q, lam_Q).
# lam_P and lam_Q are the thresholds saved in results.csv (clean and M2), so only S(P, lam_Q) is new.
# Max softmax, eps 0.05, severity 3 to 5, all 24 networks, 5 splits.
# Output: outputs/block3_review/decomposition_orders.csv and decomposition_orders.log.
#
# Usage: python scripts/6_block3_review/decomposition_orders.py [--results outputs/savings_under_shift/results.csv]

import argparse
import os

import numpy as np
import pandas as pd

from eesavings.analysis import make_splits
from eesavings.data.cifar import CORRUPTIONS
from eesavings.fys import compute_exits
from eesavings.scores import exit_scores
from review_common import OUT, RESULTS, all_runs, fractions_of, load_logits, start_log

EPS = 0.05


def savings_at(scores, lam, rows, fractions):
    return 1.0 - fractions[compute_exits(scores[rows], lam) - 1].mean()


def add_parts(out):
    # the two parts of the drop in the two orders, and their mean (Shapley), from s_pp, s_qp, s_pq and s_qq
    out["drop"] = out.s_pp - out.s_qq
    out["shift_1"], out["lambda_1"] = out.s_pp - out.s_qp, out.s_qp - out.s_qq
    out["lambda_2"], out["shift_2"] = out.s_pp - out.s_pq, out.s_pq - out.s_qq
    out["shift_shapley"] = (out.shift_1 + out.shift_2) / 2
    out["lambda_shapley"] = (out.lambda_1 + out.lambda_2) / 2
    return out


def network_rows(df, dataset, regime, seed, splits):
    fractions = fractions_of(dataset)
    d = df[(df.dataset == dataset) & (df.regime == regime) & (df.seed == seed)]
    clean = d[d.method == "clean"].set_index("split")
    m1 = d[d.method == "M1"].set_index(["corruption", "severity", "split"])
    m2 = d[d.method == "M2"].set_index(["corruption", "severity", "split"])
    clean_scores = exit_scores(load_logits(dataset, regime, seed), "max_softmax")
    rows = []
    for corruption in CORRUPTIONS:
        for severity in [3, 4, 5]:
            for t, (_, _, ev) in enumerate(splits):
                lam_p, lam_q = clean.loc[t, "lambda"], m2.loc[(corruption, severity, t), "lambda"]
                s_pp, s_qp = clean.loc[t, "savings"], m1.loc[(corruption, severity, t), "savings"]
                s_qq = m2.loc[(corruption, severity, t), "savings"]
                s_pq = savings_at(clean_scores, lam_q, ev, fractions)
                # sanity: recomputing S(P, lam_P) from the logits gives the saved clean savings
                assert abs(savings_at(clean_scores, lam_p, ev, fractions) - s_pp) < 1e-9
                rows.append({"dataset": dataset, "regime": regime, "seed": seed, "corruption": corruption,
                             "severity": severity, "split": t, "lam_p": lam_p, "lam_q": lam_q,
                             "s_pp": s_pp, "s_qp": s_qp, "s_pq": s_pq, "s_qq": s_qq})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--out-dir", default=OUT)
    args = parser.parse_args()
    start_log("decomposition_orders", args.out_dir)
    df = pd.read_csv(args.results, dtype={"l2": str})
    df = df[(df.score == "max_softmax") & np.isclose(df.eps, EPS)]
    splits = make_splits(5)
    rows = []
    for dataset, regime, seed in all_runs():
        rows += network_rows(df, dataset, regime, seed, splits)
        print(dataset, regime, seed, "done", flush=True)
    out = add_parts(pd.DataFrame(rows))
    out.to_csv(os.path.join(args.out_dir, "decomposition_orders.csv"), index=False)

    pd.set_option("display.width", 200)
    parts = ["shift_1", "lambda_1", "shift_2", "lambda_2", "shift_shapley", "lambda_shapley"]
    print("\nshare of the drop (ratio of means), by dataset:")
    g = out.groupby("dataset")[parts + ["drop"]].mean()
    print(g[parts].div(g["drop"], axis=0).round(3).to_string())
    print("\nby dataset and regime:")
    g = out.groupby(["dataset", "regime"])[parts + ["drop"]].mean()
    print(pd.concat([g[["drop"]].round(3), g[parts].div(g["drop"], axis=0).round(3)], axis=1).to_string())
    print("\nshare of the shift part per seed (order 1, order 2), to see the noise:")
    g = out.groupby(["dataset", "regime", "seed"])[["shift_1", "shift_2", "drop"]].mean()
    print((g[["shift_1", "shift_2"]].div(g["drop"], axis=0)).unstack("seed").round(3).to_string())
    print("\nshare of cases where lambda goes DOWN from clean to M2 (lam_q < lam_p): %.3f"
          % (out.lam_q < out.lam_p).mean())
    print("mean lambda: clean %.4f, M2 %.4f" % (out.lam_p.mean(), out.lam_q.mean()))


if __name__ == "__main__":
    main()
