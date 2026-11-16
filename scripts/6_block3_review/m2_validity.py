# Review of block 3: is the 8-11% of M2 cases above eps only noise of the 3,000 eval rows? From the pilots
# outputs/fable_review/parte1/m2_validity.py and m2_validity_analysis.py (same logic and same random draws).
# For every network and every domain (clean, and the 15 standard corruptions at 5 severities), 100 random splits
# (the first 5 are the splits of results.csv). For each split lambda is calibrated with UCB on the 3,000 cal rows
# (max softmax, eps 0.05, delta 0.1) and its risk is measured three ways:
#   eval     the 3,000 eval rows, as in results.csv
#   noncal   the 7,000 rows outside the calibration (adapt + eval: M2 with max softmax never uses the adapt rows)
#   all      the whole 10,000 rows, the finite population the calibration rows are drawn from
# Two oracles of the threshold on the whole population:
#   oracle_fst   the last lambda of the fixed sequence before the first one with population risk above eps
#   oracle_best  the lambda with the highest savings among all those with population risk <= eps
# Then: 1) the violation rate on eval, noncal and all, per domain and overall, with a cluster bootstrap over
# networks and corruptions; 2) the worst domains; 3) the savings left on the table by M2 against the two oracles.
# Output: outputs/block3_review/m2_validity.csv (one row per network, domain and split) and m2_validity.log.
#
# Usage: python scripts/6_block3_review/m2_validity.py [--workers 3] [--splits 100] [--from-csv]
#        (--from-csv: only the analysis, from the m2_validity.csv already written)

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from eesavings.analysis import LAMBDAS, consistency_losses, make_splits, ucb_index_fast
from eesavings.data.cifar import CORRUPTIONS, SEVERITIES
from eesavings.scores import exit_scores
from review_common import FAMILY_OF, OUT, all_runs, fractions_of, load_logits, start_log

EPS, DELTA = 0.05, 0.1


def oracles(risk_all, savings_all):
    # lambda index of the two oracles of the threshold, from the population risk and savings of every lambda
    above = np.flatnonzero(risk_all > EPS)
    k_fst = len(risk_all) - 1 if len(above) == 0 else max(0, above[0] - 1)
    k_best = int(np.argmax(np.where(risk_all <= EPS, savings_all, -np.inf)))
    return k_fst, k_best


def one_network(job):
    dataset, regime, seed, n_splits = job
    fractions = fractions_of(dataset)
    splits = make_splits(n_splits)
    rows = []
    domains = [("none", 0)] + [(c, s) for c in CORRUPTIONS for s in SEVERITIES]
    for corruption, severity in domains:
        logits = load_logits(dataset, regime, seed, None if severity == 0 else corruption, severity)
        losses, exits = consistency_losses(logits, exit_scores(logits, "max_softmax"))
        savings_all = 1.0 - fractions[exits - 1].mean(axis=1)  # (K,) on the whole population
        risk_all = losses.mean(axis=1)
        k_fst, k_best = oracles(risk_all, savings_all)
        for t, (adapt, cal, ev) in enumerate(splits):
            k = ucb_index_fast(losses[:, cal], EPS, DELTA)
            noncal = np.concatenate([adapt, ev])
            rows.append({
                "dataset": dataset, "regime": regime, "seed": seed, "corruption": corruption, "severity": severity,
                "split": t, "k": k, "lambda": LAMBDAS[k],
                "risk_eval": losses[k, ev].mean(), "risk_noncal": losses[k, noncal].mean(), "risk_all": risk_all[k],
                "savings_eval": 1.0 - fractions[exits[k, ev] - 1].mean(), "savings_all": savings_all[k],
                "k_oracle_fst": k_fst, "savings_oracle_fst": savings_all[k_fst], "risk_oracle_fst": risk_all[k_fst],
                "k_oracle_best": k_best, "savings_oracle_best": savings_all[k_best],
            })
    print(dataset, regime, seed, "done", flush=True)
    return rows


def violation_rates(df, rng):
    print("\n1) violation rate of M2 (share of calibrations whose lambda has risk above eps), 100 splits per domain")
    shifted = df[df.severity > 0]
    for label, d in [("clean", df[df.severity == 0]), ("shifted", shifted)]:
        for dataset, g in d.groupby("dataset"):
            line = f"{label:8s} {dataset:8s}"
            for name in ["eval", "noncal", "all"]:
                table = g.groupby(["net", "corruption"])[f"viol_{name}"].mean().unstack("corruption").to_numpy()
                boot = []
                for _ in range(1000):
                    a = rng.integers(0, table.shape[0], size=table.shape[0])
                    b = rng.integers(0, table.shape[1], size=table.shape[1])
                    boot.append(table[np.ix_(a, b)].mean())
                lo, hi = np.percentile(boot, [2.5, 97.5])
                line += f" | {name}: {g[f'viol_{name}'].mean():.4f} [{lo:.3f}, {hi:.3f}]"
            print(line)
    print("   mean risk at the chosen lambda: eval %.4f, noncal %.4f, all %.4f; sd over calibrations of risk_all %.4f"
          % (shifted.risk_eval.mean(), shifted.risk_noncal.mean(), shifted.risk_all.mean(), shifted.risk_all.std()))


def per_domain(df):
    print("\n2) per domain (network x corruption x severity): distribution of the violation rate over the 100 splits")
    shifted = df[df.severity > 0]
    per = df.groupby(["net", "domain"])[["viol_eval", "viol_noncal", "viol_all"]].mean()
    print(per.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(3).to_string())
    for name in ["viol_noncal", "viol_all"]:
        # with 100 splits a rate above 0.1 + 2 * sqrt(0.1 * 0.9 / 100) = 0.16 would be evidence against the guarantee
        print(f"   domains with {name} above delta = 0.1: {(per[name] > DELTA).mean():.3f}; above 0.16: "
              f"{(per[name] > 0.16).mean():.4f} ({(per[name] > 0.16).sum()} of {len(per)})")
    print("   worst 8 domains by viol_all:\n", per.sort_values("viol_all", ascending=False).head(8).round(3).to_string())
    fam = shifted.assign(family=shifted.corruption.map(FAMILY_OF))
    print("   by family: ",
          fam.groupby("family")[["viol_eval", "viol_noncal", "viol_all"]].mean().round(3).to_dict("index"))
    print("   by severity:",
          shifted.groupby("severity")[["viol_eval", "viol_noncal", "viol_all"]].mean().round(3).to_dict("index"))


def against_oracles(df):
    print("\n3) savings on the whole population: M2 against the oracles of the threshold (mean over networks and "
          "domains)")
    first5 = df[df.split < 5]
    g = df.groupby(["dataset", "severity"])[["savings_all", "savings_oracle_fst", "savings_oracle_best", "risk_all",
                                             "risk_oracle_fst"]].mean()
    g["price_fst"] = g.savings_oracle_fst - g.savings_all
    g["price_best"] = g.savings_oracle_best - g.savings_all
    print(g.round(4).to_string())
    print("   how often the best oracle differs from the fixed sequence oracle: %.3f of the domains; mean gap %.4f"
          % ((df.k_oracle_best != df.k_oracle_fst).groupby([df.net, df.domain]).first().mean(),
             (df.savings_oracle_best - df.savings_oracle_fst).mean()))
    print("   sd of the M2 savings over the 100 calibrations of a domain (population savings, so only the calibration "
          "noise): %.4f" % df.groupby(["net", "domain"]).savings_all.std().mean())
    print("   sd of savings_eval over the first 5 splits: %.4f; over all 100: %.4f"
          % (first5.groupby(["net", "domain"]).savings_eval.std().mean(),
             df.groupby(["net", "domain"]).savings_eval.std().mean()))
    return g


def analyse(df, out_dir):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)
    df["net"] = df.dataset + "_" + df.regime + "_" + df.seed.astype(str)
    df["domain"] = df.corruption + "_" + df.severity.astype(str)
    for name in ["eval", "noncal", "all"]:
        df[f"viol_{name}"] = df[f"risk_{name}"] > EPS
    print("rows", len(df), "networks", df.net.nunique(), "domains per network", df.domain.nunique(), "splits",
          df.split.nunique())
    first5 = df[df.split < 5]
    print("\ncheck against results.csv conventions (first 5 splits, shifted domains): cases above eps on eval %.4f"
          % first5[first5.severity > 0].viol_eval.mean())
    violation_rates(df, rng)
    per_domain(df)
    against_oracles(df).to_csv(os.path.join(out_dir, "m2_validity_oracles.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--splits", type=int, default=100)
    parser.add_argument("--out-dir", default=OUT)
    parser.add_argument("--from-csv", action="store_true", help="only the analysis, from m2_validity.csv")
    args = parser.parse_args()
    start_log("m2_validity", args.out_dir)
    path = os.path.join(args.out_dir, "m2_validity.csv")
    if not args.from_csv:
        jobs = [(d, r, s, args.splits) for d, r, s in all_runs()]
        with Pool(args.workers) as pool:
            results = pool.map(one_network, jobs, chunksize=1)
        df = pd.DataFrame([row for rows in results for row in rows])
        df.to_csv(path, index=False)
        print("saved", len(df), "rows")
    analyse(pd.read_csv(path), args.out_dir)


if __name__ == "__main__":
    main()
