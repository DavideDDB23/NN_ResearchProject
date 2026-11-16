# Review of block 3: what results.csv says once the dependence between cases is respected. From the pilot
# outputs/fable_review/parte1/block3_from_csv.py (sections A to G, same logic and same random draws). Only reads
# results.csv (savings_under_shift.py). Sections:
#   A) noise floor: standard deviation of the M2 savings between seeds, splits and corruptions
#   B) joint_sd against joint, seed by seed, with the paired differences of the same seed
#   C) share of the clean savings left under shift, with a hierarchical bootstrap (seed, corruption, split), and by
#      corruption family
#   D) violation rates of M1 and M2 with cluster bootstrap intervals, and by corruption family
#   E) M1 with its gross savings next to its realized risk, against M2; M1 and eps (from the pilot
#      reference_mechanical.py, section 3)
#   F) week 4 test: the table again, with an interval on the share of cases that recover a quarter, for every seed
#      with M8b rows
#   G) exit scores (H3) and M8a as paired differences over the networks
# Max softmax and eps 0.05 unless said otherwise. One random generator (seed 0) for all the bootstraps, used in
# the order of the sections. Output: outputs/block3_review/intervals.log and intervals_*.csv.
#
# Usage: python scripts/6_block3_review/intervals.py [--results outputs/savings_under_shift/results.csv]
#        [--out-dir outputs/block3_review]

import argparse
import os

import numpy as np
import pandas as pd

from review_common import DATASETS, FAMILY_OF, OUT, REGIMES, RESULTS, start_log

EPS = 0.05


def section(title):
    print("\n" + "=" * 110 + "\n" + title + "\n" + "=" * 110)


def noise_floor(ms):
    section("A) Noise floor: M2 savings, severity 3, eps 0.05, max softmax")
    m2 = ms[(ms.method == "M2") & (ms.severity == 3)]
    per_seed = m2.groupby(["dataset", "regime", "seed"]).savings.mean().unstack("seed")
    per_seed["sd_seeds"] = per_seed[[0, 1, 2]].std(axis=1, ddof=1)
    print("mean savings per seed (over 15 corruptions and 5 splits) and sd between the 3 seeds:\n",
          per_seed.round(4).to_string())
    sd_split = m2.groupby(["dataset", "regime", "seed", "corruption"]).savings.std(ddof=1)
    print("\nsd between the 5 splits, network and corruption fixed: mean %.4f, median %.4f, 90th percentile %.4f"
          % (sd_split.mean(), sd_split.median(), sd_split.quantile(0.9)))
    by_corr = m2.groupby(["dataset", "regime", "seed", "corruption"]).savings.mean()
    sd_corr = by_corr.groupby(["dataset", "regime", "seed"]).std(ddof=1)
    print("sd between the 15 corruptions, network fixed: mean %.4f (cifar10 %.4f, cifar100 %.4f)"
          % (sd_corr.mean(), sd_corr.xs("cifar10").mean(), sd_corr.xs("cifar100").mean()))
    print("\nsame for the clean savings (sd between seeds of the split mean):")
    clean = ms[ms.method == "clean"].groupby(["dataset", "regime", "seed"]).savings.mean().unstack("seed")
    clean["sd_seeds"] = clean[[0, 1, 2]].std(axis=1, ddof=1)
    print(clean.round(4).to_string())
    return per_seed


def joint_sd_against_joint(ms):
    section("B) joint_sd against joint, seed by seed (M2, max softmax, eps 0.05)")
    rows = []
    for sev in [1, 3, 5]:
        t = ms[(ms.method == "M2") & (ms.severity == sev) & ms.regime.isin(["joint", "joint_sd"])]
        t = t.groupby(["dataset", "regime", "seed"]).savings.mean().unstack("seed")
        t["mean"], t["sd"] = t[[0, 1, 2]].mean(axis=1), t[[0, 1, 2]].std(axis=1, ddof=1)
        print(f"severity {sev}:\n", t.round(3).to_string())
        for dataset in DATASETS:
            a, b = t.loc[(dataset, "joint_sd"), [0, 1, 2]].to_numpy(), t.loc[(dataset, "joint"), [0, 1, 2]].to_numpy()
            diff = a.mean() - b.mean()
            se = np.sqrt(a.var(ddof=1) / 3 + b.var(ddof=1) / 3)
            print(f"   {dataset}: joint_sd - joint = {diff:+.4f}, Welch standard error {se:.4f}, ratio "
                  f"{diff / se:+.1f}; joint_sd seeds above the best joint seed: {(a > b.max()).sum()} of 3")
            # paired by seed: the two regimes of the same seed start from the same initialization
            paired = a - b
            paired_se = paired.std(ddof=1) / np.sqrt(3)
            print(f"   {dataset}: paired differences by seed (seed 0, 1, 2): "
                  + ", ".join(f"{x:+.4f}" for x in paired)
                  + f"; mean {paired.mean():+.4f}, paired standard error {paired_se:.4f}")
            for seed in range(3):
                rows.append({"dataset": dataset, "severity": sev, "seed": seed, "joint": b[seed], "joint_sd": a[seed],
                             "difference": paired[seed]})
    return pd.DataFrame(rows)


def share_left(d, sev, eps):
    # ratio of means over regimes and seeds, as in the README
    d = d[np.isclose(d.eps, eps)]
    c = d[d.method == "clean"].groupby(["regime", "seed"]).savings.mean()
    m = d[(d.method == "M2") & (d.severity == sev)].groupby(["regime", "seed"]).savings.mean()
    return m.mean() / c.mean()


def bootstrap_share(df, dataset, sev, eps, rng, n_boot=1000):
    # resample seeds inside each regime, then corruptions, then splits (the same draw for clean and M2)
    d = df[(df.dataset == dataset) & (df.score == "max_softmax") & np.isclose(df.eps, eps)]
    m2 = d[(d.method == "M2") & (d.severity == sev)]
    corrs = sorted(m2.corruption.unique())
    cube = m2.set_index(["regime", "seed", "corruption", "split"]).savings.unstack("split")
    cube = cube.to_numpy().reshape(len(REGIMES), 3, len(corrs), 5)  # index order is sorted: regime, seed, corruption
    clean = d[d.method == "clean"].set_index(["regime", "seed", "split"]).savings.unstack("split")
    clean = clean.to_numpy().reshape(len(REGIMES), 3, 5)
    return hierarchical_bootstrap(cube, clean, rng, n_boot)


def hierarchical_bootstrap(cube, clean, rng, n_boot=1000):
    # cube (regimes, seeds, corruptions, splits) of M2 savings, clean (regimes, seeds, splits): 95% interval of
    # sum of the M2 means / sum of the clean means
    n_seeds, n_corrs, n_splits = cube.shape[1:]
    out = []
    for _ in range(n_boot):
        num, den = 0.0, 0.0
        for r in range(cube.shape[0]):
            for s in rng.integers(0, n_seeds, size=n_seeds):
                cs = rng.integers(0, n_corrs, size=n_corrs)
                sp = rng.integers(0, n_splits, size=n_splits)
                num += cube[r, s][np.ix_(cs, sp)].mean()
                den += clean[r, s][sp].mean()
        out.append(num / den)
    return np.percentile(out, [2.5, 97.5])


def shares(df, ms, rng):
    section("C) Share of the clean savings left with M2 (max softmax), hierarchical bootstrap")
    rows = []
    for dataset in DATASETS:
        d = df[(df.dataset == dataset) & (df.score == "max_softmax")]
        for sev, eps in [(1, 0.05), (3, 0.05), (5, 0.05), (3, 0.02), (3, 0.1)]:
            lo, hi = bootstrap_share(df, dataset, sev, eps, rng)
            share = share_left(d, sev, eps)
            print(f"{dataset} severity {sev} eps {eps}: share left {share:.3f}  "
                  f"95% hierarchical bootstrap [{lo:.3f}, {hi:.3f}]")
            rows.append({"dataset": dataset, "severity": sev, "eps": eps, "share_left": share, "low": lo, "high": hi})
    print("\nby corruption family (severity 3, eps 0.05): M2 savings / clean savings, mean over regimes and seeds")
    fam = ms[(ms.method == "M2") & (ms.severity == 3)].groupby(["dataset", "family"]).savings.mean().unstack("family")
    cl = ms[ms.method == "clean"].groupby("dataset").savings.mean()
    print(fam.div(cl, axis=0).round(3).to_string())
    print("\nby corruption (severity 3, eps 0.05): M2 savings / clean savings")
    cor = ms[(ms.method == "M2") & (ms.severity == 3)].groupby(["dataset", "corruption"]).savings.mean()
    cor = cor.unstack("dataset")
    print(cor.div(cl, axis=1).round(2).sort_values("cifar10").T.to_string())
    return pd.DataFrame(rows), fam.div(cl, axis=0)


def cluster_boot_rate(d, flag, rng, n_boot=1000):
    # two-level cluster bootstrap: networks (regime, seed), then corruptions; splits and severities stay together
    d = d.assign(flag=flag.astype(float), net=d.regime + d.seed.astype(str))
    table = d.groupby(["net", "corruption"]).flag.mean().unstack("corruption").to_numpy()
    return table_bootstrap(table, rng, n_boot)


def table_bootstrap(table, rng, n_boot=1000):
    # 95% interval of the mean of a (clusters, corruptions) table, resampling rows and columns
    out = []
    for _ in range(n_boot):
        nets = rng.integers(0, table.shape[0], size=table.shape[0])
        cors = rng.integers(0, table.shape[1], size=table.shape[1])
        out.append(table[np.ix_(nets, cors)].mean())
    return np.percentile(out, [2.5, 97.5])


def violations(df, ms, rng):
    section("D) Violation rates (risk on the 3,000 eval rows above eps), max softmax, eps 0.05")
    rows = []
    for method in ["M1", "M2"]:
        for dataset in DATASETS:
            for sev in [1, 2, 3, 4, 5]:
                d = ms[(ms.method == method) & (ms.dataset == dataset) & (ms.severity == sev)]
                flag = d.risk > EPS
                se = np.sqrt(EPS * (1 - EPS) / 3000)
                lo, hi = cluster_boot_rate(d, flag, rng)
                naive = np.sqrt(flag.mean() * (1 - flag.mean()) / len(d))
                print(f"{method} {dataset} severity {sev}: mean risk {d.risk.mean():.4f}, cases above eps "
                      f"{flag.mean():.3f} [{lo:.3f}, {hi:.3f}] (naive binomial se {naive:.3f}), "
                      f"above eps + 2 se {(d.risk > EPS + 2 * se).mean():.3f}")
                rows.append({"method": method, "dataset": dataset, "severity": sev, "mean_risk": d.risk.mean(),
                             "above_eps": flag.mean(), "low": lo, "high": hi, "naive_se": naive,
                             "above_eps_plus_2se": (d.risk > EPS + 2 * se).mean()})
    d = ms[ms.method == "M2"]
    print("\nM2, all severities: cases above eps %.4f; by dataset:" % (d.risk > EPS).mean(),
          (d.risk > EPS).groupby(d.dataset).mean().round(4).to_dict())
    print("M2 mean risk %.4f, median %.4f; clean: cases above eps %.3f (n = %d), mean risk %.4f"
          % (d.risk.mean(), d.risk.median(), (ms[ms.method == "clean"].risk > EPS).mean(),
             (ms.method == "clean").sum(), ms[ms.method == "clean"].risk.mean()))
    print("M2 cases above eps by score and eps (all severities):")
    allm2 = df[df.method == "M2"]
    print((allm2.risk > allm2.eps).groupby([allm2.score, allm2.eps]).mean().unstack().round(3).to_string())
    print("M2 cases above eps by family (max softmax, eps 0.05):",
          (d.risk > EPS).groupby(d.family).mean().round(3).to_dict())
    print("M1 cases above eps by family and dataset (severity 3):")
    d1 = ms[(ms.method == "M1") & (ms.severity == 3)]
    print((d1.risk > EPS).groupby([d1.dataset, d1.family]).mean().unstack().round(2).to_string())
    print("M1 mean risk by family and dataset (severity 3):")
    print(d1.groupby(["dataset", "family"]).risk.mean().unstack().round(3).to_string())
    return pd.DataFrame(rows)


def m1_against_m2(df, ms):
    section("E) M1: raw savings and realized risk against M2 (max softmax, eps 0.05), mean over regimes, seeds, "
            "corruptions")
    t = ms[ms.method.isin(["M1", "M2"])].groupby(["dataset", "severity", "method"]).agg(
        risk=("risk", "mean"), savings=("savings", "mean"), valid_savings=("valid_savings", "mean")).unstack("method")
    print(t.round(3).to_string())
    print("\nhow much the 'valid savings' of M1 depend on the 3,000 eval rows: share of M1 cases with risk within 2 se "
          "of eps (their valid savings flip between 0 and the full value with the eval sample):")
    se = np.sqrt(EPS * (1 - EPS) / 3000)
    d1 = ms[ms.method == "M1"]
    print(((d1.risk - EPS).abs() < 2 * se).groupby([d1.dataset, d1.severity]).mean().unstack().round(2).to_string())

    # from the pilot reference_mechanical.py, section 3
    print("\nsensitivity to eps (max softmax, n_cal 3000): M1 cases above eps and mean risk at severity 3, M2 share "
          "left is in calibration_budget.log")
    m1 = df[(df.score == "max_softmax") & (df.method == "M1") & (df.severity == 3)]
    print(m1.assign(viol=m1.risk > m1.eps).groupby(["dataset", "eps"]).agg(
        above_eps=("viol", "mean"), risk=("risk", "mean"), savings=("savings", "mean")).round(3).to_string())
    m1s1 = df[(df.score == "max_softmax") & (df.method == "M1") & np.isclose(df.eps, EPS)]
    m2 = df[(df.score == "max_softmax") & (df.method == "M2") & np.isclose(df.eps, EPS)]
    key = ["dataset", "regime", "seed", "corruption", "severity", "split"]
    both = m1s1.set_index(key)[["risk", "savings"]].join(m2.set_index(key)[["savings"]].rename(
        columns={"savings": "savings_m2"}))
    both["ok_and_more"] = (both.risk <= EPS) & (both.savings > both.savings_m2)
    print("\nM1 cases with risk <= eps, and with risk <= eps AND more savings than M2, by dataset and severity:")
    print(both.assign(ok=both.risk <= EPS).groupby(["dataset", "severity"])[["ok", "ok_and_more"]].mean()
          .round(3).to_string())
    return t


def week4_test(df, rng):
    section("F) Week 4 test on the adaptation (eps 0.05, severity >= 3), with intervals, for every seed with M8b")
    rows = []
    seeds = sorted(df[df.method == "M8b"].seed.unique())
    for score in ["max_softmax", "agreement"]:
        for seed in seeds:
            d = df[(df.seed == seed) & (df.score == score) & np.isclose(df.eps, EPS) & df.split.isin([0, 1, 2])]
            clean0 = d[d.method == "clean"].groupby(["dataset", "regime"]).savings.mean()
            m = d[d.method.isin(["M2", "M8b"])].groupby(["dataset", "regime", "corruption", "severity", "method"])
            m = m.savings.mean().unstack("method").reset_index()
            m["drop"] = [clean0[a, b] for a, b in zip(m.dataset, m.regime)] - m.M2
            m["gain"] = m.M8b - m.M2
            m["rec"] = np.where(m["drop"] >= 0.01, m.gain / m["drop"], np.nan)
            hard = m[(m.severity >= 3) & m.rec.notna()]
            print(f"\nscore {score}, seed {seed}:")
            for (dataset, regime), g in hard.groupby(["dataset", "regime"]):
                share = (g.rec >= 0.25).mean()
                lo, hi = corruption_bootstrap(g, rng)
                # pooled recovery: total gain / total drop (not a median of ratios)
                print(f"  {dataset:8s} {regime:8s} cases {len(g)}: mean drop {g['drop'].mean():.3f} gain "
                      f"{g.gain.mean():.3f} median recovery {g.rec.median():.2f}, pooled recovery "
                      f"{g.gain.sum() / g['drop'].sum():.2f}, share >= 1/4: {share:.2f} [{lo:.2f}, {hi:.2f}] -> "
                      f"{'passes' if share >= 0.5 else 'fails'}")
                rows.append({"score": score, "seed": seed, "dataset": dataset, "regime": regime, "cases": len(g),
                             "drop": g["drop"].mean(), "gain": g.gain.mean(), "median_recovery": g.rec.median(),
                             "pooled_recovery": g.gain.sum() / g["drop"].sum(), "share_recovering_a_quarter": share,
                             "low": lo, "high": hi, "passes": share >= 0.5})
            # sd of the gain between the 3 splits (same network and domain): the noise of one M8b - M2 difference
            g = d[d.method.isin(["M2", "M8b"])].set_index(["dataset", "regime", "corruption", "severity", "split",
                                                            "method"])
            g = g.savings.unstack("method")
            g["gain"] = g.M8b - g.M2
            print("  sd of the paired gain between the 3 splits: %.4f (mean gain %.4f)"
                  % (g.gain.groupby(["dataset", "regime", "corruption", "severity"]).std(ddof=1).mean(), g.gain.mean()))
            print("  mean gain by severity:",
                  g.gain.groupby(["dataset", "severity"]).mean().unstack().round(3).to_dict("index"))
            print("  share of (network, domain, split) rows where M8b is BELOW M2: %.3f" % (g.gain < 0).mean())
            risk = d[d.method == "M8b"]
            print("  M8b cases above eps: %.3f (n = %d)" % ((risk.risk > EPS).mean(), len(risk)))
    return pd.DataFrame(rows)


def corruption_bootstrap(g, rng, n_boot=2000):
    # the 45 cases are 15 corruptions x 3 severities: resample the corruptions (cluster) to get an interval on the
    # share of cases that recover at least a quarter
    cors = g.corruption.unique()
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(cors, size=len(cors))
        boot.append(np.mean(np.concatenate([(g[g.corruption == c].rec >= 0.25).to_numpy() for c in pick])))
    return np.percentile(boot, [2.5, 97.5])


def scores_and_m8a(df):
    section("G) Exit scores (H3) and M8a: paired differences over the networks (M2, eps 0.05, severity 3 to 5)")
    d = df[np.isclose(df.eps, EPS) & (df.method == "M2") & (df.severity >= 3)]
    by = d.groupby(["dataset", "regime", "seed", "score"]).savings.mean().unstack("score")
    for other in ["temperature", "label", "agreement"]:
        diff = (by[other] - by["max_softmax"]).unstack("seed")
        diff["mean"], diff["sd_seeds"] = diff[[0, 1, 2]].mean(axis=1), diff[[0, 1, 2]].std(axis=1, ddof=1)
        print(f"\n{other} - max_softmax, per seed:\n", diff.round(4).to_string())
    cl = df[np.isclose(df.eps, EPS) & (df.method == "clean")].groupby(["dataset", "regime", "seed", "score"])
    cl = cl.savings.mean().unstack("score")
    cl_mean, by_mean = cl.groupby(["dataset", "regime"]).mean(), by.groupby(["dataset", "regime"]).mean()
    print("\nclean savings by score, mean over seeds:\n", cl_mean.round(3).to_string())
    print("largest difference between two scores, clean: %.3f; M2 severity 3 to 5: %.3f"
          % ((cl_mean.max(axis=1) - cl_mean.min(axis=1)).max(), (by_mean.max(axis=1) - by_mean.min(axis=1)).max()))
    a = df[np.isclose(df.eps, EPS) & (df.score == "agreement") & (df.severity >= 3) & df.method.isin(["M2", "M8a"])]
    a = a.groupby(["dataset", "regime", "seed", "method"]).savings.mean().unstack("method")
    a["diff"] = a.M8a - a.M2
    print("\nM8a - M2 (agreement score), per network: mean by dataset",
          a["diff"].groupby("dataset").mean().round(4).to_dict(),
          "; networks where M8a is below M2: %d of %d" % ((a["diff"] < 0).sum(), len(a)))
    print(a["diff"].unstack("seed").round(4).to_string())
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--out-dir", default=OUT)
    args = parser.parse_args()
    start_log("intervals", args.out_dir)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    rng = np.random.default_rng(0)

    df = pd.read_csv(args.results, dtype={"l2": str})
    df["family"] = df.corruption.map(FAMILY_OF)
    ms = df[(df.score == "max_softmax") & np.isclose(df.eps, EPS)]

    out = args.out_dir
    noise_floor(ms).to_csv(os.path.join(out, "intervals_noise_floor.csv"))
    joint_sd_against_joint(ms).to_csv(os.path.join(out, "intervals_joint_sd.csv"), index=False)
    share, family = shares(df, ms, rng)
    share.to_csv(os.path.join(out, "intervals_share_left.csv"), index=False)
    family.to_csv(os.path.join(out, "intervals_share_left_by_family.csv"))
    violations(df, ms, rng).to_csv(os.path.join(out, "intervals_violations.csv"), index=False)
    m1_against_m2(df, ms).to_csv(os.path.join(out, "intervals_m1_against_m2.csv"))
    week4_test(df, rng).to_csv(os.path.join(out, "intervals_week4.csv"), index=False)
    scores_and_m8a(df).to_csv(os.path.join(out, "intervals_m8a.csv"))
    print("\nsaved in", out)


if __name__ == "__main__":
    main()
