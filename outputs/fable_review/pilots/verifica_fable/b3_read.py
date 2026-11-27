# Reading of pilot B3 (15 November 2026), from the outputs of b3_decomp.py and b3_scores.py.
#   Part 1 (C1)  the drop of the oracle savings split into traffic and reliability in the two orders; shares as ratios
#                of means; two-way bootstrap (networks x corruptions) for the reliability share; checks of the model;
#                the risk at the clean oracle lambda with the two reliabilities; AUROC of the max softmax by exit
#   Part 2 (C2)  gains of the learned scores over M2 (max softmax), with intervals; cases above eps; AUROC; M8a (from
#                results.csv) and scale and bias per class (logit_diag_n4000 of B2.03) on the same cases; payback
# It only prints. Usage (from the repository root): python outputs/fable_review/pilots/verifica_fable/b3_read.py

import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/6_block3_review")
from review_common import FAMILY_OF, fractions_of  # noqa: E402

B3 = "outputs/fable_review/pilots/B3_meccanismo"
B2 = "outputs/fable_review/pilots/B2_adattamento"
RES = "outputs/savings_under_shift_v2/results.csv"
KEY = ["dataset", "regime", "seed", "corruption", "severity", "split"]
REPS = 2000


def read(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pattern))], ignore_index=True)


def table(df, col):
    d = df.assign(network=df.dataset + "/" + df.regime + "/" + df.seed.astype(str))
    return d.groupby(["network", "corruption"])[col].mean().unstack("corruption")


def boot(num, den=None, reps=REPS, seed=0):
    # 2.5 and 97.5 percentiles of mean(num) (/ mean(den)), networks and corruptions resampled together
    a = num.to_numpy()
    b = None if den is None else den.reindex(index=num.index, columns=num.columns).to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        i, j = rng.integers(a.shape[0], size=a.shape[0]), rng.integers(a.shape[1], size=a.shape[1])
        m = np.nanmean(a[np.ix_(i, j)])
        out[r] = m if b is None else m / np.nanmean(b[np.ix_(i, j)])
    return np.percentile(out, [2.5, 97.5])


def part1():
    d = read(f"{B3}/decomp/*.csv")
    clean, d = d[d.severity == 0], d[d.severity > 0].copy()
    d["drop"] = d.S_PP - d.S_QQ
    d["traffic_1"], d["rel_1"] = d.S_PP - d.S_QP, d.S_QP - d.S_QQ
    d["rel_2"], d["traffic_2"] = d.S_PP - d.S_PQ, d.S_PQ - d.S_QQ
    d["family"] = d.corruption.map(FAMILY_OF)
    parts = ["traffic_1", "rel_1", "rel_2", "traffic_2"]
    print(f"Part 1: {d[['dataset', 'regime', 'seed']].drop_duplicates().shape[0]} networks, {len(d)} shifted domains")

    print("\n1a. Checks of the reliability model")
    print("   largest gap along the grid between model and empirical risk: clean mean "
          f"{clean.gap_P.mean():.4f} max {clean.gap_P.max():.4f}; shifted mean {d.gap_Q.mean():.4f} max "
          f"{d.gap_Q.max():.4f}; share of shifted domains above 0.01: {(d.gap_Q > 0.01).mean():.3f}")
    print(f"   oracle savings, model minus empirical: clean mean {(clean.S_PP - clean.S_PP_emp).mean():+.4f} max abs "
          f"{(clean.S_PP - clean.S_PP_emp).abs().max():.4f}; shifted mean {(d.S_QQ - d.S_QQ_emp).mean():+.4f} "
          f"max abs {(d.S_QQ - d.S_QQ_emp).abs().max():.4f}")

    print("\n1b. Drop and shares of the drop (ratio of means), by dataset x severity")
    g = d.groupby(["dataset", "severity"])[parts + ["drop"]].mean()
    out = g[["drop"]].copy()
    for p in parts:
        out[p] = g[p] / g["drop"]
    print(out.round(3).to_string())

    print("\n1c. Reliability share at severity 3 by dataset x regime, both orders, with a two-way bootstrap interval")
    for (dataset, regime), x in d[d.severity == 3].groupby(["dataset", "regime"]):
        s = []
        for p in ["rel_1", "rel_2"]:
            lo, hi = boot(table(x, p), table(x, "drop"))
            s.append(f"{p} {x[p].mean() / x['drop'].mean():.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"   {dataset} {regime:9s} drop {x['drop'].mean():.3f}; " + "; ".join(s))
    print("   C1 (CIFAR10, joint and joint_sd, severity 3):")
    x = d[(d.dataset == "cifar10") & d.regime.isin(["joint", "joint_sd"]) & (d.severity == 3)]
    for p in ["rel_1", "rel_2"]:
        lo, hi = boot(table(x, p), table(x, "drop"))
        print(f"      {p}: {x[p].mean() / x['drop'].mean():.3f} [{lo:.3f}, {hi:.3f}]")
    print("   by seed (CIFAR10 joint and joint_sd, severity 3):",
          (x.groupby("seed")[["rel_1", "rel_2"]].mean().div(x.groupby("seed")["drop"].mean(), axis=0)).round(3)
          .to_dict())

    print("\n1d. Reliability share at severity 3 by family (order 1 and order 2)")
    g = d[d.severity == 3].groupby(["dataset", "family"])[["rel_1", "rel_2", "drop"]].mean()
    print(pd.DataFrame({"drop": g["drop"], "rel_1": g.rel_1 / g["drop"], "rel_2": g.rel_2 / g["drop"]})
          .round(3).to_string())

    print("\n1e. Risk at the clean oracle lambda: empirical, model with the shifted reliability, model with the clean "
          "reliability; savings there")
    print(d.groupby(["dataset", "severity"])[["risk_Q_at_lamPP_emp", "risk_QQ_at_lamPP", "risk_QP_at_lamPP",
                                              "savings_Q_at_lamPP", "S_PP"]].mean().round(4).to_string())
    print("   lambda of the oracles (mean): ",
          d.groupby(["dataset", "severity"])[["lam_PP", "lam_QQ", "lam_QP", "lam_PQ"]].mean().round(4).to_dict())

    print("\n1f. AUROC of the max softmax of exit l for its agreement with exit 9 (mean over networks and domains)")
    a = pd.concat([clean, d]).groupby(["dataset", "severity"])[[f"auroc_{l}" for l in range(1, 9)]].mean()
    a["mean_1_8"] = a.mean(axis=1)
    print(a.round(3).to_string())


def part2():
    s = read(f"{B3}/scores/*.csv")
    print(f"\n\nPart 2: {s[['dataset', 'regime', 'seed']].drop_duplicates().shape[0]} networks, "
          f"{int((s.config == 'm2').sum())} cases")
    res = pd.read_csv(RES, usecols=KEY + ["score", "eps", "method", "lambda", "risk", "savings"])
    res = res[np.isclose(res.eps, 0.05) & (res.split == 0) & res.severity.isin([3, 5])]
    m2ref = res[(res.method == "M2") & (res.score == "max_softmax")]
    m = s[s.config == "m2"].merge(m2ref, on=KEY, suffixes=("", "_ref"))
    print(f"2a. Check: m2 against M2 of results.csv, {len(m)} rows, largest differences savings "
          f"{np.abs(m.savings - m.savings_ref).max():.2e}, lambda {np.abs(m['lambda'] - m.lambda_ref).max():.2e}")

    base = s[s.config == "m2"].set_index(KEY)
    idx = pd.MultiIndex.from_frame(s[KEY])
    s["gain"] = s.savings.to_numpy() - base.savings.reindex(idx).to_numpy()
    s["above"] = (s.risk > 0.05).astype(float)
    s["above_m2"] = (base.risk.reindex(idx).to_numpy() > 0.05).astype(float)
    # M8a (agreement score re-fitted on the adapt rows) and scale and bias per class (B2.03) on the same cases
    m8a = res[(res.method == "M8a") & (res.score == "agreement")][KEY + ["savings", "risk"]].assign(config="M8a")
    diag = read(f"{B2}/runs_mech24/*.csv")
    diag = diag[(diag.config == "logit_diag_n4000") & (diag.split == 0)][KEY + ["savings", "risk"]]
    extra = pd.concat([m8a, diag.assign(config="logit_diag_n4000 (B2.03)")], ignore_index=True)
    eidx = pd.MultiIndex.from_frame(extra[KEY])
    extra["gain"] = extra.savings.to_numpy() - base.savings.reindex(eidx).to_numpy()
    extra["above"] = (extra.risk > 0.05).astype(float)
    extra["above_m2"] = (base.risk.reindex(eidx).to_numpy() > 0.05).astype(float)
    allc = pd.concat([s, extra], ignore_index=True).dropna(subset=["gain"])
    order = ["ms_cost", "cur_clean", "cur_clean_cost", "cur_adapt", "cur_adapt_cost", "hist_clean",
             "hist_clean_cost", "hist_adapt", "hist_adapt_cost", "M8a", "logit_diag_n4000 (B2.03)"]

    print("\n2b. Gain over M2 (max softmax) by dataset x severity, two-way bootstrap interval; cases above eps")
    for (dataset, severity), g in allc.groupby(["dataset", "severity"]):
        print(f"\n   {dataset} severity {severity}: M2 savings {base.loc[base.index.get_level_values('dataset') == dataset].savings.mean():.4f} "
              f"(both severities); M2 above eps {g[g.config == 'ms_cost'].above_m2.mean():.3f}")
        for config in order:
            c = g[g.config == config]
            if not len(c):
                continue
            lo, hi = boot(table(c, "gain"))
            print(f"      {config:26s} {c.gain.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  positive {(c.gain > 0).mean():.2f}  "
                  f"above eps {c.above.mean():.3f}  cases {len(c)}")

    print("\n2c. Gain over M2 at severity 3 by regime")
    print(allc[allc.severity == 3].groupby(["dataset", "regime", "config"]).gain.mean().unstack("config")[order]
          .round(4).to_string())

    print("\n2d. Gain of hist_adapt_cost at severity 3 by family, and by seed")
    h = allc[(allc.config == "hist_adapt_cost") & (allc.severity == 3)]
    print(h.assign(family=h.corruption.map(FAMILY_OF)).groupby(["dataset", "family"]).gain.mean().unstack("family")
          .round(4).to_string())
    print(h.groupby(["dataset", "seed"]).gain.mean().unstack("seed").round(4).to_string())

    print("\n2e. Payback at severity 3 of the adapt scores (4,000 audited rows, training negligible)")
    for (dataset, config), g in allc[(allc.severity == 3) & allc.config.isin(
            ["cur_adapt_cost", "hist_adapt_cost"])].groupby(["dataset", "config"]):
        c9 = fractions_of(dataset)[-1]
        s_m2 = base[(base.index.get_level_values("dataset") == dataset)
                    & (base.index.get_level_values("severity") == 3)].savings.mean()
        gain = g.gain.mean()
        print(f"   {dataset} {config}: gain {gain:+.4f}, payback "
              f"{4000 * (c9 - (1 - s_m2)) / gain if gain > 0 else np.inf:,.0f} inputs")

    print("\n2f. AUROC for the agreement, mean over exits 1-8 and by exit, by dataset x severity")
    a = s[~s.config.str.endswith("_cost")].groupby(["dataset", "severity", "config"])[
        [f"auroc_{l}" for l in range(1, 9)]].mean()
    a["mean_1_8"] = a.mean(axis=1)
    print(a.round(3).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_columns", 30)
    part1()
    part2()
