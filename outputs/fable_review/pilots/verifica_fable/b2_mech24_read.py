# Reading of the follow-up B2.03 (15 November 2026), with code separate from b2_summary.py. M2 and M8b come from
# outputs/savings_under_shift_v2/results.csv (max softmax, eps 0.05, splits 0, 1, 2), the maps on the logits from
# outputs/fable_review/pilots/B2_adattamento/runs_mech24/. Shares are ratios of mean gains over M2 on the same cases.
# Intervals: two-way bootstrap over networks and corruptions (cells averaged over the splits), 2,000 resamples.
#   0. check: the rows m2 of the job against the rows M2 of results.csv
#   1. part 1 by dataset x severity: gains, share of the gain of M8b kept, cases above eps
#   2. by regime and by seed (sd between seeds), and 750 rows against 3,000
#   3. compute, payback and D1 at severity 3
#   4. accuracy of the system at the cost of M2
#   5. part 2: BatchNorm statistics then maps, against BN alone and against the maps alone (seed 0, split 0)
# It only prints. Usage (from the repository root): python outputs/fable_review/pilots/verifica_fable/b2_mech24_read.py

import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/6_block3_review")
from review_common import fractions_of  # noqa: E402

B2 = "outputs/fable_review/pilots/B2_adattamento"
RES = "outputs/savings_under_shift_v2/results.csv"
COST = "outputs/block3_review/adaptation_cost.csv"
KEY = ["dataset", "regime", "seed", "corruption", "severity", "split"]
CONFIGS = ["logit_diag_n4000", "logit_diag_n1000", "logit_bias_n4000", "logit_bias_n1000", "logit_temp_n4000",
           "logit_temp_n1000"]
REPS = 2000


def read(sub):
    return pd.concat([pd.read_csv(f, dtype={"l2": str}) for f in sorted(glob.glob(f"{B2}/{sub}/*.csv"))],
                     ignore_index=True)


def cells(df, col):
    # mean over the splits, one value per (network, corruption): a table networks x corruptions
    d = df.assign(network=df.dataset + "/" + df.regime + "/" + df.seed.astype(str))
    return d.groupby(["network", "corruption"])[col].mean().unstack("corruption")


def bootstrap(num, den=None, reps=REPS, seed=0):
    # num, den: tables networks x corruptions; returns the 2.5 and 97.5 percentiles of mean(num) / mean(den)
    # (or of mean(num) when den is None), networks and corruptions resampled together
    a = num.to_numpy()
    b = None if den is None else den.reindex(index=num.index, columns=num.columns).to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        i, j = rng.integers(a.shape[0], size=a.shape[0]), rng.integers(a.shape[1], size=a.shape[1])
        m = np.nanmean(a[np.ix_(i, j)])
        out[r] = m if b is None else m / np.nanmean(b[np.ix_(i, j)])
    return np.percentile(out, [2.5, 97.5])


def main():
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 300)
    new = read("runs_mech24")
    res = pd.read_csv(RES, usecols=KEY + ["score", "eps", "method", "lambda", "risk", "savings"])
    res = res[(res.score == "max_softmax") & np.isclose(res.eps, 0.05) & res.method.isin(["M2", "M8b"])
              & res.split.isin([0, 1, 2]) & res.severity.isin([3, 5])]
    print(f"{len(new)} rows of B2.03; networks {new[['dataset', 'regime', 'seed']].drop_duplicates().shape[0]}; "
          f"cases with m2 {int((new.config == 'm2').sum())}")

    print("\n0. Check: m2 of the job against M2 of results.csv")
    m = new[new.config == "m2"].merge(res[res.method == "M2"], on=KEY, suffixes=("", "_ref"))
    diffs = {"lambda": np.abs(m["lambda"] - m.lambda_ref).max(),
             "risk (float32)": np.abs(m.risk.astype(np.float32) - m.risk_ref.astype(np.float32)).max(),
             "savings": np.abs(m.savings - m.savings_ref).max()}
    print(f"   {len(m)} rows compared, largest differences {diffs}"
          + ("" if max(diffs.values()) == 0 and len(m) == int((new.config == 'm2').sum()) else "  CHECK FAILED"))

    base = res.pivot_table(index=KEY, columns="method", values=["savings", "risk"])
    base.columns = [f"{v}_{k}" for v, k in base.columns]
    base = base.reset_index()
    p1 = new[new.config.isin(CONFIGS)].merge(base, on=KEY)
    p1["gain"] = p1.savings - p1.savings_M2
    p1["gain_m8b"] = p1.savings_M8b - p1.savings_M2
    m2rows = new[new.config == "m2"][KEY + ["acc_at_m2_cost", "acc_system"]].rename(
        columns={"acc_at_m2_cost": "acc_m2_at_cost", "acc_system": "acc_system_m2"})
    p1 = p1.merge(m2rows, on=KEY)

    print("\n1. Gains over M2 and share of the gain of M8b, by dataset x severity (24 networks, splits 0 to 2)")
    for (dataset, severity), g in p1.groupby(["dataset", "severity"]):
        ref = g[g.config == CONFIGS[0]]
        m8b = cells(ref, "gain_m8b")
        lo, hi = bootstrap(m8b)
        print(f"\n   {dataset} severity {severity}: M8b gain {ref.gain_m8b.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; "
              f"above eps: M2 {(ref.risk_M2 > 0.05).mean():.3f}, M8b {(ref.risk_M8b > 0.05).mean():.3f} "
              f"({len(ref)} cases)")
        for config in CONFIGS:
            c = g[g.config == config]
            num = cells(c, "gain")
            glo, ghi = bootstrap(num)
            slo, shi = bootstrap(num, cells(c, "gain_m8b"))
            print(f"      {config:17s} gain {c.gain.mean():+.4f} [{glo:+.4f}, {ghi:+.4f}]; share of M8b "
                  f"{c.gain.mean() / c.gain_m8b.mean():.2f} [{slo:.2f}, {shi:.2f}]; positive {(c.gain > 0).mean():.2f}; "
                  f"above eps {(c.risk > 0.05).mean():.3f}; T {c.train_fwd.mean():.2f}")

    print("\n2a. Share of the gain of M8b kept, by regime (and the gain of M8b)")
    t = p1.groupby(["dataset", "severity", "regime", "config"]).agg(g=("gain", "mean"), m=("gain_m8b", "mean"))
    share = (t.g / t.m).unstack("config")[CONFIGS]
    share["M8b_gain"] = t.m.groupby(["dataset", "severity", "regime"]).first()
    print(share.round(2).to_string())
    print("\n2b. By seed: share of M8b kept by logit_diag_n4000, and the gains")
    d = p1[p1.config == "logit_diag_n4000"]
    s = d.groupby(["dataset", "severity", "seed"]).agg(diag=("gain", "mean"), m8b=("gain_m8b", "mean"))
    s["share"] = s.diag / s.m8b
    print(s.round(4).unstack("seed").to_string())
    sd = d.groupby(["dataset", "severity", "regime", "seed"]).gain.mean().groupby(
        ["dataset", "severity", "regime"]).std()
    print("   sd between seeds of the gain of logit_diag_n4000 (per setting), mean over the regimes:",
          sd.groupby(["dataset", "severity"]).mean().round(4).to_dict(), "; largest:", round(sd.max(), 4))
    print("\n2c. 750 rows against 3,000: gain of the n1000 version over the gain of the n4000 version, by regime")
    t2 = p1.groupby(["dataset", "severity", "regime", "config"]).gain.mean().unstack("config")
    for kind in ["diag", "bias", "temp"]:
        t2[kind] = t2[f"logit_{kind}_n1000"] / t2[f"logit_{kind}_n4000"]
    print(t2[["diag", "bias", "temp"]].round(2).to_string())

    print("\n3. Compute, payback and D1 at severity 3 (M8b training: adaptation_cost.csv, seed 0, joint, 3 domains)")
    cost = pd.read_csv(COST).groupby("dataset").train_full_forwards.mean()
    for dataset, g in p1[p1.severity == 3].groupby("dataset"):
        c9 = fractions_of(dataset)[-1]
        s_m2 = g[g.config == CONFIGS[0]].savings_M2.mean()
        gain_m8b = g[g.config == CONFIGS[0]].gain_m8b.mean()
        t_m8b = cost[dataset]
        pay_m8b = (4000 * (c9 - (1 - s_m2)) + t_m8b) / gain_m8b
        print(f"   {dataset}: M8b training {t_m8b:,.0f} forwards, payback {pay_m8b:,.0f} inputs")
        for config in CONFIGS:
            c = g[g.config == config]
            gain, t, n = c.gain.mean(), c.train_fwd.mean(), c.n_rows_paid.mean()
            payback = (n * (c9 - (1 - s_m2)) + t) / gain if gain > 0 else np.inf
            share, ratio = gain / gain_m8b, t / t_m8b
            ok = share >= 0.7 and ratio <= 0.10 and payback < 30000
            print(f"      {config:17s} share {share:.2f}, training {t:.2f} ({ratio:.5f} of M8b), payback "
                  f"{payback:,.0f}: D1 conditions {'met' if ok else 'not met'}")

    print("\n4. System accuracy at the cost where M2 is certified, minus that of M2 (points)")
    p1["acc_gain"] = 100 * (p1.acc_at_m2_cost - p1.acc_m2_at_cost)
    print(p1.groupby(["dataset", "severity", "config"]).acc_gain.mean().unstack("config")[CONFIGS].round(2)
          .to_string())

    print("\n5. Part 2: BatchNorm statistics of the 4,000 adapt rows, then maps (seed 0, joint and joint_sd, split 0)")
    bn = read("runs_bn")
    hd = read("runs_heads")
    k5 = ["dataset", "regime", "corruption", "severity", "split"]
    m2 = new[(new.config == "m2") & (new.seed == 0) & (new.split == 0) & new.regime.isin(["joint", "joint_sd"])]
    m2 = m2[k5 + ["savings", "acc_system"]].rename(columns={"savings": "s_m2", "acc_system": "a_m2"})
    parts = [new[new.config.isin(["c_logit_bias_n4000", "c_logit_diag_n4000"])],
             new[new.config.isin(["logit_diag_n4000", "logit_bias_n4000"]) & (new.seed == 0) & (new.split == 0)
                 & new.regime.isin(["joint", "joint_sd"])],
             bn[(bn.config == "bn_n4000") & (bn.split == 0)],
             hd[hd.config.isin(["c_logit_grid_n4000", "logit_grid_n4000"]) & (hd.split == 0)]]
    p2 = pd.concat([p[k5 + ["config", "savings", "acc_system", "risk"]] for p in parts], ignore_index=True)
    p2 = p2.merge(m2, on=k5)
    p2["gain"], p2["acc"] = p2.savings - p2.s_m2, 100 * (p2.acc_system - p2.a_m2)
    out = p2.groupby(["dataset", "severity", "config"]).agg(gain=("gain", "mean"), acc_points=("acc", "mean"),
                                                            above_eps=("risk", lambda x: (x > 0.05).mean()),
                                                            cases=("gain", "size"))
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
