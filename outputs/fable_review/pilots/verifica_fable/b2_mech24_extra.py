# Two checks on B2.03 (15 November 2026), with separate code, before writing the result:
#   1. the guarantee: share of cases above eps and above eps + 2 standard errors (eval rows: se = sqrt(eps (1 - eps) /
#      3000)) for logit_diag, M2 and M8b on the same cases, and the paired difference diag - M2 of the share above eps
#      with a two-way bootstrap over networks and corruptions (the method of fase_a/m8b_violations.py)
#   2. absolute gains by regime, and the temperature per head against the fair gain of the 8 thresholds per exit of
#      pilot B1 (exit_path_cal - shared_cal_fine), regime by regime at severity 3
# It only prints. Usage (from the repository root): python outputs/fable_review/pilots/verifica_fable/b2_mech24_extra.py

import glob

import numpy as np
import pandas as pd

B2 = "outputs/fable_review/pilots/B2_adattamento"
B1 = "outputs/fable_review/pilots/B1_bilancio/b1_levels.csv"
RES = "outputs/savings_under_shift_v2/results.csv"
KEY = ["dataset", "regime", "seed", "corruption", "severity", "split"]
EPS = 0.05
SE = np.sqrt(EPS * (1 - EPS) / 3000)


def two_way(cells_a, cells_b, reps=2000, seed=0):
    # mean of (a - b) with networks and corruptions resampled together; tables networks x corruptions
    a, b = cells_a.to_numpy(), cells_b.reindex(index=cells_a.index, columns=cells_a.columns).to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        i, j = rng.integers(a.shape[0], size=a.shape[0]), rng.integers(a.shape[1], size=a.shape[1])
        out[r] = np.nanmean(a[np.ix_(i, j)] - b[np.ix_(i, j)])
    return np.nanmean(a - b), *np.percentile(out, [2.5, 97.5])


def cells(df, col):
    d = df.assign(network=df.dataset + "/" + df.regime + "/" + df.seed.astype(str))
    return d.groupby(["network", "corruption"])[col].mean().unstack("corruption")


def main():
    pd.set_option("display.width", 220)
    new = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{B2}/runs_mech24/*.csv"))], ignore_index=True)
    res = pd.read_csv(RES, usecols=KEY + ["score", "eps", "method", "risk", "savings"])
    res = res[(res.score == "max_softmax") & np.isclose(res.eps, EPS) & res.method.isin(["M2", "M8b"])
              & res.split.isin([0, 1, 2]) & res.severity.isin([3, 5])]
    wide = res.pivot_table(index=KEY, columns="method", values=["risk", "savings"])
    wide.columns = [f"{v}_{k}" for v, k in wide.columns]
    d = new[new.config.isin(["logit_diag_n4000", "logit_diag_n1000", "logit_bias_n4000", "logit_temp_n4000"])]
    d = d.merge(wide.reset_index(), on=KEY)

    print(f"1. Guarantee (eps {EPS}, eps + 2 se = {EPS + 2 * SE:.4f}); shares over the cases, paired with M2 and M8b")
    for (dataset, severity), g in d.groupby(["dataset", "severity"]):
        ref = g[g.config == "logit_diag_n4000"]
        print(f"   {dataset} severity {severity} ({len(ref)} cases): M2 above eps {(ref.risk_M2 > EPS).mean():.3f}, "
              f"above eps + 2se {(ref.risk_M2 > EPS + 2 * SE).mean():.3f}; M8b {(ref.risk_M8b > EPS).mean():.3f}, "
              f"{(ref.risk_M8b > EPS + 2 * SE).mean():.3f}; mean risk M2 {ref.risk_M2.mean():.4f}")
        for config, c in g.groupby("config"):
            c = c.assign(viol=(c.risk > EPS).astype(float), viol_m2=(c.risk_M2 > EPS).astype(float))
            m, lo, hi = two_way(cells(c, "viol"), cells(c, "viol_m2"))
            print(f"      {config:17s} above eps {(c.risk > EPS).mean():.3f}, above eps + 2se "
                  f"{(c.risk > EPS + 2 * SE).mean():.3f}, mean risk {c.risk.mean():.4f}, worst {c.risk.max():.4f}; "
                  f"difference with M2 {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n2a. Absolute gains over M2 by regime (mean over seeds, corruptions, splits)")
    d["gain"], d["gain_m8b"] = d.savings - d.savings_M2, d.savings_M8b - d.savings_M2
    t = d.groupby(["dataset", "severity", "regime", "config"]).gain.mean().unstack("config")
    t["M8b"] = d[d.config == "logit_diag_n4000"].groupby(["dataset", "severity", "regime"]).gain_m8b.mean()
    print(t.round(4).to_string())

    print("\n2b. Severity 3: temperature per head (B2.03, splits 0-2) against the fair gain of the 8 thresholds per exit")
    print("    (B1: exit_path_cal - shared_cal_fine, splits 0-4), by setting")
    b1 = pd.read_csv(B1, usecols=KEY + ["level", "savings"])
    b1 = b1[(b1.severity == 3) & b1.level.isin(["exit_path_cal", "shared_cal_fine"])]
    b1 = b1.pivot_table(index=KEY, columns="level", values="savings")
    fair = (b1.exit_path_cal - b1.shared_cal_fine).groupby(["dataset", "regime"]).mean()
    temp = d[(d.config == "logit_temp_n4000") & (d.severity == 3)].groupby(["dataset", "regime"]).gain.mean()
    diag = d[(d.config == "logit_diag_n4000") & (d.severity == 3)].groupby(["dataset", "regime"]).gain.mean()
    out = pd.DataFrame({"thresholds_per_exit_B1": fair, "temp_B203": temp, "diag_B203": diag})
    print(out.round(4).to_string())
    print("   means:", out.groupby(level=0).mean().round(4).to_dict())


if __name__ == "__main__":
    main()
