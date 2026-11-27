# Summary of pilot B1 (the compute budget of the savings under shift), from b1_levels.csv written by b1_levels.py:
#   checks        ucb_3000 against outputs/savings_under_shift/results.csv; ucb_1000, ucb_250 and crc_* against
#                 outputs/block3_review/calibration_budget.csv; oracle_shared - ucb_3000 next to price_fst of
#                 m2_validity_oracles.csv; the invariants of the rule with 8 thresholds and of the input oracles
#   levels        mean savings, risk and accuracy of every level by dataset x severity (and x regime)
#                 -> b1_table_levels.csv
#   margins       the gaps between the levels, also by regime, by corruption family and by seed
#                 -> b1_table_margins.csv
#   net savings   savings of M2 net of the calibration forwards, for 3 horizons H -> b1_net_savings.csv
#   accuracy      the levels chosen on the accuracy lost against the levels chosen on the risk
#   exit 9        how well the accuracy of exit 9 predicts the savings of ucb_3000, network by network
#   figure        b1_levels.png
# With a subset of networks or domains everything runs on the rows present. What it prints also goes to
# <dir>/b1_summary.log. Specification: SPECIFICA.md in this folder.
#
# Usage (from the root of the repository):
#   python outputs/fable_review/pilots/B1_bilancio/b1_summary.py [--dir DIR]

import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "scripts", "6_block3_review"))

from eesavings.analysis import LAMBDAS  # noqa: E402
from review_common import FAMILY_OF, RESULTS, fractions_of, start_log  # noqa: E402

CALIB = "outputs/block3_review/calibration_budget.csv"
ORACLES = "outputs/block3_review/m2_validity_oracles.csv"
EPS, DELTA = 0.05, 0.1
TOL = 1e-9
CASE = ["dataset", "regime", "seed", "corruption", "severity", "split"]
NET = ["dataset", "regime", "seed"]
LEVELS = ["oracle_input_eps", "oracle_input", "oracle_exit_thresholds", "oracle_shared", "exit_path_cal",
          "shared_cal_fine", "shared_cal", "ucb_3000", "ucb_1000", "ucb_250", "crc_3000", "crc_250", "crc_100", "acc_rel",
          "acc_abs"]
# exit_vs_shared_fair (added by Fable): the 8 thresholds against the shared one when both choose ONE number on cal
# and both fill the budget there (exit_path_cal - shared_cal_fine); exit_vs_shared compares rules whose risk on eval
# is different (see b1_levels.py)
MARGINS = ["total", "input_vs_exit", "exit_vs_shared", "price", "input_eps_vs_input", "input_vs_shared",
           "exit_vs_shared_fair"]
PARTS = ["input_vs_exit", "exit_vs_shared", "price"]  # they add up to total
HORIZONS = [10_000, 100_000, 1_000_000]
# the figure shows the fair pair (exit_path_cal, shared_cal_fine), not oracle_exit_thresholds and oracle_shared, whose
# risks on eval differ (Fable, after the full run)
FIGURE = [("oracle_input_eps", "-"), ("oracle_input", "-"), ("exit_path_cal", "-"), ("shared_cal_fine", "-"),
          ("ucb_3000", "-"), ("ucb_250", "-"), ("acc_rel", "--"), ("acc_abs", "--")]


def show(title, table):
    print(f"\n{title}\n{table.round(4).to_string()}")


def wide_savings(lv):
    # one row per case (network, domain, split), one column per level
    return lv.set_index(CASE + ["level"]).savings.unstack("level")


# ---------------- checks ----------------

def compare(ours, ref, name):
    # lambda, risk and savings of our rows against the reference rows of the same cases; True if they agree
    m = ours.merge(ref[CASE + ["lambda", "risk", "savings"]], on=CASE, how="left", suffixes=("", "_ref"))
    assert len(m) == len(ours), "more than one reference row for a case"
    missing = int(m.savings_ref.isna().sum())
    # the reference risks are float32 means of the float32 losses: compared at float32 precision
    risk32 = (m.risk.astype(np.float32).astype(float) - m.risk_ref.astype(np.float32).astype(float)).abs().max()
    diffs = {"lambda": (m.lam - m["lambda"]).abs().max(), "risk": risk32,
             "savings": (m.savings - m.savings_ref).abs().max()}
    print(f"   {name}: {len(m) - missing} rows compared, {missing} without a reference row; largest difference "
          f"lambda {diffs['lambda']:.1e}, risk {diffs['risk']:.1e} (in float64 {(m.risk - m.risk_ref).abs().max():.1e})"
          f", savings {diffs['savings']:.1e}")
    if missing > 0 or len(m) == 0 or not all(d <= TOL for d in diffs.values()):
        print(f"   CHECK FAILED: {name}: {missing} rows without a reference, differences {diffs}")


def check_calibration(lv):
    print("(a) ucb_3000 against results.csv (max_softmax, eps 0.05, clean at severity 0, M2 on the corruptions)")
    res = pd.read_csv(RESULTS, usecols=CASE + ["score", "eps", "method", "lambda", "risk", "savings"])
    res = res[(res.score == "max_softmax") & np.isclose(res.eps, EPS)
              & (((res.method == "clean") & (res.severity == 0)) | ((res.method == "M2") & (res.severity > 0)))]
    compare(lv[lv.level == "ucb_3000"], res, "ucb_3000")
    print("(b) ucb_1000, ucb_250 (delta 0.1, eps 0.05) and crc_* (eps 0.05) against calibration_budget.csv")
    cb = pd.read_csv(CALIB)
    for level in ["ucb_1000", "ucb_250", "crc_3000", "crc_250", "crc_100"]:
        method, n_cal = level.split("_")
        sel = (cb.method == method) & (cb.n_cal == int(n_cal)) & np.isclose(cb.eps, EPS)
        if method == "ucb":
            sel &= np.isclose(cb.delta, DELTA)
        compare(lv[lv.level == level], cb[sel], level)


def check_oracle_price(w):
    price = (w.oracle_shared - w.ucb_3000).groupby(["dataset", "severity"]).mean().rename("oracle_shared-ucb_3000")
    ref = pd.read_csv(ORACLES).set_index(["dataset", "severity"])[["price_fst", "price_best"]]
    show("(c) oracle_shared - ucb_3000 (oracle chosen on fit, measured on eval) next to price_fst and price_best of "
         "m2_validity_oracles.csv (oracle on the whole population: a different protocol)",
         pd.concat([price, ref], axis=1, join="inner"))


def check_invariants(lv):
    print("\n(d) invariants")
    oet = lv[lv.level == "oracle_exit_thresholds"].set_index(CASE)
    shared = lv[lv.level == "oracle_shared"].set_index(CASE).reindex(oet.index)
    bad_risk = int((oet.risk_fit > EPS + 1e-12).sum())
    bad_savings = int((oet.savings_fit < shared.savings_fit - 1e-12).sum())
    gain = (oet.savings_fit - shared.savings_fit).mean()
    print(f"   oracle_exit_thresholds on fit: {len(oet)} cases, {bad_risk} with mean loss above eps, {bad_savings} "
          f"with savings below oracle_shared (mean gain on fit {gain:.4f})")
    oi, oie = lv[lv.level == "oracle_input"], lv[lv.level == "oracle_input_eps"]
    bad_oi, bad_oie = int((oi.risk != 0).sum()), int((oie.risk > EPS + 1e-12).sum())
    print(f"   oracle_input: {len(oi)} cases, {bad_oi} with risk not 0; oracle_input_eps: {len(oie)} cases, {bad_oie} "
          f"with risk above 0.05 (rows moved: min {oie.n_moved.min():.0f}, max {oie.n_moved.max():.0f})")
    if bad_risk or bad_savings or bad_oi or bad_oie:
        print(f"   CHECK FAILED: invariants: {bad_risk} {bad_savings} {bad_oi} {bad_oie}")


# ---------------- levels ----------------

def level_stats(lv, by):
    d = lv.assign(above_eps=lv.risk > EPS)
    return d.groupby(by + ["level"]).agg(savings=("savings", "mean"), risk=("risk", "mean"),
                                        above_eps=("above_eps", "mean"), acc_system=("acc_system", "mean"),
                                        acc_exit9=("acc_exit9", "mean"), cases=("savings", "size"))


def levels_section(lv, out_dir):
    print("\n\n=== Levels: mean over networks, corruptions and splits, measured on eval ===")
    pooled = level_stats(lv, ["dataset", "severity"])
    for stat in ["savings", "risk", "above_eps", "acc_system", "acc_exit9"]:
        show(f"{stat} by dataset x severity", pooled[stat].unstack(["dataset", "severity"]).reindex(LEVELS))
    by_regime = level_stats(lv, ["dataset", "regime", "severity"])
    show("savings by dataset x regime x severity",
         by_regime.savings.unstack("severity").reindex(LEVELS, level="level"))
    table = pd.concat([pooled.reset_index().assign(regime="all"), by_regime.reset_index()], ignore_index=True)
    table[["dataset", "regime", "severity", "level", "savings", "risk", "above_eps", "acc_system", "acc_exit9",
           "cases"]].to_csv(os.path.join(out_dir, "b1_table_levels.csv"), index=False)


# ---------------- margins ----------------

def case_margins(w):
    m = pd.DataFrame(index=w.index)
    m["total"] = w.oracle_input_eps - w.ucb_3000
    m["input_vs_exit"] = w.oracle_input_eps - w.oracle_exit_thresholds
    m["exit_vs_shared"] = w.oracle_exit_thresholds - w.oracle_shared
    m["price"] = w.oracle_shared - w.ucb_3000
    m["input_eps_vs_input"] = w.oracle_input_eps - w.oracle_input
    m["input_vs_shared"] = w.oracle_input - w.oracle_shared
    m["exit_vs_shared_fair"] = w.exit_path_cal - w.shared_cal_fine
    return m.reset_index()


def with_shares(g):
    # share of total of the three parts, as a ratio of the means
    for part in PARTS:
        g[f"share_{part}"] = g[part] / g.total
    return g


def margin_table(m, by):
    # pooled means, means of every seed, and standard deviation between the seeds of the means of every seed
    pooled = with_shares(m.groupby(by)[MARGINS].mean()).assign(stat="mean")
    per_seed = with_shares(m.groupby(by + ["seed"])[MARGINS].mean())
    sd = per_seed.groupby(level=by).std(ddof=1).assign(stat="sd_seeds")
    seeds = per_seed.reset_index("seed")
    seeds["stat"] = "seed" + seeds.pop("seed").astype(str)
    return pd.concat([pooled, seeds, sd])


def margins_section(w, out_dir):
    print("\n\n=== Margins (savings on eval): total = oracle_input_eps - ucb_3000 = input_vs_exit + exit_vs_shared + "
          "price ===")
    m = case_margins(w)
    m["family"] = m.corruption.map(FAMILY_OF).fillna("clean")
    groupings = {"dataset_severity": (m, ["dataset", "severity"]),
                 "dataset_regime_severity": (m, ["dataset", "regime", "severity"]),
                 "dataset_family_severity3": (m[m.severity == 3], ["dataset", "family"])}
    tables = []
    for name, (d, by) in groupings.items():
        t = margin_table(d, by)
        show(f"{name}: mean", t[t.stat == "mean"].drop(columns="stat"))
        show(f"{name}: standard deviation between the seeds (of the means of every seed)",
             t[t.stat == "sd_seeds"].drop(columns="stat"))
        tables.append(t.reset_index().assign(grouping=name))
    show("dataset_severity: means of every seed", tables[0][tables[0].stat.str.startswith("seed")]
         .set_index(["dataset", "severity", "stat"])[MARGINS])
    pd.concat(tables, ignore_index=True).to_csv(os.path.join(out_dir, "b1_table_margins.csv"), index=False)

    for margin in ("exit_vs_shared", "exit_vs_shared_fair"):
        for severity in (0, 1, 3, 5):
            d = m[m.severity == severity]
            if d.empty:
                continue
            s = d.groupby(["dataset", "regime"])[margin].mean()
            print(f"\n{margin} at severity {severity} by setting (dataset x regime):\n{s.round(4).to_string()}\n"
                  f"   mean {s.mean():.4f}; settings at 0.02 or above: {(s >= 0.02).sum()} of {len(s)}")
            per_seed = d.groupby(["seed", "dataset", "regime"])[margin].mean()
            g = per_seed.groupby("seed")
            print("   per seed: mean", g.mean().round(4).to_dict(), "; settings at 0.02 or above",
                  (per_seed >= 0.02).groupby("seed").sum().to_dict(), "of", g.size().to_dict())


# ---------------- net savings of M2 ----------------

def net_savings_section(lv, out_dir):
    print("\n\n=== Net savings of M2: net = (n_cal * (1 - c9) + (H - n_cal) * S) / H, from calibration_budget.csv ===")
    cb = pd.read_csv(CALIB)
    sel = np.isclose(cb.eps, EPS) & (((cb.method == "ucb") & np.isclose(cb.delta, DELTA)) | (cb.method == "crc"))
    present = lv[["dataset", "regime", "seed", "corruption", "severity"]].drop_duplicates()
    cb = cb[sel & cb.severity.isin([0, 3, 5])].merge(present, on=list(present.columns))
    s = cb.groupby(["dataset", "severity", "method", "n_cal"]).savings.mean().rename("S").reset_index()
    s["c9"] = s.dataset.map(lambda d: fractions_of(d)[-1])
    net = pd.concat([s.assign(H=h, net=(s.n_cal * (1 - s.c9) + (h - s.n_cal) * s.S) / h) for h in HORIZONS],
                    ignore_index=True)
    table = net.pivot_table(index=["dataset", "severity", "method", "n_cal"], columns="H", values="net")
    show("net savings (columns: H); S is the mean savings on eval", table.join(s.set_index(table.index.names).S))
    best = net.loc[net.groupby(["dataset", "severity", "method", "H"]).net.idxmax()]
    show("best n_cal for every H", best.pivot_table(index=["dataset", "severity", "method"], columns="H",
                                                    values="n_cal"))
    net.to_csv(os.path.join(out_dir, "b1_net_savings.csv"), index=False)


# ---------------- accuracy lost ----------------

def accuracy_section(lv):
    print("\n\n=== Accuracy lost: levels chosen on the accuracy lost on fit against the levels chosen on the risk ===")
    d = lv[lv.level.isin(["acc_abs", "acc_rel", "ucb_3000", "oracle_shared"])]
    # everything leaves at exit 1 on fit when the savings on fit are those of exit 1. The ties go to the smallest k,
    # so in that case k is usually a few steps before the last of the grid: k == last alone undercounts
    exit1 = d.dataset.map(lambda dataset: 1.0 - fractions_of(dataset)[0])
    d = d.assign(acc_lost=d.acc_exit9 - d.acc_system, at_last=d.k == len(LAMBDAS) - 1,
                 all_exit1=(d.savings_fit - exit1).abs() <= 1e-12)
    g = d.groupby(["dataset", "level", "severity"]).agg(savings=("savings", "mean"), acc_lost=("acc_lost", "mean"),
                                                         risk=("risk", "mean"), at_last=("at_last", "mean"),
                                                         all_exit1=("all_exit1", "mean"))
    s = g.savings.unstack("severity")
    for severity in (3, 5):
        if severity in s and 0 in s:
            s[f"left_{severity}"] = s[severity] / s[0]
    show("savings by dataset x severity, and share of the clean savings left at severity 3 and 5", s)
    show("mean acc_exit9 - acc_system on eval", g.acc_lost.unstack("severity"))
    show("mean risk on eval", g.risk.unstack("severity"))
    acc = (slice(None), ["acc_abs", "acc_rel"])
    show("acc_abs and acc_rel: share of cases where everything leaves at exit 1 on fit (savings on fit = 1 - fr[0])",
         g.all_exit1.unstack("severity").loc[acc, :])
    show("acc_abs and acc_rel: share of cases where k is literally the last of the grid (lambda 0)",
         g.at_last.unstack("severity").loc[acc, :])


# ---------------- savings against the accuracy of exit 9 ----------------

def exit9_section(lv):
    print("\n\n=== Savings of ucb_3000 against the accuracy of exit 9, per network, on its shifted domains (mean over "
          "the splits) ===")
    u = lv[(lv.level == "ucb_3000") & (lv.severity > 0)]
    u = u.groupby(NET + ["corruption", "severity"])[["savings", "acc_exit9"]].mean().reset_index()
    rows = []
    for net, d in u.groupby(NET):
        if d.corruption.nunique() < 2:
            continue
        errors = []
        for corruption in d.corruption.unique():
            train, test = d[d.corruption != corruption], d[d.corruption == corruption]
            slope, intercept = np.polyfit(train.acc_exit9, train.savings, 1)
            errors += list((test.savings - (slope * test.acc_exit9 + intercept)).abs())
        rows.append({**dict(zip(NET, net)), "domains": len(d), "spearman": spearmanr(d.savings, d.acc_exit9)[0],
                     "loco_mae": np.mean(errors)})
    if not rows:
        print("   fewer than 2 corruptions: skipped")
        return
    st = pd.DataFrame(rows)
    show("per network (loco_mae: mean absolute error of savings ~ acc_exit9, leaving one corruption out)", st)
    show("over the networks", st[["spearman", "loco_mae"]].agg(["median", "min", "max"]))


# ---------------- figure ----------------

def figure(lv, out_dir):
    datasets = [d for d in ["cifar10", "cifar100"] if d in set(lv.dataset)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.2), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        s = lv[lv.dataset == dataset].groupby(["level", "severity"]).savings.mean()
        for level, style in FIGURE:
            if level in s.index.get_level_values("level"):
                ax.plot(s[level].index, s[level].values, style, marker="o", label=level)
        ax.set_title(dataset.upper())
        ax.set_xlabel("severity")
        ax.set_ylabel("savings (mean on eval)")
        ax.set_xticks([0, 1, 3, 5])
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "b1_levels.png"), dpi=150)
    print("\nsaved b1_levels.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=HERE)
    args = parser.parse_args()
    start_log("b1_summary", args.dir)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 500)
    lv = pd.read_csv(os.path.join(args.dir, "b1_levels.csv"))
    print(f"{len(lv)} rows: {lv[NET].drop_duplicates().shape[0]} networks, "
          f"{lv[['corruption', 'severity']].drop_duplicates().shape[0]} domains, {lv.split.nunique()} splits, "
          f"{lv.level.nunique()} levels")
    print("\n=== Checks ===")
    check_calibration(lv)
    w = wide_savings(lv)
    check_oracle_price(w)
    check_invariants(lv)
    sec = lv[lv.level == "oracle_exit_thresholds"].seconds
    print(f"   fit_exit_thresholds: {sec.mean():.3f} s per (case, split) on average, {sec.max():.3f} s at most")
    levels_section(lv, args.dir)
    margins_section(w, args.dir)
    net_savings_section(lv, args.dir)
    accuracy_section(lv)
    exit9_section(lv)
    figure(lv, args.dir)


if __name__ == "__main__":
    main()

