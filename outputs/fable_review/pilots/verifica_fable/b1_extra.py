# Fable's reading of pilot B1 (14 November 2026), computed from b1_levels.csv and b1_exits.csv with separate code (it
# does not use b1_summary.py):
#   1. the levels by dataset x severity, and the share of the clean savings left under shift, level by level
#   2. the fair gain of the thresholds per exit (exit_path_cal - shared_cal_fine) with a two-way bootstrap interval
#      (networks and corruptions resampled together), by dataset x severity and by setting
#   3. the drop of the savings from clean to shifted for M2 and for the input oracle, with the same interval
#   4. the input oracle by regime
#   5. where the inputs could leave: the distribution of the first agreeing exit
# It only prints. Usage (from the repository root): python outputs/fable_review/pilots/verifica_fable/b1_extra.py

import numpy as np
import pandas as pd

B1 = "outputs/fable_review/pilots/B1_bilancio"
CASE = ["dataset", "regime", "seed", "corruption", "severity", "split"]
REPS = 2000


def two_way_bootstrap(cells, reps=REPS, seed=0):
    # cells: Series indexed by (network, corruption), one value per cell (already a mean over the splits).
    # Networks and corruptions are resampled with replacement, both; returns the 2.5 and 97.5 percentiles of the mean
    table = cells.unstack("corruption")
    values = table.to_numpy()
    rng = np.random.default_rng(seed)
    n_net, n_cor = values.shape
    means = np.empty(reps)
    for r in range(reps):
        rows = rng.integers(n_net, size=n_net)
        cols = rng.integers(n_cor, size=n_cor)
        means[r] = np.nanmean(values[np.ix_(rows, cols)])
    return np.percentile(means, [2.5, 97.5])


def cells_of(series):
    # mean over the splits, one value per (network, corruption)
    s = series.reset_index()
    s["network"] = s.dataset + "/" + s.regime + "/" + s.seed.astype(str)
    return s.groupby(["network", "corruption"])[series.name].mean()


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    lv = pd.read_csv(f"{B1}/b1_levels.csv")
    w = lv.set_index(CASE + ["level"]).savings.unstack("level")
    risk = lv.set_index(CASE + ["level"]).risk.unstack("level")
    acc = lv.set_index(CASE + ["level"]).acc_system.unstack("level")
    acc9 = lv[lv.level == "ucb_3000"].set_index(CASE).acc_exit9

    print("1. Mean savings by dataset x severity, and the share of the clean savings left (ratio of the means)")
    levels = ["oracle_input_eps", "oracle_input", "exit_path_cal", "shared_cal_fine", "oracle_shared", "ucb_3000",
              "ucb_250", "acc_rel", "acc_abs"]
    m = w[levels].groupby(["dataset", "severity"]).mean()
    for dataset in ["cifar10", "cifar100"]:
        t = m.loc[dataset].T
        for s in (1, 3, 5):
            t[f"left_{s}"] = t[s] / t[0]
        print(dataset, "\n", t.round(3).to_string())

    print("\n2. Fair gain of the thresholds per exit: exit_path_cal - shared_cal_fine on eval, with the risk of the two")
    fair = (w.exit_path_cal - w.shared_cal_fine).rename("fair")
    for dataset in ["cifar10", "cifar100"]:
        for severity in (0, 1, 3, 5):
            sel = fair.xs(dataset, level="dataset", drop_level=False).xs(severity, level="severity", drop_level=False)
            lo, hi = two_way_bootstrap(cells_of(sel))
            r1 = risk.exit_path_cal.loc[sel.index].mean()
            r2 = risk.shared_cal_fine.loc[sel.index].mean()
            per_seed = sel.groupby("seed").mean()
            print(f"   {dataset} severity {severity}: {sel.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; by seed "
                  f"{per_seed.round(4).tolist()}; risk on eval {r1:.4f} against {r2:.4f}; share of cases with a "
                  f"positive gain {np.mean(sel > 0):.2f}")
    by_setting = fair.groupby(["dataset", "regime", "severity"]).mean().unstack("severity")
    print("   by setting:\n", by_setting.round(4).to_string())
    print("   settings at 0.02 or above:", (by_setting >= 0.02).sum().to_dict(), "of", len(by_setting))
    seeds = fair.groupby(["dataset", "regime", "seed", "severity"]).mean().unstack("severity")
    print("   per seed, settings at 0.02 or above:\n", (seeds >= 0.02).groupby("seed").sum().to_string())
    print("   sd between seeds of the setting means (mean over the 8 settings):",
          seeds.groupby(["dataset", "regime"]).std().mean().round(4).to_dict())

    print("\n3. Drop of the savings from clean to shifted (same network), M2 against the input oracle")
    clean = w.xs(0, level="severity").droplevel("corruption")  # (dataset, regime, seed, split)
    for level in ["ucb_3000", "oracle_input", "oracle_input_eps", "exit_path_cal"]:
        base = clean[level]
        for dataset in ["cifar10", "cifar100"]:
            out = []
            for severity in (3, 5):
                sh = w[level].xs(dataset, level="dataset", drop_level=False).xs(severity, level="severity")
                b = base.reindex(sh.droplevel("corruption").index).to_numpy()
                drop = pd.Series(b - sh.to_numpy(), index=sh.index, name="drop")
                d = drop.reset_index()
                d["network"] = d.dataset + "/" + d.regime + "/" + d.seed.astype(str)
                lo, hi = two_way_bootstrap(d.groupby(["network", "corruption"])["drop"].mean())
                out.append(f"severity {severity}: clean {b.mean():.3f}, drop {drop.mean():.3f} [{lo:.3f}, {hi:.3f}], "
                           f"left {1 - drop.mean() / b.mean():.3f}")
            print(f"   {level:17s} {dataset}: " + "; ".join(out))

    print("\n4. The input oracle (no disagreement) and M2 by regime")
    t = w[["oracle_input", "ucb_3000"]].groupby(["dataset", "regime", "severity"]).mean().unstack("severity")
    print(t.round(3).to_string())

    print("\n5. Accuracy of the system at M2 minus accuracy of exit 9 (eval rows), by dataset x severity")
    gap = (acc.ucb_3000 - acc9).groupby(["dataset", "severity"]).mean().unstack("severity")
    print(gap.round(4).to_string())

    ex = pd.read_csv(f"{B1}/b1_exits.csv")
    print("\n6. Share of inputs whose first agreeing exit is exit l (all rows), by dataset x severity")
    pi = ex.groupby(["dataset", "severity", "exit"]).oracle_pi.mean().unstack("exit")
    print(pi.round(3).to_string())
    print("   accuracy of the exits (mean over networks and corruptions)")
    print(ex.groupby(["dataset", "severity", "exit"]).acc.mean().unstack("exit").round(3).to_string())
    print("   agreement of the exits with exit 9")
    print(ex.groupby(["dataset", "severity", "exit"]).agree.mean().unstack("exit").round(3).to_string())


if __name__ == "__main__":
    main()
