# Tables and figures of the savings under shift, from outputs/savings_under_shift/results.csv
# (written by scripts/5_savings_under_shift/savings_under_shift.py).
# Conventions:
#   - savings of the calibrated methods (clean, M2, M8a, M8b) are their mean over splits: their risk is below eps
#     with probability 1 - delta. M1 keeps the clean lambda, so its savings count only where its risk is below eps
#     ("valid savings", 0 otherwise); its raw savings are the illusory savings of the project document.
#   - averages over the 15 standard corruptions; seeds averaged where all three are there (M8b: seed 0 only).
#   - recovery = (S_M8b - S_M2) / (S_clean - S_M2), on seed 0 and eps = 0.05, per (corruption, severity), with the
#     max softmax score (heads adapted, score not re-fitted) and with the agreement score (both re-fitted); the week 4
#     criterion asks recovery >= 1/4 on at least half of the (corruption, severity >= 3) cases.
#
# Usage: python scripts/5_savings_under_shift/savings_figures.py [--results outputs/savings_under_shift/results.csv]

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eesavings.savings import RESNET20_EXIT_MACS, RESNET20_FULL_MACS, cost_fractions
from eesavings.scores import SCORES
from eesavings.train import REGIMES

EPS = 0.05
DATASETS = ["cifar10", "cifar100"]
# categorical colors in fixed order (reference palette of the dataviz guidelines)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e5e4df"
METHOD_LABELS = {"M1": "M1 clean λ kept (valid savings only)", "M2": "M2 λ recalibrated on the shifted data",
                 "M8b": "M8b heads adapted, then λ recalibrated (M2 + gain measured on seed 0)"}


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def savings_column(df):
    # calibrated methods: mean savings; M1: valid savings
    return np.where(df["method"] == "M1", df["valid_savings"], df["savings"])


def curves(df, score, methods):
    # mean savings against severity (0 = clean) for every (dataset, regime, method), over all seeds.
    # M8b exists only for seed 0 and 3 splits, so its curve is the M2 curve plus the gain of M8b over M2 measured
    # on seed 0 and on the same splits (a paired difference, not two numbers from different networks).
    d = df[(df.eps == EPS) & (df.score == score)].copy()
    d["value"] = savings_column(d)
    clean = d[d.method == "clean"].groupby(["dataset", "regime", "seed"]).value.mean()
    rows = []
    for method in [m for m in methods if m != "M8b"]:
        m = d[d.method == method].groupby(["dataset", "regime", "seed", "severity"]).value.mean().reset_index()
        for (dataset, regime, seed), g in m.groupby(["dataset", "regime", "seed"]):
            rows.append({"dataset": dataset, "regime": regime, "seed": seed, "method": method, "severity": 0,
                         "value": clean[dataset, regime, seed]})
            rows += [{"dataset": dataset, "regime": regime, "seed": seed, "method": method, "severity": s,
                      "value": v} for s, v in zip(g.severity, g.value)]
    out = pd.DataFrame(rows).groupby(["dataset", "regime", "method", "severity"]).value.mean().reset_index()
    if "M8b" in methods:
        gain = paired_gain(df, score)
        m2 = out[out.method == "M2"].merge(gain, on=["dataset", "regime", "severity"], how="left")
        m2["value"] = m2.value + m2.gain.fillna(0)  # no gain at severity 0, where nothing is adapted
        out = pd.concat([out, m2.drop(columns="gain").assign(method="M8b")], ignore_index=True)
    return out


def paired_gain(df, score):
    # mean of M8b - M2 on seed 0 and on the splits of M8b, by (dataset, regime, severity)
    d = df[(df.eps == EPS) & (df.score == score) & (df.seed == 0)]
    d = d[d.split.isin(d[d.method == "M8b"].split.unique())]
    key = ["dataset", "regime", "corruption", "severity", "split"]
    m8b = d[d.method == "M8b"].set_index(key).savings
    m2 = d[d.method == "M2"].set_index(key).savings
    gain = (m8b - m2.reindex(m8b.index)).reset_index(name="gain")
    return gain.groupby(["dataset", "regime", "severity"]).gain.mean().reset_index()


def grid_plot(table, series, labels, title, path, ylabel="savings at ε = 0.05"):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True, sharey="row")
    for i, dataset in enumerate(DATASETS):
        for j, regime in enumerate(REGIMES):
            ax = axes[i, j]
            style(ax)
            for k, s in enumerate(series):
                d = table[(table.dataset == dataset) & (table.regime == regime) & (table.key == s)]
                ax.plot(d.severity, d.value, color=COLORS[k], lw=2, marker="o", ms=4, label=labels[s])
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(f"{dataset.upper()} · {regime}", fontsize=9, color=TEXT)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8, color=TEXT)
            if i == 1:
                ax.set_xlabel("severity (0 = clean)", fontsize=8, color=TEXT)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(series), frameon=False,
               fontsize=8)
    fig.suptitle(title, fontsize=10, color=TEXT, y=1.07)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def decomposition(df, score="max_softmax", severities=(3, 4, 5)):
    # contribution of every exit to S_clean - S_M2, split into the shift at the clean lambda (clean -> M1) and
    # the change of lambda (M1 -> M2), from the exit distributions on the same eval rows of each split
    pi = [f"pi_{l}" for l in range(1, 10)]
    run = ["dataset", "regime", "seed", "split"]
    d = df[(df.eps == EPS) & (df.score == score)]
    clean = d[d.method == "clean"][run + pi].rename(columns={c: c + "_clean" for c in pi})
    m1 = d[(d.method == "M1") & d.severity.isin(severities)][run + ["corruption", "severity"] + pi]
    m2 = d[(d.method == "M2") & d.severity.isin(severities)][run + ["corruption", "severity"] + pi]
    m = m2.merge(m1, on=run + ["corruption", "severity"], suffixes=("_m2", "_m1")).merge(clean, on=run)
    rows = []
    for dataset, g in m.groupby("dataset"):
        f = cost_fractions(RESNET20_EXIT_MACS[dataset], RESNET20_FULL_MACS[dataset])
        p_clean, p_m1, p_m2 = (g[[c + suffix for c in pi]].to_numpy() for suffix in ["_clean", "_m1", "_m2"])
        for part, (a, b) in {"shift at the clean λ": (p_clean, p_m1), "change of λ": (p_m1, p_m2)}.items():
            contrib = (a - b) * (1 - f)  # savings_change_by_exit, one row per case
            for l in range(9):
                rows.append(pd.DataFrame({"dataset": dataset, "regime": g.regime.to_numpy(), "part": part,
                                          "exit": l + 1, "drop": contrib[:, l]}))
    return pd.concat(rows).groupby(["dataset", "regime", "part", "exit"]).drop.mean().reset_index()


def recovery(df, score, min_drop=0.01):
    # seed 0, eps 0.05, on the splits of M8b: per (dataset, regime, corruption, severity) drop = S_clean - S_M2
    # and recovery = (S_M8b - S_M2) / drop; cases with drop < min_drop have nothing to recover
    d = df[(df.seed == 0) & (df.score == score) & (df.eps == EPS)]
    d = d[d.split.isin(d[d.method == "M8b"].split.unique())]
    clean = d[d.method == "clean"].groupby(["dataset", "regime"]).savings.mean()
    m = d[d.method.isin(["M2", "M8b"])].groupby(["dataset", "regime", "corruption", "severity", "method"])
    m = m.savings.mean().unstack("method").reset_index()
    m["clean"] = [clean[a, b] for a, b in zip(m.dataset, m.regime)]
    m["drop"] = m.clean - m.M2
    m["recovery_M8b"] = np.where(m["drop"] >= min_drop, (m.M8b - m.M2) / m["drop"], np.nan)
    return m.assign(score=score)


def adaptation_test(rec):
    # criterion of the project document, on the (corruption, severity >= 3) cases with something to recover
    hard = rec[(rec.severity >= 3) & rec.recovery_M8b.notna()]
    rows = []
    for (dataset, regime), g in hard.groupby(["dataset", "regime"]):
        rows.append({"score": g.score.iloc[0], "dataset": dataset, "regime": regime, "cases": len(g),
                     "drop": g["drop"].mean(), "gain_M8b": (g.M8b - g.M2).mean(),
                     "median_recovery_M8b": g.recovery_M8b.median(),
                     "share_recovering_a_quarter": (g.recovery_M8b >= 0.25).mean()})
    table = pd.DataFrame(rows)
    table["passes"] = table.share_recovering_a_quarter >= 0.5
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/savings_under_shift/results.csv")
    args = parser.parse_args()
    out = os.path.dirname(args.results)
    df = pd.read_csv(args.results, dtype={"l2": str})
    pd.set_option("display.width", 220)

    # 1) the answer to the first question: savings against severity, for every method (max softmax, as in
    #    Fast yet Safe)
    c = curves(df, "max_softmax", ["M1", "M2", "M8b"]).rename(columns={"method": "key"})
    grid_plot(c, ["M1", "M2", "M8b"], METHOD_LABELS,
              "Savings at ε = 0.05 under shift, max softmax score (mean over the 15 standard corruptions)",
              f"{out}/savings_by_method.png")
    c.to_csv(f"{out}/savings_by_method.csv", index=False)

    # 2) exit scores (H3): best savings without adaptation (M2) for every score
    s = pd.concat([curves(df, score, ["M2"]).assign(key=score) for score in SCORES])
    grid_plot(s, SCORES, {k: k for k in SCORES},
              "Savings at ε = 0.05 with λ recalibrated on the shifted data (M2), by exit score",
              f"{out}/savings_by_score.png")
    s.to_csv(f"{out}/savings_by_score.csv", index=False)

    # 3) where the savings go (H1): drop decomposed over the exits, severity 3 to 5, max softmax
    dec = decomposition(df)
    dec.to_csv(f"{out}/savings_drop_by_exit.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True, sharey="row")
    for i, dataset in enumerate(DATASETS):
        for j, regime in enumerate(REGIMES):
            ax = axes[i, j]
            style(ax)
            for k, part in enumerate(["shift at the clean λ", "change of λ"]):
                d = dec[(dec.dataset == dataset) & (dec.regime == regime) & (dec.part == part)]
                ax.bar(d.exit + (k - 0.5) * 0.38, d["drop"], width=0.36, color=COLORS[k], label=part)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xticks(range(1, 10))
            ax.set_title(f"{dataset.upper()} · {regime}", fontsize=9, color=TEXT)
            if j == 0:
                ax.set_ylabel("savings lost at this exit", fontsize=8, color=TEXT)
            if i == 1:
                ax.set_xlabel("exit", fontsize=8, color=TEXT)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Where the savings go: drop from clean to M2 by exit, severity 3 to 5, max softmax", fontsize=10,
                 color=TEXT, y=1.07)
    fig.tight_layout()
    fig.savefig(f"{out}/savings_drop_by_exit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4) the week 4 test on the adaptation (seed 0)
    rec = pd.concat([recovery(df, score) for score in ["max_softmax", "agreement"]])
    rec.to_csv(f"{out}/adaptation_recovery.csv", index=False)
    gate = pd.concat([adaptation_test(rec[rec.score == score]) for score in ["max_softmax", "agreement"]])
    gate.to_csv(f"{out}/adaptation_test.csv", index=False)
    m8a = df[(df.eps == EPS) & (df.score == "agreement") & df.method.isin(["M2", "M8a"]) & (df.severity >= 3)]
    m8a = m8a.groupby(["dataset", "method"]).savings.mean().unstack()

    # printed tables, rounded, for the README
    table = c[c.severity.isin([0, 1, 3, 5])].pivot_table(index=["dataset", "regime"], columns=["key", "severity"],
                                                          values="value")
    print("\nSavings at eps = 0.05 (max softmax) by method and severity:\n", table.round(3).to_string())
    m2 = s[s.severity.isin([0, 3, 5])].pivot_table(index=["dataset", "regime"], columns=["key", "severity"],
                                                   values="value")
    print("\nM2 savings by exit score:\n", m2.round(3).to_string())
    risk = df[(df.eps == EPS) & (df.method == "M1")].groupby(["dataset", "score", "severity"]).risk.mean().unstack()
    print("\nM1 risk (clean lambda kept), mean over regimes and seeds:\n", risk.round(3).to_string())
    groups = dec.assign(group=pd.cut(dec.exit, [0, 3, 6, 9], labels=["exits 1-3", "exits 4-6", "exits 7-9"]))
    print("\nDrop by group of exits (severity 3-5, max softmax):\n",
          groups.groupby(["dataset", "regime", "group"], observed=True)["drop"].sum().unstack().round(3).to_string())
    print("\nWeek 4 test on the adaptation (seed 0, severity >= 3):\n", gate.round(3).to_string(index=False))
    print("\nRe-fitting only the agreement score (M8a) against M2, severity >= 3:\n", m8a.round(3).to_string())
    l2 = df[(df.method == "M8b") & (df.score == "max_softmax")].l2.str.split(" ", expand=True).astype(float)
    print("\nL2 strength chosen for the adapted heads (share of heads, by exit):\n",
          l2.apply(lambda col: col.value_counts(normalize=True)).round(2).to_string())
    print("\nsaved in", out)


if __name__ == "__main__":
    main()
