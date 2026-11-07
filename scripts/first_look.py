# First look at the trained networks: a check of the whole pipeline before the analysis of week 4.
#   1) accuracy of every exit on the clean test set and on CIFAR-C (15 standard corruptions, then the 4 extra);
#   2) how well two exit scores separate correct from wrong predictions on the clean test set (AUROC, exits 1..8):
#      the max softmax and the learned confidence unit (target "label");
#   3) Fast yet Safe with UCB (WSR), consistency risk, eps = 0.05, delta = 0.1, max softmax score.
#      On 100 random splits of the clean test set (5,000 calibration, 5,000 evaluation): lambda is chosen on the
#      clean calibration half and kept fixed; on the evaluation half we measure risk and savings on clean images
#      and on the same images of CIFAR-C, for each severity. The splits are the same for every network.
# The lambda grid is finer near 1 than the one of Fast yet Safe, because on CIFAR the top softmax is often > 0.99.
#
# Usage: python scripts/first_look.py [--out-dir outputs/cifar]   (after scripts/export_outputs.py)

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eesavings.data.cifar import CORRUPTIONS, EXTRA_CORRUPTIONS, SEVERITIES
from eesavings.risk.exits import conf_measures
from eesavings.fys import losses_and_exits
from eesavings.risk.calibration import ucb_lam
from eesavings.savings import RESNET20_EXIT_MACS, RESNET20_FULL_MACS, cost_fractions, savings_from_exits
from eesavings.train import REGIMES

LAMBDAS = np.concatenate([[1.0], 1 - np.logspace(-4, 0, 200)])  # descending, row 0 is lambda = 1 (no early exit)
EPS = 0.05
DELTA = 0.1
N_TRIALS = 100
N_CAL = 5000


def risk_and_savings(logits, labels, ks, splits, fractions):
    # mean consistency risk and savings on the evaluation half of every split, at the chosen lambda index
    losses, exits = losses_and_exits(logits, labels, LAMBDAS, "prediction-consistency")
    risk = [losses[k, p[N_CAL:]].mean() for k, p in zip(ks, splits)]
    savings = [savings_from_exits(exits[k, p[N_CAL:]], fractions) for k, p in zip(ks, splits)]
    return np.array(risk), np.array(savings)


def score_auroc(run_dir):
    # AUROC of max softmax and of the confidence unit for "the exit is correct", on the clean test set
    preds = os.path.join(run_dir, "preds")
    logits, labels = np.load(f"{preds}/test_logits.npy"), np.load(f"{preds}/test_labels.npy")
    conf_unit = np.load(f"{preds}/test_conf.npy")
    max_softmax = conf_measures(logits)
    correct = logits.argmax(axis=2) == labels[:, None]
    rows = []
    for l in range(logits.shape[1] - 1):
        rows.append({"exit": l + 1, "auroc_max_softmax": roc_auc_score(correct[:, l], max_softmax[:, l]),
                     "auroc_conf_unit": roc_auc_score(correct[:, l], conf_unit[:, l])})
    return rows


def analyse_run(run_dir, dataset, splits):
    preds = os.path.join(run_dir, "preds")
    fractions = cost_fractions(RESNET20_EXIT_MACS[dataset], RESNET20_FULL_MACS[dataset])
    logits, labels = np.load(f"{preds}/test_logits.npy"), np.load(f"{preds}/test_labels.npy")

    # lambda chosen on the clean calibration half of each split
    losses, _ = losses_and_exits(logits, labels, LAMBDAS, "prediction-consistency")
    ks = [ucb_lam(losses[:, p[:N_CAL]], EPS, DELTA) for p in splits]

    # violation_rate: fraction of splits whose risk (averaged over the corruptions of the group) is above eps;
    # violation_rate_per_corruption: fraction of (corruption, split) pairs above eps, so that a corruption that
    # breaks the risk control on its own is not hidden by the average
    rows = []
    risk, savings = risk_and_savings(logits, labels, ks, splits, fractions)
    rows.append({"severity": 0, "group": "clean", "risk": risk.mean(), "violation_rate": np.mean(risk > EPS),
                 "violation_rate_per_corruption": np.mean(risk > EPS),
                 "savings": savings.mean(), "lambda_median": np.median(LAMBDAS[ks])})

    labels_c = np.load(f"{preds}/cifar_c/labels.npy")
    for group, corruptions in [("standard", CORRUPTIONS), ("extra", EXTRA_CORRUPTIONS)]:
        per_severity = {s: ([], []) for s in SEVERITIES}
        for corruption in corruptions:
            logits_c = np.load(f"{preds}/cifar_c/{corruption}_logits.npy", mmap_mode="r")
            for s in SEVERITIES:
                risk, savings = risk_and_savings(np.asarray(logits_c[s - 1]), labels_c, ks, splits, fractions)
                per_severity[s][0].append(risk)
                per_severity[s][1].append(savings)
        for s in SEVERITIES:
            risk_each = np.array(per_severity[s][0])  # (corruptions, splits)
            risk = risk_each.mean(axis=0)  # mean over corruptions, one value per split
            rows.append({"severity": s, "group": group, "risk": risk.mean(),
                         "violation_rate": np.mean(risk > EPS),
                         "violation_rate_per_corruption": np.mean(risk_each > EPS),
                         "savings": np.mean(per_severity[s][1]), "lambda_median": np.median(LAMBDAS[ks])})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/cifar")
    out_dir = parser.parse_args().out_dir

    run_dirs = sorted(glob.glob(f"{out_dir}/*/*/seed*/metrics.json"))
    rng = np.random.default_rng(0)
    splits = [rng.permutation(10000) for _ in range(N_TRIALS)]

    acc_rows, auroc_rows, fys_rows = [], [], []
    for path in run_dirs:
        run_dir = os.path.dirname(path)
        dataset, regime, seed = run_dir.split(os.sep)[-3:]
        with open(path) as f:
            metrics = json.load(f)
        info = {"dataset": dataset, "regime": regime, "seed": int(seed[4:])}

        # accuracy of every exit: clean test (severity 0), then CIFAR-C averaged over each group of corruptions
        for l, a in enumerate(metrics["test"]):
            acc_rows.append({**info, "group": "clean", "severity": 0, "exit": l + 1, "accuracy": a})
        for group, corruptions in [("standard", CORRUPTIONS), ("extra", EXTRA_CORRUPTIONS)]:
            for s in SEVERITIES:
                for l in range(len(metrics["test"])):
                    a = np.mean([metrics["cifar_c"][c][s - 1][l] for c in corruptions])
                    acc_rows.append({**info, "group": group, "severity": s, "exit": l + 1, "accuracy": a})

        for row in score_auroc(run_dir):
            auroc_rows.append({**info, **row})
        for row in analyse_run(run_dir, dataset, splits):
            fys_rows.append({**info, **row})
        print("done", run_dir, flush=True)

    acc_df, auroc_df, fys_df = pd.DataFrame(acc_rows), pd.DataFrame(auroc_rows), pd.DataFrame(fys_rows)
    acc_df.to_csv(f"{out_dir}/first_look_accuracy.csv", index=False)
    auroc_df.to_csv(f"{out_dir}/first_look_auroc.csv", index=False)
    fys_df.to_csv(f"{out_dir}/first_look_fys.csv", index=False)

    pd.set_option("display.width", 200)
    for group in ["clean", "standard", "extra"]:
        d = acc_df[acc_df.group == group]
        table = d.groupby(["dataset", "regime", "severity", "exit"]).accuracy.mean().unstack("exit")
        print(f"\nAccuracy per exit, {group} (mean over seeds"
              + ("" if group == "clean" else " and corruptions") + "):")
        print(table.round(3).to_string())
    print("\nAUROC for 'the exit is correct', clean test, exits 1-8 (min and max over exits of the mean over seeds):")
    mean_auroc = auroc_df.groupby(["dataset", "regime", "exit"])[["auroc_max_softmax", "auroc_conf_unit"]].mean()
    print(mean_auroc.groupby(["dataset", "regime"]).agg(["min", "max"]).round(3).to_string())
    print(f"\nFast yet Safe, UCB, eps = {EPS}, delta = {DELTA}, lambda calibrated on clean data (mean over seeds):")
    table = fys_df[fys_df.group != "extra"].groupby(["dataset", "regime", "severity"])
    print(table[["risk", "violation_rate", "violation_rate_per_corruption", "savings", "lambda_median"]]
          .mean().round(3).to_string())
    print(f"\non clean data the violation rate should be about <= {DELTA} (binomial noise with {N_TRIALS} splits: "
          f"about +- {2 * np.sqrt(DELTA * (1 - DELTA) / N_TRIALS):.2f})")

    # figure: risk and savings against severity (standard corruptions), one line per regime
    datasets = sorted(fys_df.dataset.unique())
    fig, ax = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 6), squeeze=False)
    for j, dataset in enumerate(datasets):
        for regime in REGIMES:
            d = fys_df[(fys_df.dataset == dataset) & (fys_df.regime == regime) & (fys_df.group != "extra")]
            if d.empty:
                continue
            d = d.groupby("severity")[["risk", "savings"]].mean()
            ax[0, j].plot(d.index, d.risk, marker="o", label=regime)
            ax[1, j].plot(d.index, d.savings, marker="o", label=regime)
        ax[0, j].axhline(EPS, color="grey", lw=0.8)
        ax[0, j].set_title(dataset)
        ax[1, j].set_xlabel("severity (0 = clean)")
    ax[0, 0].set_ylabel("consistency risk")
    ax[1, 0].set_ylabel("savings")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/first_look.png", dpi=150)
    print("\nsaved in", out_dir)


if __name__ == "__main__":
    main()
