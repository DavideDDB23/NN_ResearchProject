# Summary of the 24 trained networks, from their exported outputs (scripts/4_export/export_outputs.py):
#   1) accuracy of every exit on the clean test set and on CIFAR-C (15 standard corruptions, then the 4 extra),
#      mean over seeds and corruptions;
#   2) how well two exit scores separate correct from wrong predictions on the clean test set (AUROC, exits 1..8):
#      the max softmax and the confidence unit of the heads fitted at export time (target "label").
# The savings under shift are computed by scripts/5_savings_under_shift/savings_under_shift.py.
#
# Usage: python scripts/4_export/summarize_networks.py [--out-dir outputs/cifar]

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eesavings.data.cifar import CORRUPTIONS, EXTRA_CORRUPTIONS, SEVERITIES
from eesavings.fys import conf_measures


def accuracy_rows(metrics, info):
    # accuracy of every exit: clean test (severity 0), then CIFAR-C averaged over each group of corruptions
    rows = [{**info, "group": "clean", "severity": 0, "exit": l + 1, "accuracy": a}
            for l, a in enumerate(metrics["test"])]
    for group, corruptions in [("standard", CORRUPTIONS), ("extra", EXTRA_CORRUPTIONS)]:
        for s in SEVERITIES:
            for l in range(len(metrics["test"])):
                a = np.mean([metrics["cifar_c"][c][s - 1][l] for c in corruptions])
                rows.append({**info, "group": group, "severity": s, "exit": l + 1, "accuracy": a})
    return rows


def auroc_rows(preds, info):
    # AUROC of max softmax and of the confidence unit for "the exit is correct", on the clean test set
    logits, labels = np.load(f"{preds}/test_logits.npy"), np.load(f"{preds}/test_labels.npy")
    conf_unit = np.load(f"{preds}/test_conf.npy")
    max_softmax = conf_measures(logits)
    correct = logits.argmax(axis=2) == labels[:, None]
    return [{**info, "exit": l + 1, "auroc_max_softmax": roc_auc_score(correct[:, l], max_softmax[:, l]),
             "auroc_conf_unit": roc_auc_score(correct[:, l], conf_unit[:, l])}
            for l in range(logits.shape[1] - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/cifar")
    out_dir = parser.parse_args().out_dir

    acc, auroc = [], []
    for path in sorted(glob.glob(f"{out_dir}/*/*/seed*/metrics.json")):
        run_dir = os.path.dirname(path)
        dataset, regime, seed = run_dir.split(os.sep)[-3:]
        info = {"dataset": dataset, "regime": regime, "seed": int(seed[4:])}
        with open(path) as f:
            acc += accuracy_rows(json.load(f), info)
        auroc += auroc_rows(os.path.join(run_dir, "preds"), info)

    acc, auroc = pd.DataFrame(acc), pd.DataFrame(auroc)
    acc.to_csv(f"{out_dir}/summary_accuracy.csv", index=False)
    auroc.to_csv(f"{out_dir}/summary_auroc.csv", index=False)

    pd.set_option("display.width", 200)
    for group in ["clean", "standard", "extra"]:
        table = acc[acc.group == group].groupby(["dataset", "regime", "severity", "exit"]).accuracy.mean()
        averaged = "seeds" if group == "clean" else "seeds and corruptions"
        print(f"\nAccuracy per exit, {group} (mean over {averaged}):")
        print(table.unstack("exit").round(3).to_string())
    print("\nAUROC for 'the exit is correct', clean test, exits 1-8 (min and max over exits of the mean over seeds):")
    mean_auroc = auroc.groupby(["dataset", "regime", "exit"])[["auroc_max_softmax", "auroc_conf_unit"]].mean()
    print(mean_auroc.groupby(["dataset", "regime"]).agg(["min", "max"]).round(3).to_string())
    print("\nsaved in", out_dir)


if __name__ == "__main__":
    main()
