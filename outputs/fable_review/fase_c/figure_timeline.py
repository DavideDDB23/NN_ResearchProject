# Phase C, the figure of the timeline (plan v3, Checkpoint 3b): net savings and risk around the shift, averaged over
# networks, corruptions and repetitions, for the methods present in the folders given, one column per dataset and one
# row per quantity. A vertical line at the shift. Curves at the resolution of the runner (100 inputs), smoothed over
# 1,000 inputs. Written by Opus, 15 November 2026.
# Usage (from the repository root):
#   python outputs/fable_review/fase_c/figure_timeline.py --dirs outputs/fable_review/fase_c/main \
#          --methods M1 M2 M4 M7 M8c --severity 3 --out outputs/fable_review/fase_c/timeline_sev3.png

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLOCK, SMOOTH = 100, 10
LABELS = {"M1": "M1 soglia pulita fissa", "M2": "M2 soglia oracolo", "M3": "M3 monitor e fallback",
          "M4": "M4 controller", "M5": "M5 adattamento, solo input arrivati in fondo",
          "M6": "M6 adattamento, solo audit", "M7": "M7 adattamento, audit rinforzati",
          "M7f": "M7f adattamento, fallback completo", "M7c": "M7c adattamento continuo",
          "M7t": "M7t adattamento con trigger sulla soglia", "M8": "M8 mappe e soglia oracolo",
          "M8c": "M8c mappe oracolo e controller", "M7w": "M7w fase con fit pesato",
          "M7cw": "M7cw continuo con fit pesato", "M7wg": "M7wg fase con fit pesato e protezione"}


def mean_curve(dirs, dataset, severity, method, key):
    # every method from the last folder that has it, never mixed across folders (main_v2 after main: M7 with the
    # fixed restart of entry C.04 replaces the flawed one; M1, M2, M8c stay from main)
    for d in reversed(dirs):
        files = sorted(glob.glob(os.path.join(d, "curves", f"{dataset}_*_{severity}_{method}.npz")))
        if files:
            c = np.mean([np.load(f)[key] for f in files], axis=0)
            return np.convolve(c, np.ones(SMOOTH) / SMOOTH, mode="valid"), len(files)
    return None, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=["M1", "M2", "M4", "M7", "M8c"])
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--marks", nargs="*", type=int, default=[],
                        help="other changes of domain, in inputs from the shift (dashed lines; 15000 for the return)")
    args = parser.parse_args()
    t0 = json.load(open(os.path.join(args.dirs[0], "args.json")))["t0"]
    datasets = ["cifar10", "cifar100"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for j, dataset in enumerate(datasets):
        for i, (key, ylabel) in enumerate([("cost", "savings netti"), ("z", "consistency risk")]):
            ax = axes[i][j]
            for m in args.methods:
                c, n = mean_curve(args.dirs, dataset, args.severity, m, key)
                if c is None:
                    continue
                x = (np.arange(len(c)) + SMOOTH) * BLOCK - t0
                y = 1 - c if key == "cost" else c
                ax.plot(x, y, label=f"{LABELS.get(m, m)} ({n})", linewidth=1.2)
            ax.axvline(0, color="black", linewidth=0.8)
            for m in args.marks:
                ax.axvline(m, color="black", linewidth=0.8, linestyle="--")
            if key == "z":
                ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{dataset.upper()}, severity {args.severity}")
            ax.grid(alpha=0.3)
        axes[1][j].set_xlabel("input dallo shift")
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()

