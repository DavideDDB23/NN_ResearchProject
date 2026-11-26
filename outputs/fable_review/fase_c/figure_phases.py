# Phase C, the figure of the plan v3 for Checkpoint 3b: the timeline of the net savings around the shift for the
# three versions of the adaptation phase (reinforced audits M7, full fallback M7f, continuous M7c) against the
# controller alone (M4), with the fraction recovered and the payback. Top: net savings, smoothed over 1,000 inputs.
# Bottom: compute saved by each version against M4 since the shift, cumulated (full forwards per input since the
# shift, i.e. the difference of the mean net savings up to that point): where it crosses zero, the adaptation has
# paid back. Curves averaged over networks, corruptions and repetitions. Written by Opus, 15 November 2026.
# Usage (from the repository root):
#   python outputs/fable_review/fase_c/figure_phases.py --dir outputs/fable_review/fase_c/variants \
#          --out outputs/fable_review/fase_c/phases_sev3.png

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLOCK, SMOOTH = 100, 10
METHODS = [("M4", "M4 controller, senza adattamento", "tab:green"),
           ("M7", "M7 audit rinforzati (25%)", "tab:red"),
           ("M7f", "M7f fallback completo", "tab:blue"),
           ("M7c", "M7c adattamento continuo", "tab:purple")]


def curves(d, dataset, method, key):
    out = [np.load(f)[key] for f in sorted(glob.glob(os.path.join(d, "curves", f"{dataset}_*_{method}.npz")))]
    return np.mean(out, axis=0) if out else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    t0 = json.load(open(os.path.join(args.dir, "args.json")))["t0"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for j, dataset in enumerate(["cifar10", "cifar100"]):
        c4 = curves(args.dir, dataset, "M4", "cost")
        if c4 is None:
            continue
        s0 = 1 - c4[(t0 - 5000) // BLOCK:t0 // BLOCK].mean()
        s1 = 1 - c4[-100:].mean()
        for m, label, color in METHODS:
            c = curves(args.dir, dataset, m, "cost")
            if c is None:
                continue
            net = np.convolve(1 - c, np.ones(SMOOTH) / SMOOTH, mode="valid")
            x = (np.arange(len(net)) + SMOOTH) * BLOCK - t0
            s2 = 1 - c[-100:].mean()
            text = label if m == "M4" else f"{label}: recupera {100 * (s2 - s1) / (s0 - s1):.0f}%"
            axes[0][j].plot(x, net, color=color, label=text, linewidth=1.2)
            if m != "M4":
                saved = np.cumsum((c4 - c)[t0 // BLOCK:]) * BLOCK  # full forwards saved since the shift
                xs = (np.arange(len(saved)) + 1) * BLOCK
                neg = np.flatnonzero(saved < 0)
                pay = (neg[-1] + 1) * BLOCK if (len(neg) and saved[-1] > 0) else (0 if not len(neg) else None)
                if pay is None:
                    lab = f"{m}: non ripagato entro lo stream"
                elif pay == 0:
                    lab = f"{m}: davanti a M4 fin dallo shift"
                else:
                    lab = f"{m}: pareggio dopo {pay:,} input".replace(",", ".")
                axes[1][j].plot(xs, saved / xs, color=color, label=lab, linewidth=1.2)
        axes[0][j].axvline(0, color="black", linewidth=0.8)
        axes[0][j].set_title(f"{dataset.upper()}, severity 3 (media su 4 reti del seed 0, 15 corruzioni)")
        axes[0][j].set_ylabel("savings netti")
        axes[0][j].grid(alpha=0.3)
        axes[0][j].legend(fontsize=7)
        axes[1][j].axhline(0, color="black", linewidth=0.8)
        axes[1][j].set_ylabel("compute risparmiato su M4 dallo shift,\nper input (forward complete)")
        axes[1][j].set_xlabel("input dallo shift")
        axes[1][j].grid(alpha=0.3)
        axes[1][j].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()

