# Phase C, entry C.21: the threshold of the guard of M7wg, chosen on clean streams only, by the rule written in the
# notebook before any run. Clean streams of 40,000 inputs, the 4 networks of seed 0 of each dataset, 10 repetitions
# (the same stream draws as the runner with seed 0); the moving average of the exit-1 probabilities is the one of
# sim.py (weight 1/window per input, start from the mean over the clean calibration rows); reference at input 10,000;
# a run raises a false alarm if the total variation from the reference goes above kappa at any input from 10,001 to
# 40,000. For each dataset the rule takes the smallest kappa of the grid with false alarms in at most 1% of the runs.
# Written by Opus 5.5, 15 November 2026. Usage (from the repository root):
#   python outputs/fable_review/fase_c/guard_calibrate.py > outputs/fable_review/fase_c/guard_calibrate.log

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "scripts/6_block3_review")
from run import load_domains  # noqa: E402
from sim import Stream, exit1_probs  # noqa: E402

GRID = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
T, T_REF, REPS, WINDOW = 40000, 10000, 10, 1000


def max_tv(dataset, regime):
    base = load_domains(dataset, regime, 0, ["clean"])
    p1 = exit1_probs(base["logits"]["clean"])
    stream = Stream([("clean", T)], REPS, seed=0)
    ema = np.tile(exit1_probs(base["cal"]["clean"]).mean(axis=0), (REPS, 1)).astype(np.float64)
    ref, worst = None, np.zeros(REPS)
    for t in range(T):
        ema += (p1[stream.idx[:, t]] - ema) / WINDOW
        if t == T_REF - 1:
            ref = ema.copy()
        elif t >= T_REF:
            worst = np.maximum(worst, 0.5 * np.abs(ema - ref).sum(axis=1))
    return worst


def main():
    for dataset in ["cifar10", "cifar100"]:
        worst = np.concatenate([max_tv(dataset, r) for r in ["joint", "disjoint", "mixed", "joint_sd"]])
        rates = {k: float(np.mean(worst > k)) for k in GRID}
        chosen = min([k for k in GRID if rates[k] <= 0.01], default=None)
        print(f"{dataset}: {len(worst)} runs; largest TV per run: median {np.median(worst):.4f}, max {worst.max():.4f}")
        print(f"   false alarm rate by kappa: {rates}")
        print(f"   chosen kappa: {chosen}")


if __name__ == "__main__":
    main()
