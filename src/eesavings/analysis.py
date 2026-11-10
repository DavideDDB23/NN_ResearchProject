# Helpers for the analysis of week 4: compute savings at a fixed risk level, under shift.
# Every quantity is measured on random splits of the 10,000 test images, and on the same rows of CIFAR-C:
#   adapt (4,000 images)  unlabeled data of the new domain, used only by the offline adaptation (M8)
#   cal (3,000)           calibration of lambda with UCB, as in Fast yet Safe
#   eval (3,000)          where risk, savings and the distribution over the exits are measured
# Methods of the project document:
#   M1  lambda calibrated on clean data and kept fixed (Fast yet Safe, static)
#   M2  lambda calibrated again on the new domain: the best savings without adaptation
#   M8  heads or scores re-fitted on the adapt rows of the new domain, then lambda calibrated again

import numpy as np

from eesavings.data.cifar import N_TEST
from eesavings.fys import losses_and_exits, ucb_wsr
from eesavings.savings import exit_distribution, savings_from_exits

# lambda grid, descending; row 0 is lambda = 1 (no early exit). Finer near 1 than the grid of Fast yet Safe,
# because on CIFAR the top softmax is often above 0.99.
LAMBDAS = np.concatenate([[1.0], 1 - np.logspace(-4, 0, 200)])
N_ADAPT = 4000
N_CAL = 3000


def make_splits(n_trials, n=N_TEST, n_adapt=N_ADAPT, n_cal=N_CAL, seed=0):
    # list of (adapt, cal, eval) index arrays, the same for every network
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(n_trials):
        perm = rng.permutation(n)
        splits.append((perm[:n_adapt], perm[n_adapt:n_adapt + n_cal], perm[n_adapt + n_cal:]))
    return splits


def consistency_losses(logits, conf):
    # consistency loss (exit used != exit 9) and exit of every input, for every lambda: (K, N) each
    return losses_and_exits(logits, np.zeros(len(logits), dtype=int), LAMBDAS, "prediction-consistency", conf=conf)


def ucb_index(cal_losses, eps, delta):
    # same choice as fys.ucb_lam: the last lambda before the first one whose UCB is not below eps
    # (fixed sequence from lambda = 1 down). Computed in order, it stops at the first failure.
    for k in range(len(cal_losses)):
        if ucb_wsr(cal_losses[k], delta=delta) >= eps:
            return max(0, k - 1)
    return len(cal_losses) - 1


def measure(losses, exits, k, rows, fractions, eps):
    # risk, savings and exit distribution on the given rows at lambda index k.
    # valid_savings counts the savings only if the risk is below eps (otherwise they are not allowed).
    risk = losses[k, rows].mean()
    savings = savings_from_exits(exits[k, rows], fractions)
    pi = exit_distribution(exits[k, rows], len(fractions))
    return {"lambda": LAMBDAS[k], "risk": risk, "savings": savings,
            "valid_savings": savings if risk <= eps else 0.0, **{f"pi_{l + 1}": p for l, p in enumerate(pi)}}
