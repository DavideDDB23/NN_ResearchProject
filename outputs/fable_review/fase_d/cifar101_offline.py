# Phase D, entry D.01: the offline confirmation on CIFAR-10.1 (natural shift), the 12 CIFAR10 networks.
#   M1  the clean threshold (UCB on the 5,000 clean calibration rows, as lam1 of the simulator) on the 2,000 images
#   M2  UCB on one half of CIFAR-10.1 (1,000 images), savings and risk on the other half, both ways; against M2 on the
#       clean test set with the same calibration size (1,000 clean calibration rows, evaluated on 1,000 stream rows)
#   maps  one scale and one bias per class (fit_diag of the simulator) fitted on one half, and the savings at the
#       empirical oracle threshold (risk <= eps on the evaluation half) with and without them; the same on clean data
# The halves are those of the online runner (permutation with seed 0). It only prints. Written by Opus 5.5,
# 15 November 2026. Usage (from the repository root):
#   python outputs/fable_review/fase_d/cifar101_offline.py > outputs/fable_review/fase_d/cifar101_offline.log

import os
import sys

import numpy as np

sys.path.insert(0, "outputs/fable_review/fase_c")
sys.path.insert(0, "scripts/6_block3_review")
from review_common import fractions_of, load_logits  # noqa: E402
from sim import EPS, fit_diag, pools, restart_lambda, tables, ucb_lambda  # noqa: E402

C101 = "outputs/fable_review/fase_d/cifar101"
FR = fractions_of("cifar10")


def at(logits, lam, scale=None, bias=None):
    # savings and consistency risk of the threshold lam on these rows
    s, p = tables(logits, scale, bias)
    above = s > lam
    e = np.where(above.any(axis=1), above.argmax(axis=1), 8)
    return 1.0 - FR[e].mean(), float((p[np.arange(len(e)), e] != p[:, 8]).mean())


def two_folds(a, b):
    # M2 (UCB on one half, evaluated on the other) and the gain of the maps at the empirical oracle threshold
    m2, gain, oracle = [], [], []
    for cal, ev in [(a, b), (b, a)]:
        m2.append(at(ev, ucb_lambda(cal)))
        ones = np.ones(len(ev))
        plain = at(ev, restart_lambda(ev, ones, FR, None, None))[0]
        scale, bias, _ = fit_diag(cal)
        mapped = at(ev, restart_lambda(ev, ones, FR, scale, bias), scale, bias)[0]
        oracle.append(plain)
        gain.append(mapped - plain)
    return np.mean([x[0] for x in m2]), np.mean([x[1] for x in m2]), np.mean(oracle), np.mean(gain)


def main():
    labels = np.load(f"{C101}/labels.npy")
    perm = np.random.default_rng(0).permutation(len(labels))
    half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
    cal_rows, stream_rows = pools()
    rows = []
    for regime in ["joint", "disjoint", "mixed", "joint_sd"]:
        for seed in [0, 1, 2]:
            clean = load_logits("cifar10", regime, seed)
            shifted = np.load(f"{C101}/{regime}_seed{seed}_logits.npy")
            lam1 = ucb_lambda(clean[cal_rows])
            m1_clean = at(clean[stream_rows], lam1)
            m1_shift = at(shifted, lam1)
            c = two_folds(clean[cal_rows[:1000]], clean[stream_rows[:1000]])
            q = two_folds(shifted[half_a], shifted[half_b])
            rows.append([m1_clean[0], m1_clean[1], m1_shift[0], m1_shift[1], c[0], c[1], q[0], q[1], c[2], q[2],
                         c[3], q[3]])
            print(f"{regime} seed{seed}: M1 clean {m1_clean[0]:.3f} (risk {m1_clean[1]:.3f}), CIFAR-10.1 "
                  f"{m1_shift[0]:.3f} (risk {m1_shift[1]:.3f}); M2 with 1,000 rows clean {c[0]:.3f} (risk {c[1]:.3f}), "
                  f"CIFAR-10.1 {q[0]:.3f} (risk {q[1]:.3f}); oracle {c[2]:.3f} and {q[2]:.3f}; gain of the maps at "
                  f"the oracle threshold clean {c[3]:+.4f}, CIFAR-10.1 {q[3]:+.4f}", flush=True)
    r = np.array(rows)
    m = r.mean(axis=0)
    print(f"\nmean over the 12 networks: M1 clean {m[0]:.4f} (risk {m[1]:.4f}), CIFAR-10.1 {m[2]:.4f} (risk {m[3]:.4f})")
    print(f"   M2 with 1,000 calibration rows: clean {m[4]:.4f} (risk {m[5]:.4f}), CIFAR-10.1 {m[6]:.4f} (risk {m[7]:.4f}); "
          f"ratio {m[6] / m[4]:.3f} (per network from {np.min(r[:, 6] / r[:, 4]):.3f} to {np.max(r[:, 6] / r[:, 4]):.3f})")
    print(f"   empirical oracle threshold: clean {m[8]:.4f}, CIFAR-10.1 {m[9]:.4f}")
    print(f"   gain of the maps at the oracle threshold: clean {m[10]:+.4f}, CIFAR-10.1 {m[11]:+.4f} "
          f"(per network from {r[:, 11].min():+.4f} to {r[:, 11].max():+.4f}); eps {EPS}")


if __name__ == "__main__":
    main()
