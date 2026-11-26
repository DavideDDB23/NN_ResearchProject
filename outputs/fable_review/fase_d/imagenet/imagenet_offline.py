# Phase D, entry D.08: offline reading of the minimal ImageNet run (tables from colab_imagenet.py in out/tables/).
# For every domain (half to calibrate, half to evaluate, permutation with seed 0):
#   M1  UCB (WSR, delta 0.1, eps 0.05) on the clean calibration half, applied to the evaluation half of the domain
#   M2  UCB on the calibration half of the domain itself
#   oracle  the largest savings with empirical risk <= eps on the evaluation half
# and the exit distributions of M1 and M2 turned into time with the batch-1 time fractions of ResNet50 measured in
# latency_resnet50.log (CPU one thread and MPS). Accuracy per exit and the test of the logit maps come from the
# summaries written on Colab. Written by Opus 5.5, 16 November 2026. Usage (from the repository root):
#   python outputs/fable_review/fase_d/imagenet/imagenet_offline.py > outputs/fable_review/fase_d/imagenet/imagenet_offline.log

import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from eesavings.analysis import LAMBDAS, ucb_index_fast  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
EPS, DELTA = 0.05, 0.1
# from latency_resnet50.log (16 November): MACs, and time over the backbone alone at batch 1, exits 1..9
MACS = np.array([0.139, 0.283, 0.390, 0.535, 0.642, 0.749, 0.894, 0.948, 1.002])
TIME = {"cpu": np.array([0.309, 0.454, 0.521, 0.639, 0.725, 0.810, 0.929, 0.993, 1.046]),
        "mps": np.array([0.194, 0.371, 0.496, 0.668, 0.800, 0.930, 1.210, 1.366, 1.506])}


def exits_of(scores, lam):
    above = scores > lam
    return np.where(above.any(axis=1), above.argmax(axis=1), 8)


def losses(scores, preds):
    # (K, n) consistency losses along the grid of LAMBDAS (from lambda = 1 down), as eesavings.analysis expects
    out = np.empty((len(LAMBDAS), len(scores)), dtype=np.float64)
    for k, lam in enumerate(LAMBDAS):
        e = exits_of(scores, lam)
        out[k] = preds[np.arange(len(e)), e] != preds[:, 8]
    return out


def at(scores, preds, lam):
    e = exits_of(scores, lam)
    hist = np.bincount(e, minlength=9) / len(e)
    risk = float((preds[np.arange(len(e)), e] != preds[:, 8]).mean())
    return 1 - float(hist @ MACS), risk, hist


def oracle(scores, preds):
    best = -np.inf
    for lam in LAMBDAS:
        s, r, _ = at(scores, preds, lam)
        if r <= EPS:
            best = max(best, s)
    return best


def main():
    names = [n[:-4] for n in sorted(os.listdir(os.path.join(OUT, "tables")))]
    data = {}
    for n in names:
        d = np.load(os.path.join(OUT, "tables", f"{n}.npz"))
        perm = np.random.default_rng(0).permutation(len(d["labels"]))
        half = len(perm) // 2
        data[n] = {k: (d["scores"].astype(np.float32)[idx], d["preds"].astype(np.int64)[idx])
                   for k, idx in [("cal", perm[:half]), ("eval", perm[half:])]}
    lam1 = float(LAMBDAS[ucb_index_fast(losses(*data["clean"]["cal"]), EPS, DELTA)])
    print(f"M1 threshold (UCB on {len(data['clean']['cal'][0])} clean rows): {lam1:.4f}\n")
    order = ["clean"] + [n for n in names if n != "clean"]
    print(f"{'domain':18s} {'acc exit 9':>10s} {'M1 savings':>10s} {'M1 risk':>8s} {'M2 savings':>10s} {'M2 risk':>8s} "
          f"{'oracle':>7s} {'time M2 cpu':>11s} {'time M2 mps':>11s}  maps (plain, half, 3000)")
    for n in order:
        summary = json.load(open(os.path.join(OUT, f"summary_{n}.json")))
        ev = data[n]["eval"]
        s1, r1, _ = at(*ev, lam1)
        lam2 = float(LAMBDAS[ucb_index_fast(losses(*data[n]["cal"]), EPS, DELTA)])
        s2, r2, h2 = at(*ev, lam2)
        tcpu, tmps = 1 - float(h2 @ TIME["cpu"]), 1 - float(h2 @ TIME["mps"])
        maps = summary.get("maps", {})
        print(f"{n:18s} {summary['accuracy_per_exit'][8]:10.4f} {s1:10.4f} {r1:8.4f} {s2:10.4f} {r2:8.4f} "
              f"{oracle(*ev):7.4f} {tcpu:11.4f} {tmps:11.4f}  "
              f"{maps.get('plain', '')}, {maps.get('maps_half', '')}, {maps.get('maps_3000', '')}")
        if n == "clean":
            print(f"{'':18s} exit distribution of M2 on clean: {np.round(h2, 3).tolist()}")
    print("\naccuracy per exit (evaluation images, from Colab):")
    for n in order:
        acc = json.load(open(os.path.join(OUT, f"summary_{n}.json")))["accuracy_per_exit"]
        print(f"   {n:18s} " + " ".join(f"{a:.3f}" for a in acc))


if __name__ == "__main__":
    main()
