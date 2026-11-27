# Phase D, entry D.02: real latency of early exit inference at batch 1, on CPU (one thread) and MPS.
# For one trained network per dataset (joint, seed 0) it times, per input:
#   - the plain ResNet20 (conv1, the 9 blocks, average pooling and exit 9, no heads): the cost of M0
#   - early exit inference forced to stop at exit l = 1..9: blocks up to l, the pooled heads 1..l (classifier only,
#     the confidence unit is not used by the max softmax score), the max softmax of each and the comparison with the
#     threshold on the host (which on MPS needs a synchronisation per exit)
# Then it turns real exit distributions into time: for the clean threshold (UCB on the clean calibration rows) on the
# clean stream rows, and for the threshold of M2 on each corruption at severity 3 (mean over the 15 corruptions), the
# expected time over the time of M0 against the expected MACs over the MACs of M0.
# Written by Opus 5.5, 15 November 2026. Usage (from the repository root, with nothing else running):
#   python outputs/fable_review/fase_d/latency.py > outputs/fable_review/fase_d/latency.log

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
sys.path.insert(0, "outputs/fable_review/fase_c")
sys.path.insert(0, "scripts/6_block3_review")
from eesavings.data.cifar import CORRUPTIONS, cifar_arrays, to_tensor_batch  # noqa: E402
from eesavings.models import MultiExitResNet20  # noqa: E402
from eesavings.train import load_model  # noqa: E402
from review_common import fractions_of, load_logits  # noqa: E402
from sim import pools, tables, ucb_lambda  # noqa: E402

WARMUP, REPS = 100, 1000


def plain(model, x):
    out = F.relu(model.bn1(model.conv1(x)))
    for block in model.blocks():
        out = block(out)
    return model.fc(F.avg_pool2d(out, out.size(3)).flatten(1)).argmax(1).item()


def early_exit(model, x, stop):
    # stop at exit `stop` (0-based); every head on the way is evaluated and compared with the threshold on the host
    out = F.relu(model.bn1(model.conv1(x)))
    for l, block in enumerate(model.blocks()):
        out = block(out)
        if l < 8:
            head = model.heads[l]
            z = head.classifier(head.features(out))
            score = F.softmax(z, dim=1).max().item()
            if l == stop or score > 2.0:  # the comparison costs the same; `stop` forces the exit
                return l
    return model.fc(F.avg_pool2d(out, out.size(3)).flatten(1)).argmax(1).item()


def timed(fn, device):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(REPS):
        if device == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        fn()
        if device == "mps":
            torch.mps.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1000


def exit_histograms(dataset):
    # exit distributions of the clean threshold on clean rows and of M2 on every corruption at severity 3
    cal_rows, stream_rows = pools()
    clean = load_logits(dataset, "joint", 0)

    def hist(logits, lam):
        s, _ = tables(logits)
        above = s > lam
        e = np.where(above.any(axis=1), above.argmax(axis=1), 8)
        return np.bincount(e, minlength=9) / len(e)

    h_clean = hist(clean[stream_rows], ucb_lambda(clean[cal_rows]))
    h_shift = np.mean([hist(x[stream_rows], ucb_lambda(x[cal_rows]))
                       for x in (load_logits(dataset, "joint", 0, c, 3) for c in CORRUPTIONS)], axis=0)
    return h_clean, h_shift


def main():
    torch.set_num_threads(1)
    for dataset, classes in [("cifar10", 10), ("cifar100", 100)]:
        fr = fractions_of(dataset)
        h_clean, h_shift = exit_histograms(dataset)
        images, _ = cifar_arrays(dataset, "test", "data")
        for device in ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]:
            model = load_model(MultiExitResNet20(classes), f"outputs/cifar/{dataset}/joint/seed0/model.pt")
            model = model.to(device).eval()
            x = to_tensor_batch(images[:1], dataset).to(device)
            with torch.no_grad():
                t0 = timed(lambda: plain(model, x), device)
                t = np.array([timed(lambda l=l: early_exit(model, x, l), device) for l in range(9)])
            rel = t / t0
            print(f"{dataset} {device}: plain ResNet20 {t0:.3f} ms per input")
            print("   exit         " + " ".join(f"{l + 1:>6d}" for l in range(9)))
            print("   time (ms)    " + " ".join(f"{v:6.3f}" for v in t))
            print("   time / M0    " + " ".join(f"{v:6.3f}" for v in rel))
            print("   MACs / M0    " + " ".join(f"{v:6.3f}" for v in fr))
            for name, h in [("clean threshold on clean data", h_clean), ("M2 at severity 3, mean of 15", h_shift)]:
                mac = 1 - float(h @ fr)
                lat = 1 - float(h @ rel)
                print(f"   {name}: savings in MACs {mac:.3f}, in time {lat:.3f} ({lat / mac:.2f} of the MAC savings)")
            print(flush=True)


if __name__ == "__main__":
    main()
