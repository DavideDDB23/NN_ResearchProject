# Phase D, entry D.05: MAC fractions and batch-1 latency of ResNet50 with 8 exits (exits.py), on CPU (one thread) and
# MPS, with random weights (the compute does not depend on the values; no download). Same measure as
# fase_d/latency.py for ResNet20: time to stop at each exit, with every head on the way evaluated and compared with
# the threshold on the host, over the time of the backbone alone. Written by Opus 5.5, 15 November 2026.
# Usage (from the repository root, with nothing else running):
#   python outputs/fable_review/fase_d/imagenet/latency_resnet50.py > outputs/fable_review/fase_d/imagenet/latency_resnet50.log

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exits import ResNet50Exits, mac_fractions  # noqa: E402


def timed(fn, device, warmup, reps):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        if device == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        fn()
        if device == "mps":
            torch.mps.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1000


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = ResNet50Exits(pretrained=False).eval()
    fr, total = mac_fractions(model)
    print(f"ResNet50 backbone: {total / 2e9:.2f} GMACs at 224x224")
    print("   MACs / M0    " + " ".join(f"{v:6.3f}" for v in fr))
    for device, warmup, reps in [("cpu", 5, 40), ("mps", 20, 200)]:
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        m = model.to(device)
        x = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            t0 = timed(lambda: m.plain(x).argmax(1).item(), device, warmup, reps)
            t = np.array([timed(lambda s=s: m.early_exit(x, s), device, warmup, reps) for s in range(9)])
        print(f"{device}: backbone alone {t0:.2f} ms per input")
        print("   time (ms)    " + " ".join(f"{v:6.2f}" for v in t))
        print("   time / M0    " + " ".join(f"{v:6.3f}" for v in t / t0))
        print("   difference   " + " ".join(f"{v:+6.3f}" for v in t / t0 - np.array(fr)), flush=True)


if __name__ == "__main__":
    main()
