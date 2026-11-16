# Review of block 3: the compute of the adaptation of the heads (M8b) is never subtracted from its savings. From the
# pilot outputs/fable_review/parte1/m8b_cost.py (same logic), measured here on 3 domains per dataset instead of 1:
# frost, gaussian_noise and jpeg_compression at severity 3, regime joint, seed 0 (the features of seed 0 are saved).
# For every domain: the function evaluations of L-BFGS in adaptation.adapt_heads (each one is a forward and a
# backward through the linear layer of a head on the 3,000 adaptation rows, 2 * N * D * C MACs), in MACs and in
# "full forwards" (one full forward of the ResNet20 = RESNET20_FULL_MACS). Then, with the mean training cost of the
# 3 domains, the payback against M2: the gain of M8b over M2 (seed 0, mean over regimes and corruptions) against
# the extra compute, counted two ways for the 4,000 adaptation inputs:
#   incremental  they run the whole network with all heads instead of leaving early with the M2 threshold
#   full         their 4,000 full forwards (the audits counted in full)
# Not counted: the recursion inside L-BFGS and the held-out loss that chooses L2.
# Output: outputs/block3_review/adaptation_cost.csv and adaptation_cost.log.
#
# Usage: python scripts/6_block3_review/adaptation_cost.py [--results outputs/savings_under_shift/results.csv]

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from eesavings.adaptation import L2_GRID, MAX_ITER, distillation_loss, head_weights, split_by_exit
from eesavings.analysis import make_splits
from eesavings.models import FEATURE_DIMS
from eesavings.savings import RESNET20_EXIT_MACS, RESNET20_FULL_MACS
from review_common import OUT, RESULTS, load_logits, run_dir, start_log

DOMAINS = [("frost", 3), ("gaussian_noise", 3), ("jpeg_compression", 3)]
N_ADAPT, N_HEADS_ROWS = 4000, 3000


def evaluation_macs(n_rows, dim, n_classes):
    # one evaluation of the loss and its gradient for a linear head: the forward x @ W.T and the gradient x.T @ g,
    # N * D * C MACs each
    return 2 * n_rows * dim * n_classes


def count_evaluations(features, teacher, weights, biases):
    # number of L-BFGS evaluations of every (head, L2), as in adaptation.fit_head
    evals = []
    for x, w0, b0 in zip(split_by_exit(features), weights, biases):
        x = torch.as_tensor(x)
        for l2 in L2_GRID:
            w, b = torch.tensor(w0, requires_grad=True), torch.tensor(b0, requires_grad=True)
            w_start = torch.tensor(w0)
            opt = torch.optim.LBFGS([w, b], max_iter=MAX_ITER, line_search_fn="strong_wolfe")
            count = [0]

            def closure():
                count[0] += 1
                opt.zero_grad()
                loss = distillation_loss(x, teacher, w, b) + l2 * ((w - w_start) ** 2).sum()
                loss.backward()
                return loss

            opt.step(closure)
            evals.append(count[0])
    return evals


def domain_cost(dataset, regime, corruption, severity):
    # L-BFGS evaluations and MACs of the training of the heads on one domain (first split, 3,000 rows)
    d = run_dir(dataset, regime, 0)
    rows = make_splits(1)[0][0][:N_HEADS_ROWS]
    logits = load_logits(dataset, regime, 0, corruption, severity)
    features = np.load(os.path.join(d, "preds", "cifar_c", f"{corruption}_features.npy"), mmap_mode="r")
    features = np.asarray(features[severity - 1])
    weights, biases = head_weights(torch.load(os.path.join(d, "model.pt"), map_location="cpu", weights_only=True))
    teacher = F.softmax(torch.as_tensor(logits[rows, -1], dtype=torch.float32), dim=1)
    start = time.time()
    evals = count_evaluations(features[rows], teacher, weights, biases)
    dims = np.repeat(FEATURE_DIMS, len(L2_GRID))  # evals are ordered by head, then by L2
    macs = sum(e * evaluation_macs(len(rows), dim, logits.shape[2]) for e, dim in zip(evals, dims))
    return {"dataset": dataset, "regime": regime, "corruption": corruption, "severity": severity,
            "evals_mean": np.mean(evals), "evals_min": min(evals), "evals_max": max(evals), "evals_total": sum(evals),
            "macs": macs, "train_full_forwards": macs / RESNET20_FULL_MACS[dataset], "seconds": time.time() - start}


def payback(ms, dataset, train_fwd):
    # gain of M8b over M2 at severity 3 and 5 (seed 0, mean over regimes and corruptions) and the payback in inputs
    # of the same domain, with the adaptation inputs counted incrementally and in full
    full = RESNET20_FULL_MACS[dataset]
    c9 = RESNET20_EXIT_MACS[dataset][-1] / full
    rows = []
    for sev in [3, 5]:
        g = ms[(ms.dataset == dataset) & (ms.severity == sev)]
        s_m2 = g[g.method == "M2"].savings.mean()
        s_m8b = g[g.method == "M8b"].savings.mean()
        gain = s_m8b - s_m2
        incremental = N_ADAPT * (c9 - (1 - s_m2))
        full_fwd = N_ADAPT * c9
        rows.append({"dataset": dataset, "severity": sev, "M2": s_m2, "M8b": s_m8b, "gain": gain,
                     "train_full_forwards": train_fwd, "adapt_incremental": incremental, "adapt_full": full_fwd,
                     "payback_incremental": (incremental + train_fwd) / gain,
                     "payback_full": (full_fwd + train_fwd) / gain,
                     "payback_without_training": incremental / gain,
                     "train_over_incremental": train_fwd / incremental, "train_over_full": train_fwd / full_fwd})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--out-dir", default=OUT)
    args = parser.parse_args()
    start_log("adaptation_cost", args.out_dir)
    torch.set_num_threads(1)
    df = pd.read_csv(args.results, dtype={"l2": str})
    ms = df[(df.score == "max_softmax") & np.isclose(df.eps, 0.05) & (df.seed == 0) & df.split.isin([0, 1, 2])]
    costs, paybacks = [], []
    for dataset in ["cifar10", "cifar100"]:
        print(f"\n{dataset}, regime joint, seed 0")
        for corruption, severity in DOMAINS:
            c = domain_cost(dataset, "joint", corruption, severity)
            costs.append(c)
            print(f"   {corruption} severity {severity}: L-BFGS evaluations per (head, L2): mean "
                  f"{c['evals_mean']:.0f}, min {c['evals_min']}, max {c['evals_max']}; total {c['evals_total']} evaluations, "
                  f"{c['macs']:.3g} MACs = {c['train_full_forwards']:,.0f} full forwards ({c['seconds']:.0f}s on one "
                  f"CPU thread)")
        train = np.array([c["train_full_forwards"] for c in costs if c["dataset"] == dataset])
        evals = np.array([c["evals_mean"] for c in costs if c["dataset"] == dataset])
        print(f"   over the {len(DOMAINS)} domains: training of the heads mean {train.mean():,.0f} full forwards "
              f"(min {train.min():,.0f}, max {train.max():,.0f}); mean evaluations per (head, L2) {evals.mean():.0f} "
              f"(min {evals.min():.0f}, max {evals.max():.0f})")
        for p in payback(ms, dataset, train.mean()):
            paybacks.append(p)
            print(f"   severity {p['severity']} (mean over regimes and corruptions, seed 0): M2 {p['M2']:.3f}, M8b "
                  f"{p['M8b']:.3f}, gain {p['gain']:.3f}")
            print(f"      extra compute: training {p['train_full_forwards']:,.0f} full forwards (mean of the domains); "
                  f"the 4,000 adaptation inputs {p['adapt_incremental']:,.0f} incremental over M2, "
                  f"{p['adapt_full']:,.0f} as full forwards")
            print(f"      payback after {p['payback_incremental']:,.0f} inputs of the same domain (incremental), "
                  f"{p['payback_full']:,.0f} (full forwards); without the training cost "
                  f"{p['payback_without_training']:,.0f}; the training is {p['train_over_incremental']:.2f} times the "
                  f"incremental cost and {p['train_over_full']:.2f} times the full forwards")
        # the cost of the recalibration alone (M2): 3,000 inputs through the whole network
        c9 = RESNET20_EXIT_MACS[dataset][-1] / RESNET20_FULL_MACS[dataset]
        s = ms[(ms.dataset == dataset) & (ms.severity == 3) & (ms.method == "M2")].savings.mean()
        for horizon in [10000, 30000, 100000]:
            net = (3000 * (1 - c9) + (horizon - 3000) * s) / horizon
            print(f"   M2 net of its 3,000 calibration forwards over {horizon} inputs of one domain: {net:.3f} "
                  f"(gross {s:.3f})")
    pd.DataFrame(costs).to_csv(os.path.join(args.out_dir, "adaptation_cost.csv"), index=False)
    pd.DataFrame(paybacks).to_csv(os.path.join(args.out_dir, "adaptation_cost_payback.csv"), index=False)


if __name__ == "__main__":
    main()
