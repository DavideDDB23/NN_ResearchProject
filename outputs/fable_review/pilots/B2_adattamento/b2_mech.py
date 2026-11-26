# Pilot B2, follow-up B2.02 (written by Fable, 15 November 2026): what the affine map on the logits of a head does.
# On the same cases as b2_heads.py (4 networks, 15 corruptions, severity 3 and 5, split 0), smaller maps on the
# logits z of each of the 8 heads, fitted by distillation towards the softmax of exit 9 as the affine map of
# b2_heads.py (same loss, full batch L-BFGS with the strong Wolfe line search, 100 iterations, no L2):
#   temp   z / T          one temperature per head: the confidence changes, the predictions do not
#   bias   z + a          one bias per class
#   diag   d * z + a      one scale and one bias per class
# on the fit rows of the row sets n4000 (adapt[:3000]) and n1000 (adapt[:750]). Then measure_config (b2_common.py):
# lambda with UCB on the cal rows, everything measured on eval. The term of comparison is logit_grid_n4000 (the full
# affine map) in runs_heads/. Only the saved logits are needed. One process, about 15 minutes.
# Output: runs_mech/<network>_<corruption>.csv (skipped if done unless --force).
#
# Usage (from the repository root): python outputs/fable_review/pilots/B2_adattamento/b2_mech.py [--force]

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b2_common import PILOT, RUNS, SEVERITIES, atomic_csv, case_rows, measure_config, run_info, run_name  # noqa: E402
from eesavings.adaptation import MAX_ITER  # noqa: E402
from eesavings.data.cifar import CORRUPTIONS, load_cifar_c  # noqa: E402
from eesavings.savings import RESNET20_FULL_MACS  # noqa: E402
from review_common import fractions_of  # noqa: E402

KINDS = ["temp", "bias", "diag"]
ROW_SETS = {"n4000": 3000, "n1000": 750}  # fit rows, the first of the adapt rows (as the grid and k configurations)


def apply_map(kind, z, params):
    # z (N, C) logits of one head; params as returned by fit_map
    if kind == "temp":
        return z * torch.exp(-params[0])
    if kind == "bias":
        return z + params[0]
    return z * params[0] + params[1]


def fit_map(kind, z, teacher):
    # the parameters start at the identity map; returns them and the number of closure evaluations
    c = z.shape[1]
    start = {"temp": [torch.zeros(1)], "bias": [torch.zeros(c)], "diag": [torch.ones(c), torch.zeros(c)]}[kind]
    params = [p.clone().requires_grad_(True) for p in start]
    optimizer = torch.optim.LBFGS(params, max_iter=MAX_ITER, line_search_fn="strong_wolfe")
    count = [0]

    def closure():
        count[0] += 1
        optimizer.zero_grad()
        loss = F.kl_div(F.log_softmax(apply_map(kind, z, params), dim=1), teacher, reduction="batchmean")
        loss.backward()
        return loss

    optimizer.step(closure)
    return [p.detach() for p in params], count[0]


def one_corruption(run_dir, corruption, out_path, data_dir):
    torch.set_num_threads(1)
    info, dataset = run_info(run_dir), run_info(run_dir)["dataset"]
    fr = fractions_of(dataset)
    labels = load_cifar_c(dataset, corruption, data_dir)[1]
    logits_c = np.load(os.path.join(run_dir, "preds", "cifar_c", f"{corruption}_logits.npy"), mmap_mode="r")
    rows_out = []
    for severity in SEVERITIES:
        logits = np.asarray(logits_c[severity - 1])
        adapt, cal, ev, rows = case_rows(0)
        src, y = logits[rows], labels[rows]
        z_all = [torch.as_tensor(np.ascontiguousarray(logits[:, l])) for l in range(8)]
        for set_name, n_fit in ROW_SETS.items():
            fit = adapt[:n_fit]
            teacher = F.softmax(torch.as_tensor(logits[fit, -1], dtype=torch.float32), dim=1)
            for kind in KINDS:
                start = time.time()
                new, evals = logits.copy(), 0
                for l in range(8):
                    params, e = fit_map(kind, z_all[l][fit], teacher)
                    evals += e
                    new[:, l] = apply_map(kind, z_all[l], params).numpy()
                scalars, _, _ = measure_config(new[rows], src, y, fr, n_fit)
                # one evaluation is a forward and a backward of N x C products (2 N C MACs), for temp too
                train_fwd = evals * 2 * n_fit * logits.shape[2] / RESNET20_FULL_MACS[dataset]
                rows_out.append({**info, "corruption": corruption, "severity": severity, "split": 0,
                                 "config": f"logit_{kind}_{set_name}", "n_rows_paid": n_fit, "n_fit": n_fit,
                                 "evals_total": evals, "train_fwd": train_fwd, "seconds": time.time() - start,
                                 **scalars})
    atomic_csv(pd.DataFrame(rows_out), out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=PILOT)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    start = time.time()
    for run_dir in RUNS:
        for corruption in CORRUPTIONS:
            path = os.path.join(args.out_dir, "runs_mech", f"{run_name(run_dir)}_{corruption}.csv")
            if os.path.exists(path) and not args.force:
                continue
            t = time.time()
            one_corruption(run_dir, corruption, path, args.data_dir)
            print(f"{run_name(run_dir)} {corruption}: {time.time() - t:.1f} s", flush=True)
    print(f"done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
