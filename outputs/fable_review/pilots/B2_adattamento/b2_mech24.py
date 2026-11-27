# Pilot B2, follow-up B2.03 (15 November 2026): the small maps on the logits of b2_mech.py (temp, bias, diag) on all
# 24 networks, and after the BatchNorm statistics of job A.
#   part 1  temp, bias, diag with the fit rows of n4000 (adapt[:3000], the rows M8b fits its heads on) and n1000
#           (adapt[:750]), on the 24 networks (4 regimes x 3 seeds), the 15 corruptions at severity 3 and 5, splits 0,
#           1, 2: the cases where outputs/savings_under_shift_v2/results.csv has M2 and M8b. A row m2 per case (the
#           source logits, measured the same way) checks the match with results.csv.
#   part 2  bias and diag fitted on the logits of the network with the BatchNorm statistics of the 4,000 adapt rows
#           (cache of job A: seed 0, joint and joint_sd, split 0), teacher the adapted exit 9: c_logit_bias_n4000 and
#           c_logit_diag_n4000, to compare with bn_n4000 and c_logit_grid_n4000 of B2.
# Every row also has acc_at_m2_cost: the system accuracy of the configuration, read on its curve over lambda at the
# cost where m2 is certified on the same case (the accuracy of m2 there is the acc_at_m2_cost of the row m2).
# Only saved logits (and the cache). CPU, --workers processes with one thread each; jobs are (network, corruption).
# Output: runs_mech24/<network>_<corruption>.csv (skipped if done unless --force).
#
# Usage (from the repository root):
#   python outputs/fable_review/pilots/B2_adattamento/b2_mech24.py [--workers 3] [--force]

import argparse
import os
import sys
import time
from itertools import zip_longest
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b2_common import PILOT, SEVERITIES, SPLITS, atomic_csv, measure_config, run_info, run_name  # noqa: E402
from b2_mech import KINDS, ROW_SETS, apply_map, fit_map  # noqa: E402
from eesavings.data.cifar import CORRUPTIONS  # noqa: E402
from eesavings.savings import RESNET20_FULL_MACS  # noqa: E402
from review_common import all_runs, fractions_of  # noqa: E402
from review_common import run_dir as run_dir_of  # noqa: E402

SPLITS_USED = [0, 1, 2]
CACHE_RUNS = {("cifar10", "joint"), ("cifar10", "joint_sd"), ("cifar100", "joint"), ("cifar100", "joint_sd")}
C_KINDS = ["bias", "diag"]


def acc_at_cost(curves, cost):
    # accuracy of a curve over lambda at a given mean cost (the cost falls with the lambda index)
    return float(np.interp(cost, curves["cost"][::-1], curves["acc"][::-1]))


def fit_and_measure(kind, logits, fit, rows, src, y, fr, n_paid, dataset, cost_m2):
    # one map per head on the fit rows (teacher: the softmax of exit 9 of these logits), then measure_config
    teacher = F.softmax(torch.as_tensor(logits[fit, -1], dtype=torch.float32), dim=1)
    new, evals = logits.copy(), 0
    for l in range(8):
        z = torch.as_tensor(np.ascontiguousarray(logits[:, l]))
        params, e = fit_map(kind, z[fit], teacher)
        evals += e
        new[:, l] = apply_map(kind, z, params).numpy()
    scalars, _, curves = measure_config(new[rows], src, y, fr, n_paid)
    # one evaluation is a forward and a backward of N x C products (2 N C MACs), as in b2_mech.py
    train_fwd = evals * 2 * len(fit) * logits.shape[2] / RESNET20_FULL_MACS[dataset]
    return scalars, evals, train_fwd, acc_at_cost(curves, cost_m2)


def one_job(job):
    dataset, regime, seed, corruption, out_path, cache_dir = job
    torch.set_num_threads(1)
    start_job = time.time()
    d = run_dir_of(dataset, regime, seed)
    info, fr = run_info(d), fractions_of(dataset)
    labels = np.load(os.path.join(d, "preds", "cifar_c", "labels.npy"))  # the test labels, checked in pilot B1
    logits_c = np.load(os.path.join(d, "preds", "cifar_c", f"{corruption}_logits.npy"), mmap_mode="r")
    out = []

    def add(severity, split, config, scalars, n_paid, n_fit, evals, train_fwd, acc_m2_cost, seconds):
        out.append({**info, "corruption": corruption, "severity": severity, "split": split, "config": config,
                    "n_rows_paid": n_paid, "n_fit": n_fit, "evals_total": evals, "train_fwd": train_fwd,
                    "acc_at_m2_cost": acc_m2_cost, "seconds": seconds, **scalars})

    for severity in SEVERITIES:
        logits = np.asarray(logits_c[severity - 1])
        for split in SPLITS_USED:
            adapt, cal, ev = SPLITS[split]
            rows = np.concatenate([cal, ev])
            src, y = logits[rows], labels[rows]
            t = time.time()
            scalars, _, curves = measure_config(src, src, y, fr, 0)
            cost_m2 = 1.0 - scalars["savings"]
            add(severity, split, "m2", scalars, 0, 0, 0, 0.0, acc_at_cost(curves, cost_m2), time.time() - t)
            for set_name, n_fit in ROW_SETS.items():
                for kind in KINDS:
                    t = time.time()
                    res = fit_and_measure(kind, logits, adapt[:n_fit], rows, src, y, fr, n_fit, dataset, cost_m2)
                    add(severity, split, f"logit_{kind}_{set_name}", res[0], n_fit, n_fit, res[1], res[2], res[3],
                        time.time() - t)
            if split == 0 and seed == 0 and (dataset, regime) in CACHE_RUNS:
                path = os.path.join(cache_dir, f"{run_name(d)}_{corruption}_sev{severity}_split0_logits.npy")
                if not os.path.exists(path):
                    print(f"   no cache {path}: c_ skipped", flush=True)
                    continue
                bn_logits = np.load(path)
                for kind in C_KINDS:
                    t = time.time()
                    res = fit_and_measure(kind, bn_logits, adapt[:3000], rows, src, y, fr, 4000, dataset, cost_m2)
                    add(severity, split, f"c_logit_{kind}_n4000", res[0], 4000, 3000, res[1], res[2], res[3],
                        time.time() - t)
    atomic_csv(pd.DataFrame(out), out_path)
    print(f"{dataset} {regime} seed{seed} {corruption}: {time.time() - start_job:.1f} s", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out-dir", default=PILOT)
    parser.add_argument("--cache-dir", default=None, help="cache of job A, default <out-dir>/cache_bn")
    parser.add_argument("--runs", nargs="*", default=None, help="dataset/regime/seedK, default all 24")
    parser.add_argument("--corruptions", nargs="*", default=list(CORRUPTIONS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cache_dir = args.cache_dir or os.path.join(args.out_dir, "cache_bn")
    runs = all_runs()
    if args.runs:
        runs = [(r.split("/")[0], r.split("/")[1], int(r.split("/")[2].removeprefix("seed"))) for r in args.runs]
    by_dataset = {}
    for dataset, regime, seed in runs:
        for corruption in args.corruptions:
            path = os.path.join(args.out_dir, "runs_mech24", f"{dataset}_{regime}_seed{seed}_{corruption}.csv")
            if os.path.exists(path) and not args.force:
                continue
            by_dataset.setdefault(dataset, []).append((dataset, regime, seed, corruption, path, cache_dir))
    # CIFAR100 (slower) and CIFAR10 one after the other, so the workers stay busy until the end
    jobs = [j for group in zip_longest(*by_dataset.values()) for j in group if j is not None]
    print(len(jobs), "(network, corruption) pairs to do, workers", args.workers, flush=True)
    start = time.time()
    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(one_job, jobs, chunksize=1):
            pass
    print(f"done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
