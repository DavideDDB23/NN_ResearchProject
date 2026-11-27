# Pilot B2, job A: the BatchNorm statistics of the network estimated again on unlabeled images of the new domain
# (SPECIFICA.md, section 5). For every network, corruption, severity 3 and 5 and split:
#   source       the saved logits (the M2 baseline of job B, measured the same way)
#   bn_n500 ... bn_n4000   the statistics of every BatchNorm layer from the images adapt[:n], in one batch
#   bn_traffic   the stem from the 4,000 adapt rows, block j only from the adapt rows that reach block j under M2
# then predict on the cal and eval rows and measure_config (b2_common.py). bn_n4000 of split 0 runs all the 10,000
# rows and saves logits and head features in cache_bn/ for job B (BN then heads).
# One process on MPS (CPU if there is none). Outputs in --out-dir: runs_bn/, heads_bn/, curves_bn/ (one file per
# network and corruption, skipped if done unless --force; new splits are added to the file) and cache_bn/.
#
# Usage (from the repository root):
#   python outputs/fable_review/pilots/B2_adattamento/b2_bn.py [--runs cifar10/joint/seed0 ...]
#          [--corruptions frost ...] [--severities 3 5] [--splits 0 1] [--out-dir ...] [--force] [--data-dir data]

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b2_common import (PILOT, RUNS, SEVERITIES, Collector, atomic_npy, case_rows, m2_choice,  # noqa: E402
                       measure_config, out_paths, resolve_run, run_info, run_name, todo_pairs, write_outputs)
from eesavings.data.cifar import CORRUPTIONS, NUM_CLASSES, load_cifar_c, to_tensor_batch  # noqa: E402
from eesavings.export import predict  # noqa: E402
from eesavings.models import N_EXITS, MultiExitResNet20  # noqa: E402
from eesavings.train import load_model  # noqa: E402
from review_common import fractions_of  # noqa: E402

N_BN = [500, 1000, 2000, 4000]
MIN_BN_ROWS = 64   # a block that fewer rows reach keeps the source statistics


def sync(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def bn_statistics(model, x, block_masks=None, min_rows=MIN_BN_ROWS):
    # Follows MultiExitResNet20.forward by hand without the heads. At every BatchNorm: mean and variance over the
    # selected rows (dims 0, 2, 3); the running statistics get the unbiased variance, and ALL the rows are
    # normalized with that mean and the biased variance (as PyTorch in training mode), so the next layer sees
    # inputs normalized with the new statistics. block_masks: None (every row everywhere) or 9 boolean tensors,
    # one per residual block; the stem always uses every row. A block with fewer than min_rows rows keeps the
    # source statistics (and normalizes with them). Returns the adapted copy, the rows used per block (9) and the
    # blocks (1..9) that kept the source statistics.
    adapted = copy.deepcopy(model).eval()
    rows_used, kept = [], []

    def normalize(bn, h, mask):
        sel = h if mask is None else h[mask]
        if mask is not None and len(sel) < min_rows:
            return F.batch_norm(h, bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps), False
        n = sel.numel() // sel.shape[1]
        var, mean = torch.var_mean(sel, dim=(0, 2, 3), correction=0)
        bn.running_mean.copy_(mean)
        bn.running_var.copy_(var * n / (n - 1))
        return F.batch_norm(h, mean, var, bn.weight, bn.bias, False, 0.0, bn.eps), True

    with torch.no_grad():
        out = F.relu(normalize(adapted.bn1, adapted.conv1(x), None)[0])
        for j, block in enumerate(adapted.blocks()):
            mask = None if block_masks is None else block_masks[j]
            h, new = normalize(block.bn1, block.conv1(out), mask)
            h, _ = normalize(block.bn2, block.conv2(F.relu(h)), mask)
            out = F.relu(h + block.shortcut(out))
            rows_used.append(len(x) if mask is None else int(mask.sum()))
            if not new:
                kept.append(j + 1)
    return adapted, rows_used, kept


def traffic_masks(exit_src, device):
    # block j (1..9) is reached by the adapt rows that leave at exit j or later under M2
    return [torch.as_tensor(np.asarray(exit_src) >= j, device=device) for j in range(1, N_EXITS + 1)]


def cache_paths(cache_dir, name, corruption, severity):
    stem = os.path.join(cache_dir, f"{name}_{corruption}_sev{severity}_split0")
    return stem + "_logits.npy", stem + "_features.npy"


def adapted_config(add_row, model, x_all, rows, src, y, fr, dataset, device, adapt_rows, masks, n_paid, config,
                   cache=None):
    # statistics pass, predict on the measured rows (all 10,000 with the features when the cache is written)
    start = time.time()
    adapted, used, kept = bn_statistics(model, to_tensor_batch(x_all[adapt_rows], dataset).to(device), masks)
    sync(device)
    t_stats = time.time() - start
    if kept:
        print(f"   {config}: blocks {kept} reached by fewer than {MIN_BN_ROWS} rows keep the source statistics "
              f"(rows per block {used})", flush=True)
    start = time.time()
    if cache is not None:
        z_all, _, f_all = predict(adapted, x_all, dataset, device, return_features=True)
        new = z_all[rows]
    else:
        new = predict(adapted, x_all[rows], dataset, device)[0]
    t_predict = time.time() - start
    if cache is not None:
        atomic_npy(cache[0], z_all.astype(np.float32))
        atomic_npy(cache[1], f_all.astype(np.float16))
    measured = measure_config(new, src, y, fr, n_paid)
    add_row(config, measured, time.time() - start + t_stats, n_rows_paid=n_paid, seconds_stats=t_stats,
            seconds_predict=t_predict, n_predicted=len(x_all) if cache is not None else len(rows),
            blocks_kept_source=len(kept), **{f"rows_block_{j + 1}": r for j, r in enumerate(used)})


def run_corruption(model, run_dir, corruption, pairs, args, device, mps_check):
    info = run_info(run_dir)
    dataset, name = info["dataset"], run_name(run_dir)
    fr = fractions_of(dataset)
    images, labels = load_cifar_c(dataset, corruption, args.data_dir)
    logits_c = np.load(os.path.join(run_dir, "preds", "cifar_c", f"{corruption}_logits.npy"), mmap_mode="r")
    col = Collector(info, corruption)
    for severity in sorted({s for s, _ in pairs}):
        x_all, src_all = np.array(images[severity - 1]), np.asarray(logits_c[severity - 1])  # images in memory
        for split in [t for s, t in pairs if s == severity]:
            adapt, cal, ev, rows = case_rows(split)
            src, y = src_all[rows], labels[rows]

            def add_row(config, measured, seconds, **extra):
                col.add(severity, split, config, measured, seconds=seconds, train_fwd=0.0, **extra)

            extra = {}
            if mps_check.get(name) is None:
                # how far the source model run now on this device is from the exported logits
                z = predict(model, x_all[rows], dataset, device)[0]
                mps_check[name] = float(np.abs(z - src).max())
                extra["mps_max_abs_diff"] = mps_check[name]
                print(f"   {name}: largest |logit now on {device} - saved logit| on the 6,000 rows of {corruption} "
                      f"severity {severity} split {split}: {mps_check[name]:.3g}", flush=True)
            start = time.time()
            measured = measure_config(src, src, y, fr, 0)
            add_row("source", measured, time.time() - start, n_rows_paid=0, **extra)

            for n in N_BN:
                cache = None
                if n == 4000 and split == 0:
                    cache = cache_paths(args.cache_dir, name, corruption, severity)
                adapted_config(add_row, model, x_all, rows, src, y, fr, dataset, device, adapt[:n], None, n,
                               f"bn_n{n}", cache)
            _, exit_src = m2_choice(src_all, adapt, cal)
            adapted_config(add_row, model, x_all, rows, src, y, fr, dataset, device, adapt,
                           traffic_masks(exit_src, device), 0, "bn_traffic")
    write_outputs(out_paths(args.out_dir, "bn", f"{name}_{corruption}"), col.runs, col.heads, col.curves, pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=RUNS)
    parser.add_argument("--corruptions", nargs="*", default=CORRUPTIONS)
    parser.add_argument("--severities", nargs="*", type=int, default=SEVERITIES)
    parser.add_argument("--splits", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--out-dir", default=PILOT)
    parser.add_argument("--cache-dir", default=None, help="default <out-dir>/cache_bn")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    args.cache_dir = args.cache_dir or os.path.join(args.out_dir, "cache_bn")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device", device, "splits", args.splits, "severities", args.severities, flush=True)
    mps_check = {}
    total = time.time()
    for run in args.runs:
        run_dir = resolve_run(run)
        info, name = run_info(run_dir), run_name(run_dir)
        model = None
        for corruption in args.corruptions:
            pairs = todo_pairs(out_paths(args.out_dir, "bn", f"{name}_{corruption}"), args.severities, args.splits,
                               args.force)
            if not pairs:
                print(f"{name} {corruption}: already done, skipped", flush=True)
                continue
            if model is None:
                model = load_model(MultiExitResNet20(NUM_CLASSES[info["dataset"]]),
                                   os.path.join(run_dir, "model.pt")).to(device).eval()
            start = time.time()
            run_corruption(model, run_dir, corruption, pairs, args, device, mps_check)
            print(f"{name} {corruption}: {len(pairs)} cases in {time.time() - start:.1f} s", flush=True)
    print(f"done in {time.time() - total:.0f} s", flush=True)


if __name__ == "__main__":
    main()

