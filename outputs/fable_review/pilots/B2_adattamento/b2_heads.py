# Pilot B2, job B: the heads re-fitted by distillation towards exit 9, with less compute than M8b
# (SPECIFICA.md, section 6). For every network, corruption, severity 3 and 5 and split:
#   what   head   the weights and bias of the linear layer of each of the 8 heads, on its cached features (as
#                 adaptation.fit_head)
#          logit  an affine map z' = z @ A.T + a on the logits of each head, A from the identity, penalty
#                 l2 * ||A - I||^2 (the same fit, on the logits instead of the features)
#   how    grid   every L2 of adaptation.L2_GRID with MAX_ITER iterations, the L2 with the lowest distillation loss
#                 on the held-out rows kept (head_grid_n4000 is M8b)
#          k5, k10, k20   L2 = 0.01 and 5, 10, 20 L-BFGS iterations, on the fit rows only
#   rows   n4000, n2000, n1000, n500 (the first n adapt rows), free (the adapt rows that reach exit 9 under M2),
#          match (as many adapt rows as free); the first 3/4 fit, the last 1/4 chooses L2
# plus m2 (the source logits) and, on split 0 when job A left its cache, the BN then heads configurations c_*.
# The closure evaluations of L-BFGS are counted (training compute, as scripts/6_block3_review/adaptation_cost.py).
# CPU, --workers processes with one thread each; jobs are (network, corruption) pairs.
# Outputs in --out-dir: runs_heads/, heads_heads/, curves_heads/ (one file per network and corruption, skipped if
# done unless --force; new splits are added to the file).
#
# Usage (from the repository root):
#   python outputs/fable_review/pilots/B2_adattamento/b2_heads.py [--runs cifar10/joint/seed0 ...]
#          [--corruptions frost ...] [--severities 3 5] [--splits 0 1] [--workers 3] [--out-dir ...] [--force]

import argparse
import os
import sys
import time
from itertools import zip_longest
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b2_common import (PILOT, RUNS, SEVERITIES, Collector, case_rows, fit_held, m2_choice,  # noqa: E402
                       measure_config, out_paths, resolve_run, row_sets, run_info, run_name, todo_pairs,
                       train_macs, write_outputs)
from eesavings.adaptation import (L2_GRID, MAX_ITER, distillation_loss, head_logits, head_weights,  # noqa: E402
                                  split_by_exit)
from eesavings.data.cifar import CORRUPTIONS, load_cifar_c  # noqa: E402
from eesavings.models import FEATURE_DIMS  # noqa: E402
from eesavings.savings import RESNET20_FULL_MACS  # noqa: E402
from review_common import fractions_of  # noqa: E402

PARAMS = ["head", "logit"]
OPTS = {"grid": None, "k5": 5, "k10": 10, "k20": 20}
SETS = ["n4000", "n2000", "n1000", "n500", "free", "match"]
L2_K = 0.01
MIN_SET_ROWS = 200
C_CONFIGS = ["c_head_grid_n4000", "c_head_k10_n4000", "c_head_k20_n4000", "c_logit_grid_n4000", "c_logit_k10_n4000"]


def fit_head_counted(x, teacher, w0, b0, l2, max_iter=MAX_ITER):
    # adaptation.fit_head, operation by operation, plus the number of closure evaluations of L-BFGS
    w = torch.tensor(w0, requires_grad=True)
    b = torch.tensor(b0, requires_grad=True)
    w_start = torch.tensor(w0)
    optimizer = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")
    count = [0]

    def closure():
        count[0] += 1
        optimizer.zero_grad()
        loss = distillation_loss(x, teacher, w, b) + l2 * ((w - w_start) ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach(), b.detach(), count[0]


def fit_affine_counted(z, teacher, l2, max_iter=MAX_ITER):
    # affine map z @ A.T + a on the logits (N, C) of one head: the same fit with A from the identity and a from 0,
    # so the penalty is l2 * ||A - I||^2
    c = z.shape[1]
    return fit_head_counted(z, teacher, np.eye(c, dtype=np.float32), np.zeros(c, dtype=np.float32), l2, max_iter)


def teacher_of(exit9_logits):
    # softmax of exit 9, as in adaptation.adapt_heads
    return F.softmax(torch.as_tensor(np.asarray(exit9_logits), dtype=torch.float32), dim=1)


def layer_inputs(param, features, logits, rows):
    # inputs of the 8 linear maps on some rows: the feature blocks of the heads (as adapt_heads) or their logits
    if param == "head":
        return [torch.as_tensor(x) for x in split_by_exit(features[rows])]
    z = logits[rows]
    return [torch.as_tensor(np.ascontiguousarray(z[:, l])) for l in range(len(FEATURE_DIMS))]


def fit_layers(param, x_fit, x_held, t_fit, t_held, starts, opt):
    # grid: every L2 with MAX_ITER, the first with the lowest loss on the held-out rows kept (as adapt_heads);
    # k5 / k10 / k20: L2 = 0.01 and that many iterations. Returns new maps, chosen L2 and the evaluations
    new, chosen, evals = [], [], []
    for l in range(len(FEATURE_DIMS)):
        if OPTS[opt] is not None:
            if param == "head":
                w, b, e = fit_head_counted(x_fit[l], t_fit, starts[l][0], starts[l][1], L2_K, OPTS[opt])
            else:
                w, b, e = fit_affine_counted(x_fit[l], t_fit, L2_K, OPTS[opt])
            new.append((w.numpy(), b.numpy()))
            chosen.append(L2_K)
            evals.append(e)
            continue
        best = None
        for l2 in L2_GRID:
            if param == "head":
                w, b, e = fit_head_counted(x_fit[l], t_fit, starts[l][0], starts[l][1], l2)
            else:
                w, b, e = fit_affine_counted(x_fit[l], t_fit, l2)
            evals.append(e)
            with torch.no_grad():
                val_loss = distillation_loss(x_held[l], t_held, w, b).item()
            if best is None or val_loss < best[0]:
                best = (val_loss, w, b, l2)
        new.append((best[1].numpy(), best[2].numpy()))
        chosen.append(best[3])
    return new, chosen, evals


def config_logits(param, features, logits, new):
    # logits (10000, 9, C) of a configuration: exits 1..8 from the new maps, exit 9 unchanged
    if param == "head":
        heads = head_logits(features, [w for w, _ in new], [b for _, b in new])
    else:
        heads = np.stack([logits[:, l] @ w.T + b for l, (w, b) in enumerate(new)], axis=1)
    return np.concatenate([heads, logits[:, -1:]], axis=1)


def cost_columns(param, opt, evals, n_fit, n_held, n_classes, dataset):
    # training compute (L-BFGS evaluations) and, apart, the held-out losses that choose L2
    dims = list(FEATURE_DIMS) if param == "head" else [n_classes] * len(FEATURE_DIMS)
    n_l2 = len(L2_GRID) if OPTS[opt] is None else 1
    macs = train_macs(evals, n_fit, np.repeat(dims, n_l2), n_classes)
    held = n_held * n_classes * sum(dims) * len(L2_GRID) if OPTS[opt] is None else 0
    full = RESNET20_FULL_MACS[dataset]
    return {"evals_total": int(sum(evals)), "train_macs": macs, "train_fwd": macs / full, "heldout_fwd": held / full}


def adapt_set(add_row, prefix, set_name, R, wanted, features, logits, starts, src, y, rows, fr, dataset, n_paid_of):
    # every wanted (what, how) on one row set; features / logits are the source ones or the cache of job A
    fit, held = fit_held(R)
    t_fit, t_held = teacher_of(logits[fit, -1]), teacher_of(logits[held, -1])
    n_classes = logits.shape[2]
    for param in PARAMS:
        opts = [o for o in OPTS if f"{prefix}{param}_{o}_{set_name}" in wanted]
        if not opts:
            continue
        x_fit, x_held = layer_inputs(param, features, logits, fit), layer_inputs(param, features, logits, held)
        for opt in opts:
            start = time.time()
            new, chosen, evals = fit_layers(param, x_fit, x_held, t_fit, t_held, starts, opt)
            t_train = time.time() - start
            measured = measure_config(config_logits(param, features, logits, new)[rows], src, y, fr,
                                      n_paid_of(opt, R, fit))
            add_row(f"{prefix}{param}_{opt}_{set_name}", measured, time.time() - start,
                    n_rows_paid=n_paid_of(opt, R, fit), n_rows_set=len(R), n_fit=len(fit),
                    n_held=len(held) if OPTS[opt] is None else 0, seconds_train=t_train,
                    l2=" ".join(f"{x:g}" for x in chosen),
                    **cost_columns(param, opt, evals, len(fit), len(held), n_classes, dataset))


def rows_paid(opt, R, fit):
    # grid uses the fit and the held-out rows, k5 / k10 / k20 only the fit rows
    return len(R) if OPTS[opt] is None else len(fit)


def wanted_configs(split):
    # split 0: all 48; split 1: head_grid_n4000 and the 36 with k5, k10, k20
    names = [f"{p}_{o}_{s}" for p in PARAMS for o in OPTS for s in SETS]
    return set(names if split == 0 else [n for n in names if n == "head_grid_n4000" or "_grid_" not in n])


def heads_job(job):
    run_dir, corruption, pairs, out_dir, cache_dir, data_dir = job
    torch.set_num_threads(1)  # several (network, corruption) pairs run in parallel processes
    start_job = time.time()
    info, name = run_info(run_dir), run_name(run_dir)
    dataset = info["dataset"]
    fr = fractions_of(dataset)
    starts = list(zip(*head_weights(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu",
                                               weights_only=True))))
    labels = load_cifar_c(dataset, corruption, data_dir)[1]  # the same labels as job A
    preds = os.path.join(run_dir, "preds", "cifar_c")
    logits_c = np.load(os.path.join(preds, f"{corruption}_logits.npy"), mmap_mode="r")
    features_c = np.load(os.path.join(preds, f"{corruption}_features.npy"), mmap_mode="r")
    col = Collector(info, corruption)
    for severity in sorted({s for s, _ in pairs}):
        logits, features = np.asarray(logits_c[severity - 1]), np.asarray(features_c[severity - 1])
        for split in [t for s, t in pairs if s == severity]:
            adapt, cal, ev, rows = case_rows(split)
            src, y = logits[rows], labels[rows]

            def add_row(config, measured, seconds, **extra):
                col.add(severity, split, config, measured, seconds=seconds, **extra)

            start = time.time()
            measured = measure_config(src, src, y, fr, 0)
            add_row("m2", measured, time.time() - start, n_rows_paid=0, train_fwd=0.0)
            _, exit_src = m2_choice(logits, adapt, cal)
            wanted = wanted_configs(split)
            for set_name, R in row_sets(adapt, exit_src).items():
                if len(R) < MIN_SET_ROWS:
                    print(f"   {name} {corruption} sev {severity} split {split}: row set {set_name} has {len(R)} rows "
                          f"(< {MIN_SET_ROWS}), skipped", flush=True)
                    continue
                n_paid_of = (lambda opt, R, fit: 0) if set_name == "free" else rows_paid
                adapt_set(add_row, "", set_name, R, wanted, features, logits, starts, src, y, rows, fr, dataset,
                          n_paid_of)
            if split == 0:
                bn_then_heads(add_row, cache_dir, name, corruption, severity, adapt, starts, src, y, rows, fr, dataset)
    write_outputs(out_paths(out_dir, "heads", f"{name}_{corruption}"), col.runs, col.heads, col.curves, pairs)
    print(f"{name} {corruption}: {len(pairs)} cases in {time.time() - start_job:.1f} s", flush=True)
    return name, corruption


def bn_then_heads(add_row, cache_dir, name, corruption, severity, adapt, starts, src, y, rows, fr, dataset):
    # heads (or logit maps) re-fitted on the features and logits of the network with the BatchNorm statistics of
    # the 4,000 adapt rows (cache of job A); teacher the adapted exit 9; the 4,000 rows are paid by the statistics
    stem = os.path.join(cache_dir, f"{name}_{corruption}_sev{severity}_split0")
    if not os.path.exists(stem + "_logits.npy") or not os.path.exists(stem + "_features.npy"):
        print(f"   {name} {corruption} sev {severity}: no cache of job A in {cache_dir}, c_* skipped", flush=True)
        return
    logits, features = np.load(stem + "_logits.npy"), np.load(stem + "_features.npy")
    start = time.time()
    measured = measure_config(logits[rows], src, y, fr, 4000)
    add_row("c_bn_n4000", measured, time.time() - start, n_rows_paid=4000, train_fwd=0.0)
    adapt_set(add_row, "c_", "n4000", adapt, set(C_CONFIGS), features, logits, starts, src, y, rows, fr, dataset,
              lambda opt, R, fit: 4000)


def interleaved_jobs(run_dirs, corruptions, args):
    # (network, corruption) pairs still to do, CIFAR10 and CIFAR100 one after the other
    by_dataset = {}
    for run_dir in run_dirs:
        name = run_name(run_dir)
        for corruption in corruptions:
            pairs = todo_pairs(out_paths(args.out_dir, "heads", f"{name}_{corruption}"), args.severities,
                               args.splits, args.force)
            if pairs:
                by_dataset.setdefault(run_info(run_dir)["dataset"], []).append(
                    (run_dir, corruption, pairs, args.out_dir, args.cache_dir, args.data_dir))
            else:
                print(f"{name} {corruption}: already done, skipped", flush=True)
    return [j for group in zip_longest(*by_dataset.values()) for j in group if j is not None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=RUNS)
    parser.add_argument("--corruptions", nargs="*", default=CORRUPTIONS)
    parser.add_argument("--severities", nargs="*", type=int, default=SEVERITIES)
    parser.add_argument("--splits", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out-dir", default=PILOT)
    parser.add_argument("--cache-dir", default=None, help="cache of job A, default <out-dir>/cache_bn")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", default="data", help="where CIFAR-10-C and CIFAR-100-C are (labels only)")
    args = parser.parse_args()
    args.cache_dir = args.cache_dir or os.path.join(args.out_dir, "cache_bn")
    jobs = interleaved_jobs([resolve_run(r) for r in args.runs], args.corruptions, args)
    print(len(jobs), "(network, corruption) pairs to do, workers", args.workers, "splits", args.splits,
          "severities", args.severities, flush=True)
    start = time.time()
    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(heads_job, jobs, chunksize=1):
            pass
    print(f"done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()

