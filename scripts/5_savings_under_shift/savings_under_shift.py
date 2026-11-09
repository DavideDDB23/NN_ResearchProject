# Week 4: how the compute savings change under shift when the consistency risk must stay below eps.
# For every trained network (outputs/cifar/<dataset>/<regime>/seed<k>) and every exit score
# (scores.SCORES, fitted on val), on random adapt / cal / eval splits of the test images (analysis.make_splits):
#   clean  lambda calibrated on clean cal, measured on clean eval (S*_P of the project document)
#   M1     the clean lambda measured on the eval rows of each corruption and severity
#   M2     lambda calibrated again on the cal rows of the corrupted images (S*_Q with the trained heads)
# and, at eps = 0.05 with the "agreement" score:
#   M8a    the agreement score re-fitted on the adapt rows of the corrupted images, then M2
#   M8b    (seed 0, whose CIFAR-C features are saved) the linear layers of the heads re-fitted by distillation
#          towards exit 9 on 3,000 adapt rows (adaptation.adapt_heads, L2 chosen on the other 1,000), the agreement
#          score re-fitted on those 1,000 held-out rows (on the rows used to fit the heads it would be overconfident),
#          then M2. The adapted heads are also measured with the max softmax score, which needs no re-fit: that
#          isolates the effect of the heads from the effect of re-fitting the score (M8a).
# Only the 15 standard corruptions. One row per (network, score, eps, method, corruption, severity, split): one
# file per network in outputs/week4/runs/ (networks already done are skipped, unless --force), all together in
# outputs/week4/results.csv; scripts/week4_figures.py makes the tables and figures.
#
# Usage: python scripts/week4_savings.py [--workers 8] [--trials 5] [--adapt-trials 3] [--runs ...] [--force]

import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from eesavings.adaptation import L2_GRID, adapt_heads, head_logits, head_weights
from eesavings.analysis import consistency_losses, make_splits, measure, ucb_index
from eesavings.data.cifar import CORRUPTIONS, SEVERITIES
from eesavings.export import confidence_targets
from eesavings.savings import RESNET20_EXIT_MACS, RESNET20_FULL_MACS, cost_fractions
from eesavings.scores import SCORES, exit_scores, fit_learned, fit_scores

EPSILONS = [0.02, 0.05, 0.1]
EPS_ADAPT = 0.05
DELTA = 0.1


N_ADAPT_HEADS = 3000  # adapt rows used to fit the heads in M8b; the others choose L2 and fit the score


def agreement_losses(logits, fit_rows):
    # agreement score re-fitted on fit_rows only (no labels), then losses and exits for every lambda
    params = {"agreement": fit_learned(logits[fit_rows], confidence_targets(logits[fit_rows], None, "agreement"))}
    return consistency_losses(logits, exit_scores(logits, "agreement", params))


def run_file(runs_dir, run_dir):
    # outputs/week4/runs/<dataset>_<regime>_seed<k>.csv
    return os.path.join(runs_dir, "_".join(run_dir.split(os.sep)[-3:]) + ".csv")


def calibration_rows(add, preds, splits, fractions):
    # clean, M1 and M2 for every score and eps, and M8a (agreement score re-fitted on the adapt rows)
    params = fit_scores(np.load(f"{preds}/val_logits.npy"), np.load(f"{preds}/val_labels.npy"))

    # clean: lambda of every (score, eps, split), reused by M1 on the corrupted images
    test_logits = np.load(f"{preds}/test_logits.npy")
    clean_k = {}
    for score in SCORES:
        losses, exits = consistency_losses(test_logits, exit_scores(test_logits, score, params))
        for eps in EPSILONS:
            for t, (_, cal, ev) in enumerate(splits):
                clean_k[score, eps, t] = ucb_index(losses[:, cal], eps, DELTA)
                add("clean", score, eps, "none", 0, t, measure(losses, exits, clean_k[score, eps, t], ev, fractions, eps))

    for corruption in CORRUPTIONS:
        logits_c = np.load(f"{preds}/cifar_c/{corruption}_logits.npy", mmap_mode="r")
        for severity in SEVERITIES:
            logits = np.asarray(logits_c[severity - 1])
            for score in SCORES:
                losses, exits = consistency_losses(logits, exit_scores(logits, score, params))
                for eps in EPSILONS:
                    for t, (_, cal, ev) in enumerate(splits):
                        add("M1", score, eps, corruption, severity, t,
                            measure(losses, exits, clean_k[score, eps, t], ev, fractions, eps))
                        k = ucb_index(losses[:, cal], eps, DELTA)
                        add("M2", score, eps, corruption, severity, t, measure(losses, exits, k, ev, fractions, eps))
            for t, (adapt, cal, ev) in enumerate(splits):
                losses, exits = agreement_losses(logits, adapt)
                k = ucb_index(losses[:, cal], EPS_ADAPT, DELTA)
                add("M8a", "agreement", EPS_ADAPT, corruption, severity, t,
                    measure(losses, exits, k, ev, fractions, EPS_ADAPT))
        print(preds, corruption, "calibration done", flush=True)


def adaptation_rows(add, run_dir, preds, splits, fractions):
    # M8b: heads adapted on the adapt rows (no labels), then lambda calibrated again; with the agreement score
    # re-fitted on the held-out adapt rows, and with the max softmax score. The chosen L2 strengths are kept.
    weights, biases = head_weights(torch.load(f"{run_dir}/model.pt", map_location="cpu", weights_only=True))
    for corruption in CORRUPTIONS:
        logits_c = np.load(f"{preds}/cifar_c/{corruption}_logits.npy", mmap_mode="r")
        features_c = np.load(f"{preds}/cifar_c/{corruption}_features.npy", mmap_mode="r")
        for severity in SEVERITIES:
            logits, features = np.asarray(logits_c[severity - 1]), np.asarray(features_c[severity - 1])
            for t, (adapt, cal, ev) in enumerate(splits):
                heads_rows, held_out = adapt[:N_ADAPT_HEADS], adapt[N_ADAPT_HEADS:]
                new_w, new_b, chosen = adapt_heads(features[heads_rows], logits[heads_rows, -1],
                                                   features[held_out], logits[held_out, -1], weights, biases)
                adapted = np.concatenate([head_logits(features, new_w, new_b), logits[:, -1:]], axis=1)
                for score in ["agreement", "max_softmax"]:
                    if score == "agreement":
                        losses, exits = agreement_losses(adapted, held_out)
                    else:
                        losses, exits = consistency_losses(adapted, exit_scores(adapted, "max_softmax"))
                    k = ucb_index(losses[:, cal], EPS_ADAPT, DELTA)
                    result = measure(losses, exits, k, ev, fractions, EPS_ADAPT)
                    add("M8b", score, EPS_ADAPT, corruption, severity, t,
                        {**result, "l2": " ".join(f"{x:g}" for x in chosen)})
        print(preds, corruption, "adaptation done", flush=True)


def analyse_run(job):
    # only_adaptation: compute only M8b and replace the M8b rows of the file of this network (the other methods
    # do not change), a much lighter run
    run_dir, n_trials, n_adapt_trials, runs_dir, only_adaptation = job
    torch.set_num_threads(1)  # several networks run in parallel processes
    dataset, regime, seed = run_dir.split(os.sep)[-3:]
    info = {"dataset": dataset, "regime": regime, "seed": int(seed[4:])}
    preds = os.path.join(run_dir, "preds")
    fractions = cost_fractions(RESNET20_EXIT_MACS[dataset], RESNET20_FULL_MACS[dataset])
    splits = make_splits(n_trials)
    rows = []

    def add(method, score, eps, corruption, severity, t, result):
        rows.append({**info, "score": score, "eps": eps, "method": method, "corruption": corruption,
                     "severity": severity, "split": t, **result})

    if not only_adaptation:
        calibration_rows(add, preds, splits, fractions)
    if os.path.exists(f"{preds}/cifar_c/{CORRUPTIONS[0]}_features.npy"):  # only seed 0 has the features
        adaptation_rows(add, run_dir, preds, splits[:n_adapt_trials], fractions)

    new = pd.DataFrame(rows)
    if only_adaptation:
        old = pd.read_csv(run_file(runs_dir, run_dir))
        new = pd.concat([old[old.method != "M8b"], new], ignore_index=True)
    new.to_csv(run_file(runs_dir, run_dir), index=False)
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/cifar")
    parser.add_argument("--results", default="outputs/week4/results.csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--adapt-trials", type=int, default=3, help="splits used for the head adaptation (M8b)")
    parser.add_argument("--runs", nargs="*", default=None, help="only these run folders (default: all exported)")
    parser.add_argument("--force", action="store_true", help="analyse again networks already done")
    parser.add_argument("--only-adaptation", action="store_true",
                        help="recompute only M8b for networks already done, keeping their other rows")
    args = parser.parse_args()

    run_dirs = args.runs or sorted(os.path.dirname(p) for p in glob.glob(f"{args.out_dir}/*/*/seed*/metrics.json"))
    runs_dir = os.path.join(os.path.dirname(args.results), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    done = [r for r in run_dirs if os.path.exists(run_file(runs_dir, r))]
    if args.only_adaptation:
        todo = done
    else:
        todo = [r for r in run_dirs if args.force or r not in done]
    print(len(todo), "networks to analyse", "(M8b only)" if args.only_adaptation else "")
    with Pool(args.workers) as pool:
        pool.map(analyse_run, [(r, args.trials, args.adapt_trials, runs_dir, args.only_adaptation) for r in todo],
                 chunksize=1)
    files = sorted(glob.glob(f"{runs_dir}/*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(args.results, index=False)
    with open(os.path.splitext(args.results)[0] + "_settings.json", "w") as f:
        json.dump({"epsilons": EPSILONS, "eps_adapt": EPS_ADAPT, "delta": DELTA, "trials": args.trials,
                   "adapt_trials": args.adapt_trials, "l2_grid": L2_GRID,
                   "networks": [os.path.basename(f)[:-4] for f in files]}, f, indent=2)
    print("saved", args.results, len(df), "rows")


if __name__ == "__main__":
    main()
