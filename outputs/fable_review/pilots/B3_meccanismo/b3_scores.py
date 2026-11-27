# Pilot B3, part 2 (hypothesis C2 of the plan v3): learned scores of the agreement with exit 9 that use the
# statistics of all the exits computed so far, with the shared threshold and with a cost-aware exit rule. Written by
# Opus on 15 November 2026; design and predictions in outputs/fable_review/quaderno.md, entry B3.01.
# For every network (24), corruption (15), severity 3 and 5, split 0 of analysis.make_splits(5):
#   m2                   max softmax, shared lambda (M2: it must equal the M2 rows of results.csv)
#   ms_cost              max softmax with the cost-aware transform
#   cur_<data>[_cost]    one logistic regression per exit on the 3 statistics of exit l (log top probability, margin,
#                        entropy: scores.output_statistics), as the agreement score of block 3 and M8a
#   hist_<data>[_cost]   ... on the statistics of exits 1..l and the indicators "exit l agrees with exit j", j < l
#   <data>: adapt (the 4,000 adapt rows of the shifted domain, which need exit 9: 4,000 audits) or clean (the 5,000
#   clean val rows of the network, no cost)
# cost-aware transform: score'_l = 1 - (1 - p_l) (c_9 - c_1) / (c_9 - c_l), clipped to [0, 1], so that one shared
# lambda means "leave at the first exit l with P(disagreement) < mu (c_9 - c_l)", mu = (1 - lambda) / (c_9 - c_1).
# Lambda with UCB on the cal rows, everything measured on the eval rows (eesavings.analysis.measure).
# AUROC of every score for the agreement of every exit with exit 9, on the eval rows.
# Reads outputs/cifar only. Output: scores/<network>.csv in this folder (skipped if done).
#
# Usage (from the repository root):
#   python outputs/fable_review/pilots/B3_meccanismo/b3_scores.py [--workers 3] [--runs cifar10/joint/seed0 ...]

import argparse
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts/6_block3_review")
from eesavings.analysis import consistency_losses, make_splits, measure, ucb_index_fast  # noqa: E402
from eesavings.data.cifar import CORRUPTIONS  # noqa: E402
from eesavings.scores import exit_scores, output_statistics  # noqa: E402
from review_common import all_runs, fractions_of, load_logits, run_dir  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EPS, DELTA = 0.05, 0.1
SEVERITIES = [3, 5]
SPLIT = 0


def feature_sets(logits):
    # per exit l (0-based, 0..7): the features of "cur" and of "hist"
    stats = [output_statistics(logits[:, l]) for l in range(8)]
    preds = logits.argmax(axis=2)
    cur = stats
    hist = []
    for l in range(8):
        agree_prev = [(preds[:, l] == preds[:, j]).astype(float)[:, None] for j in range(l)]
        hist.append(np.concatenate(stats[:l + 1] + agree_prev, axis=1))
    return {"cur": cur, "hist": hist}


def agreement(logits):
    preds = logits.argmax(axis=2)
    return (preds[:, :8] == preds[:, 8:9]).astype(int)  # (N, 8)


def fit_models(feats, agree, rows):
    # one logistic regression per exit, trained on the given rows
    models = []
    for l in range(8):
        y = agree[rows, l]
        if y.min() == y.max():
            models.append(float(y[0]))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models.append(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
                          .fit(feats[l][rows], y))
    return models


def predict_scores(models, feats):
    # (N, 9) P(agreement) at every exit, last column 1
    n = len(feats[0])
    out = np.ones((n, 9))
    for l, model in enumerate(models):
        out[:, l] = model if isinstance(model, float) else model.predict_proba(feats[l])[:, 1]
    return out


def cost_aware(scores, fr):
    # score'_l = 1 - (1 - p_l) (c_9 - c_1) / (c_9 - c_l), clipped to [0, 1]; the last column stays 1
    mult = (fr[8] - fr[0]) / (fr[8] - fr[:8])
    out = scores.copy()
    out[:, :8] = np.clip(1.0 - (1.0 - scores[:, :8]) * mult, 0.0, 1.0)
    return out


def one_network(job):
    (dataset, regime, seed), out_path, corruptions = job
    start = time.time()
    fr = fractions_of(dataset)
    adapt, cal, ev = make_splits(5)[SPLIT]
    key = {"dataset": dataset, "regime": regime, "seed": seed}

    # the models trained on the clean val rows, once per network
    val = np.load(os.path.join(run_dir(dataset, regime, seed), "preds", "val_logits.npy"))
    val_feats, val_agree = feature_sets(val), agreement(val)
    clean_models = {name: fit_models(val_feats[name], val_agree, np.arange(len(val))) for name in ["cur", "hist"]}

    rows, auc_rows = [], []
    for corruption in corruptions:
        for severity in SEVERITIES:
            logits = load_logits(dataset, regime, seed, corruption, severity)
            feats, agree = feature_sets(logits), agreement(logits)
            ms = exit_scores(logits, "max_softmax")
            scores = {"m2": ms, "ms_cost": cost_aware(ms, fr)}
            for name in ["cur", "hist"]:
                p_adapt = predict_scores(fit_models(feats[name], agree, adapt), feats[name])
                p_clean = predict_scores(clean_models[name], feats[name])
                scores[f"{name}_adapt"], scores[f"{name}_adapt_cost"] = p_adapt, cost_aware(p_adapt, fr)
                scores[f"{name}_clean"], scores[f"{name}_clean_cost"] = p_clean, cost_aware(p_clean, fr)
            case = {**key, "corruption": corruption, "severity": severity, "split": SPLIT}
            for config, sc in scores.items():
                losses, exits = consistency_losses(logits, sc)
                k = ucb_index_fast(losses[:, cal], EPS, DELTA)
                result = measure(losses, exits, k, ev, fr, EPS)
                rows.append({**case, "config": config, "k": k, **result, "risk": float(result["risk"]),
                             "n_rows_paid": 4000 if config.split("_")[1:2] == ["adapt"] else 0})
                if not config.endswith("_cost"):  # the transform keeps the order within an exit
                    auc = {f"auroc_{l + 1}": roc_auc_score(agree[ev, l], sc[ev, l])
                           if 0 < agree[ev, l].mean() < 1 else np.nan for l in range(8)}
                    auc_rows.append({**case, "config": config, **auc})
    df = pd.DataFrame(rows).merge(pd.DataFrame(auc_rows), on=list(case) + ["config"], how="left")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path + ".tmp", index=False)
    os.replace(out_path + ".tmp", out_path)
    print(dataset, regime, seed, f"done in {time.time() - start:.1f} s", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--runs", nargs="*", default=None, help="dataset/regime/seedK, default all 24")
    parser.add_argument("--corruptions", nargs="*", default=None, help="only for a trial")
    parser.add_argument("--out-dir", default=HERE)
    args = parser.parse_args()
    corruptions = args.corruptions or list(CORRUPTIONS)  # passed in the job: the workers are spawned, not forked
    runs = all_runs()
    if args.runs:
        runs = [(r.split("/")[0], r.split("/")[1], int(r.split("/")[2].removeprefix("seed"))) for r in args.runs]
    jobs = []
    for dataset, regime, seed in runs:
        path = os.path.join(args.out_dir, "scores", f"{dataset}_{regime}_seed{seed}.csv")
        if not os.path.exists(path):
            jobs.append(((dataset, regime, seed), path, corruptions))
    print(len(jobs), "networks to do", flush=True)
    start = time.time()
    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(one_network, jobs, chunksize=1):
            pass
    print(f"done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
