# Pilot B3, part 1 (hypothesis C1 of the plan v3): the drop of the savings under shift split into traffic (fewer
# inputs pass the threshold) and reliability (at the same scores, fewer inputs agree with exit 9). Written by Opus on
# 15 November 2026; design and predictions in outputs/fable_review/quaderno.md, entry B3.01.
# For every network, the clean test set P and every shifted domain Q (15 standard corruptions at severity 1, 3, 5):
#   g_l^D(x)  P(exit l disagrees with exit 9 | score history up to l), one logistic regression per exit l = 1..8 and
#             domain D on z(s_l), z(m_l), their squares and product, with s_l the max softmax of exit l, m_l the
#             largest score of exits 1..l-1 (an input reaches exit l only if m_l <= lambda) and z(s) = -log(1 - s).
#             Cross-fitted on two halves of the 10,000 rows: a row of D gets a prediction from the model fitted on the
#             other half; a model of B applied to another domain never saw its rows. Exit 9 always agrees.
#   R(A, B)(lam)  mean over the rows of A of g^B at the exit used at lam (scores of A, reliability of B)
#   S(A, lam)     savings of the scores of A at lam
#   lam(A, B)     the lambda of the grid with the largest S(A, lam) and R(A, B)(lam) <= eps (an oracle, no UCB)
# S_PP = S(P, lam(P, P)), S_QQ, S_QP = S(Q, lam(Q, P)) (traffic of Q, reliability of P), S_PQ = S(P, lam(P, Q)).
#   order 1: traffic S_PP - S_QP, reliability S_QP - S_QQ;  order 2: reliability S_PP - S_PQ, traffic S_PQ - S_QQ
# Checks for every domain: the largest gap along the grid between the model risk R(D, D) and the empirical risk, and
# the empirical oracle (lambda and savings). Also the AUROC of the max softmax of every exit for its agreement with
# exit 9 (all 10,000 rows), and the risk at the clean oracle lambda with the two reliabilities.
# Reads outputs/cifar only. Output: decomp/<dataset>_<regime>_seed<k>.csv in this folder (skipped if done).
#
# Usage (from the repository root):
#   python outputs/fable_review/pilots/B3_meccanismo/b3_decomp.py [--workers 3] [--runs cifar10/joint/seed0 ...]

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
from eesavings.analysis import LAMBDAS  # noqa: E402
from eesavings.data.cifar import CORRUPTIONS  # noqa: E402
from eesavings.scores import exit_scores  # noqa: E402
from review_common import all_runs, fractions_of, load_logits  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 0.05
SEVERITIES = [1, 3, 5]


def z(s):
    return -np.log(1.0 - s + 1e-7)


def features(s8, l):
    # features of the model of exit l (0-based) from the scores of exits 1..8
    zs = z(s8[:, l])
    if l == 0:
        return np.stack([zs, zs ** 2], axis=1)
    zm = z(s8[:, :l].max(axis=1))
    return np.stack([zs, zm, zs ** 2, zm ** 2, zs * zm], axis=1)


class Domain:
    # scores, disagreements, exits for every lambda, and the reliability models of one domain (logits (N, 9, C))
    def __init__(self, logits, folds, fr):
        self.s8 = exit_scores(logits, "max_softmax")[:, :8]
        preds = logits.argmax(axis=2)
        self.dis = (preds[:, :8] != preds[:, 8:9]).astype(float)                    # (N, 8)
        above = self.s8[None, :, :] > LAMBDAS[:, None, None]                         # (K, N, 8)
        self.exits = np.where(above.any(axis=2), above.argmax(axis=2), 8)            # (K, N), 0-based, 8 = exit 9
        self.savings = 1.0 - fr[self.exits].mean(axis=1)                             # (K,)
        dis9 = np.concatenate([self.dis, np.zeros((len(self.dis), 1))], axis=1)
        self.risk_emp = np.take_along_axis(dis9[None], self.exits[..., None], axis=2)[..., 0].mean(axis=1)
        self.folds = folds
        self.models = [[self.fit(l, fold) for fold in folds] for l in range(8)]  # models[l][f]: fitted on fold f

    def fit(self, l, rows):
        y = self.dis[rows, l]
        if y.min() == y.max():
            return float(y[0])  # one class only: a constant
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return make_pipeline(StandardScaler(), LogisticRegression(C=100.0, max_iter=2000)).fit(
                features(self.s8[rows], l), y)

    def predict(self, other):
        # (N_other, 9): P(disagreement) at every exit for the rows of `other`, with the models of this domain. A row
        # in fold f gets the model fitted on the other fold (cross-fitting when other is self)
        out = np.zeros((len(other.s8), 9))
        for f, rows in enumerate(other.folds):
            for l in range(8):
                model = self.models[l][1 - f]
                if isinstance(model, float):
                    out[rows, l] = model
                else:
                    out[rows, l] = model.predict_proba(features(other.s8[rows], l))[:, 1]
        return out


def risk_curve(scores_domain, reliability):
    # R(A, B)(lam) for every lambda: reliability (N, 9) from B's models on the rows of A
    return np.take_along_axis(reliability[None], scores_domain.exits[..., None], axis=2)[..., 0].mean(axis=1)


def oracle(savings, risk):
    # index of the lambda with the largest savings and risk <= eps (ties: the smallest index)
    feasible = risk <= EPS + 1e-12
    return int(np.argmax(np.where(feasible, savings, -np.inf)))


def auroc(domain):
    out = {}
    for l in range(8):
        agree = 1 - domain.dis[:, l]
        out[f"auroc_{l + 1}"] = roc_auc_score(agree, domain.s8[:, l]) if 0 < agree.mean() < 1 else np.nan
    return out


def one_network(job):
    (dataset, regime, seed), out_path = job
    start = time.time()
    fr = fractions_of(dataset)
    perm = np.random.default_rng(0).permutation(10000)
    folds = [np.sort(perm[:5000]), np.sort(perm[5000:])]
    P = Domain(load_logits(dataset, regime, seed), folds, fr)
    rel_pp = P.predict(P)
    r_pp = risk_curve(P, rel_pp)
    k_pp, k_pp_emp = oracle(P.savings, r_pp), oracle(P.savings, P.risk_emp)
    key = {"dataset": dataset, "regime": regime, "seed": seed}
    rows = [{**key, "corruption": "none", "severity": 0, "S_PP": P.savings[k_pp], "S_PP_emp": P.savings[k_pp_emp],
             "gap_P": np.abs(r_pp - P.risk_emp).max(), **auroc(P)}]
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            Q = Domain(load_logits(dataset, regime, seed, corruption, severity), folds, fr)
            rel_qq, rel_qp = Q.predict(Q), P.predict(Q)    # on the rows of Q: reliability of Q, of P
            rel_pq = Q.predict(P)                           # on the rows of P: reliability of Q
            r_qq, r_qp, r_pq = risk_curve(Q, rel_qq), risk_curve(Q, rel_qp), risk_curve(P, rel_pq)
            k_qq, k_qp, k_pq = oracle(Q.savings, r_qq), oracle(Q.savings, r_qp), oracle(P.savings, r_pq)
            k_qq_emp = oracle(Q.savings, Q.risk_emp)
            rows.append({**key, "corruption": corruption, "severity": severity,
                         "S_PP": P.savings[k_pp], "S_QQ": Q.savings[k_qq], "S_QP": Q.savings[k_qp],
                         "S_PQ": P.savings[k_pq], "S_PP_emp": P.savings[k_pp_emp], "S_QQ_emp": Q.savings[k_qq_emp],
                         "lam_PP": LAMBDAS[k_pp], "lam_QQ": LAMBDAS[k_qq], "lam_QP": LAMBDAS[k_qp],
                         "lam_PQ": LAMBDAS[k_pq], "gap_P": np.abs(r_pp - P.risk_emp).max(),
                         "gap_Q": np.abs(r_qq - Q.risk_emp).max(),
                         # at the clean oracle lambda: the risk of Q with its own and with the clean reliability
                         "risk_Q_at_lamPP_emp": Q.risk_emp[k_pp], "risk_QQ_at_lamPP": r_qq[k_pp],
                         "risk_QP_at_lamPP": r_qp[k_pp], "savings_Q_at_lamPP": Q.savings[k_pp], **auroc(Q)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path + ".tmp", index=False)
    os.replace(out_path + ".tmp", out_path)
    print(dataset, regime, seed, f"done in {time.time() - start:.1f} s", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--runs", nargs="*", default=None, help="dataset/regime/seedK, default all 24")
    parser.add_argument("--out-dir", default=HERE)
    args = parser.parse_args()
    runs = all_runs()
    if args.runs:
        runs = [(r.split("/")[0], r.split("/")[1], int(r.split("/")[2].removeprefix("seed"))) for r in args.runs]
    jobs = []
    for dataset, regime, seed in runs:
        path = os.path.join(args.out_dir, "decomp", f"{dataset}_{regime}_seed{seed}.csv")
        if not os.path.exists(path):
            jobs.append(((dataset, regime, seed), path))
    print(len(jobs), "networks to do", flush=True)
    start = time.time()
    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(one_network, jobs, chunksize=1):
            pass
    print(f"done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
