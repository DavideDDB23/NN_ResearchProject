# Exit scores for the shared threshold of Fast yet Safe, computed from the logits of every exit (N, L, C).
#   max_softmax  top softmax probability, as in Fast yet Safe
#   temperature  top softmax probability after per-exit temperature scaling (Guo et al. 2017), T_l fitted on val
#   label        learned P(exit l is correct), the target of the confidence of Pomponi et al. 2022
#   agreement    learned P(exit l agrees with exit 9), the target of the consistency risk; it needs no labels
# The learned scores are one logistic regression per exit on three statistics of the head output (log top
# probability, margin between the two top probabilities, entropy). A linear unit on the pooled features alone
# separates correct and wrong predictions poorly (scripts/4_export/summarize_networks.py), and in a check on one network
# adding the features to these statistics did not raise the AUROC.
# The last exit always gets score 1: an input that reaches it exits there anyway.

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eesavings.export import confidence_targets
from eesavings.fys import softmax

SCORES = ["max_softmax", "temperature", "label", "agreement"]


def output_statistics(logits):
    # (N, C) logits of one exit -> (N, 3): log top probability, top1 - top2 probability, entropy
    p = softmax(logits, axis=1)
    top = np.sort(p, axis=1)
    entropy = -np.sum(p * np.log(p + 1e-12), axis=1)
    return np.stack([np.log(top[:, -1] + 1e-12), top[:, -1] - top[:, -2], entropy], axis=1)


def fit_temperatures(logits, labels):
    # one temperature per exit (exits 1..L-1), minimizing the negative log likelihood on the labelled val set
    temperatures = []
    for l in range(logits.shape[1] - 1):
        z = logits[:, l].astype(np.float64)

        def nll(log_t):
            p = softmax(z / np.exp(log_t), axis=1)
            return -np.mean(np.log(p[np.arange(len(labels)), labels] + 1e-12))

        temperatures.append(float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)))
    return np.array(temperatures)


def fit_learned(logits, targets, seed=0):
    # one logistic regression per exit (L2 chosen by 5-fold cross validation on the log loss);
    # targets (N, L-1) binary, e.g. from export.confidence_targets
    models = []
    for l in range(logits.shape[1] - 1):
        assert 0 < targets[:, l].mean() < 1, "the target must contain both classes"
        clf = LogisticRegressionCV(Cs=10, l1_ratios=(0,), cv=5, scoring="neg_log_loss", max_iter=5000,
                                   random_state=seed, use_legacy_attributes=False)
        models.append(make_pipeline(StandardScaler(), clf).fit(output_statistics(logits[:, l]), targets[:, l]))
    return models


def fit_scores(logits, labels):
    # parameters of the scores that need fitting, from the val set of a network
    return {
        "temperature": fit_temperatures(logits, labels),
        "label": fit_learned(logits, confidence_targets(logits, labels, "label")),
        "agreement": fit_learned(logits, confidence_targets(logits, labels, "agreement")),
    }


def exit_scores(logits, name, params=None):
    # (N, L, C) logits -> (N, L) scores in [0, 1], last column 1
    n, n_exits, _ = logits.shape
    scores = np.ones((n, n_exits))
    for l in range(n_exits - 1):
        if name == "max_softmax":
            scores[:, l] = softmax(logits[:, l], axis=1).max(axis=1)
        elif name == "temperature":
            scores[:, l] = softmax(logits[:, l] / params["temperature"][l], axis=1).max(axis=1)
        elif name in ("label", "agreement"):
            scores[:, l] = params[name][l].predict_proba(output_statistics(logits[:, l]))[:, 1]
        else:
            raise ValueError(name)
    return scores
