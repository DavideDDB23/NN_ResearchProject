# Losses of the early exit prediction with respect to the full network.
# Copied from Fast yet Safe (metodj/RC-EENN, img_cls/utils.py, commit 3db9a52).
# Our main one is "prediction-consistency": 1 if the exit used predicts a different class
# from the final exit, 0 otherwise. It does not need labels.

import numpy as np


def compute_risk(output_early, output_full, criterion, gt_labels=None, non_negative=True):
    assert output_early.shape == output_full.shape, "Prediction shapes must match."
    if criterion in ["confidence-brier", "confidence-hellinger", "confidence-brier-top-pred"]:
        assert output_early.ndim == 2, "Prediction must be 2D (N, C)."
    else:
        assert output_early.ndim == 1, "Prediction must be 1D (N, )."
    if "confidence" in criterion:
        assert 0 <= output_full.min() and output_full.max() <= 1
        assert 0 <= output_early.min() and output_early.max() <= 1

    if criterion == "prediction-consistency":
        loss = 1.0 - (output_full == output_early).astype(np.float32)

    elif criterion == "prediction-gt-gap":
        assert gt_labels is not None, "Ground truth labels must be provided."
        loss = (output_early != gt_labels).astype(np.float32) - (output_full != gt_labels).astype(np.float32)

    elif criterion == "confidence-hellinger":
        loss = hellinger_distance(output_early, output_full)

    elif criterion == "confidence-brier":
        assert gt_labels is not None, "Ground truth labels must be provided."
        loss = brier_score(output_early, gt_labels) - brier_score(output_full, gt_labels)

    elif criterion == "confidence-brier-top-pred":
        top = output_full.argmax(axis=1)
        loss = brier_score(output_early, top) - brier_score(output_full, top)

    else:
        raise ValueError("Invalid risk measure.")

    if non_negative:
        loss = np.maximum(loss, 0.0)

    risk = loss.mean()
    assert risk <= 1

    return risk, loss


def brier_score(conf, labels, mean=False, C=1.0):
    assert len(conf.shape) == 2
    assert len(labels.shape) == 1
    assert conf.shape[0] == labels.shape[0]

    labels_one_hot = np.eye(conf.shape[1])[labels]

    if mean:
        brier = np.mean((conf - labels_one_hot) ** 2, axis=1)
    else:
        brier = np.sum((conf - labels_one_hot) ** 2, axis=1)
    return C * brier


def hellinger_distance(p, q):
    assert len(p.shape) == 2
    assert len(q.shape) == 2
    assert p.shape == q.shape

    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=1)) / np.sqrt(2)
