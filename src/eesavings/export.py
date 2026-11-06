# Run a trained multi exit network on a set of images, and fit its confidence units.
#
# Confidence units: one logistic regression per exit (1..8) on the features of the pooled head, fitted on the
# val set with the network frozen. With target "label" the unit predicts whether the exit is correct
# (a post hoc approximation of the learned confidence of Pomponi et al. 2022, who train it with the network and
# with a class balanced loss); with target "agreement" it predicts whether the exit agrees with exit 9.
# We use the plain (not class balanced) loss: Fast yet Safe uses one threshold for all exits, and class balancing
# would shift the scores of each exit by a different amount.

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from eesavings.data.cifar import to_tensor_batch
from eesavings.models import FEATURE_DIMS, N_EXITS


def predict(model, images, dataset, device, batch_size=1000, return_features=False):
    # images: uint8 (N, 32, 32, 3). Returns logits (N, 9, C) float32, confidences (N, 9) float32
    # and, if asked, the head features (N, 4352) float32 (exits 1..8 one after the other).
    model.eval()
    logits, confs, feats = [], [], []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = to_tensor_batch(images[i:i + batch_size], dataset).to(device)
            if return_features:
                z, c, f = model(x, return_features=True)
                feats.append(torch.cat(f, dim=1).cpu())
            else:
                z, c = model(x)
            logits.append(z.cpu())
            confs.append(c.cpu())
    out = [torch.cat(logits).numpy(), torch.cat(confs).numpy()]
    if return_features:
        out.append(torch.cat(feats).numpy())
    return tuple(out)


def split_features(features):
    # (N, 4352) -> list of the 8 feature blocks, one per exit
    ends = np.cumsum(FEATURE_DIMS)
    return np.split(features, ends[:-1], axis=1)


def confidence_targets(logits, labels, target):
    # (N, 8) binary targets: exit l correct ("label") or in agreement with exit 9 ("agreement")
    preds = logits.argmax(axis=2)
    if target == "label":
        return (preds[:, :-1] == labels[:, None]).astype(int)
    if target == "agreement":
        return (preds[:, :-1] == preds[:, -1:]).astype(int)
    raise ValueError(target)


def fit_confidence(features, targets, n_folds=5, seed=0):
    # one logistic regression per exit, L2 strength chosen by cross validation on the log loss.
    # Returns the weights (list of arrays) and biases (list of floats) of the 8 confidence units.
    weights, biases = [], []
    for f, t in zip(split_features(features.astype(np.float64)), targets.T):
        assert 0 < t.mean() < 1, "the target must contain both classes"
        scaler = StandardScaler().fit(f)
        clf = LogisticRegressionCV(Cs=10, l1_ratios=(0,), cv=n_folds, scoring="neg_log_loss", max_iter=5000,
                                   random_state=seed, use_legacy_attributes=False)  # l1_ratios=(0,) is plain L2
        clf.fit(scaler.transform(f), t)
        # fold the standardization into the linear unit: w'x + b' = w (x - mean) / scale + b
        w = clf.coef_[0] / scaler.scale_
        b = clf.intercept_[0] - np.sum(clf.coef_[0] * scaler.mean_ / scaler.scale_)
        weights.append(w)
        biases.append(b)
    return weights, biases


def set_confidence(model, weights, biases):
    # copy the fitted logistic regressions into the confidence units of the heads
    with torch.no_grad():
        for l in range(N_EXITS - 1):
            unit = model.heads[l].confidence
            unit.weight.copy_(torch.as_tensor(weights[l], dtype=unit.weight.dtype).view_as(unit.weight))
            unit.bias.fill_(float(biases[l]))
    return model
