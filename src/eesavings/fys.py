# Losses and exits for every threshold of a grid, starting from the logits of all exits.
# Adapted from img_cls_main in Fast yet Safe (metodj/RC-EENN, img_cls/utils.py, commit 3db9a52).
# Changes: the logits are passed directly, so the same code works on the ImageNet pickles
# and on our CIFAR networks, and `conf` (N, L) lets us pass our own exit scores
# (for example a learned confidence) instead of the top softmax probability.

import numpy as np

from eesavings.data.fys_logits import read_img_cls_data
from eesavings.risk.exits import compute_exits, conf_measures, softmax
from eesavings.risk.losses import compute_risk


def losses_and_exits(logits, targets, lambdas, risk_type, non_negative=False, conf=None):
    # logits: (N, L, C), lambdas in descending order
    # returns losses (K, N) and exits (K, N), with K = len(lambdas)
    if conf is None:
        conf = conf_measures(logits)  # top softmax probability of each exit
    preds = np.argmax(logits, axis=2)
    preds_full = preds[:, -1]
    probs_full = softmax(logits[:, -1], axis=1)

    lam_losses, lam_exits = [], []
    for lam in lambdas:
        exits = compute_exits(conf, threshold=lam)

        if "confidence" in risk_type:
            logits_ee = logits[np.arange(len(logits)), exits - 1]
            outputs_early = softmax(logits_ee, axis=1)
            outputs_full = probs_full
        else:
            outputs_early = preds[np.arange(len(preds)), exits - 1]
            outputs_full = preds_full

        gt_labels = targets if risk_type in {"prediction-gt-gap", "confidence-brier"} else None
        _, losses = compute_risk(outputs_early, outputs_full, criterion=risk_type,
                                 gt_labels=gt_labels, non_negative=non_negative)

        lam_losses.append(losses)
        lam_exits.append(exits)

    return np.array(lam_losses), np.array(lam_exits)


def img_cls_main(model, dataset, path, lambdas, risk_type, non_negative=False):
    # same interface as the original function
    logits, targets = read_img_cls_data(path, model, dataset)
    return losses_and_exits(logits, targets, lambdas, risk_type, non_negative=non_negative)
