# Exit decision with one confidence threshold shared by all exits.
# Copied from Fast yet Safe (metodj/RC-EENN, img_cls/utils.py, commit 3db9a52).
# Shapes: logits (N, L, C), confidences (N, L). Exits are numbered 1..L, L is the final exit.

import numpy as np


def softmax(x, axis=None):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def conf_measures(input_tensor, input_type="logits", conf_measure="top_softmax"):
    # confidence of every exit, (N, L, C) -> (N, L)
    assert len(input_tensor.shape) == 3
    assert input_type in ["logits", "probs"]

    if conf_measure == "top_softmax":
        if input_type == "logits":
            input_tensor = softmax(input_tensor, axis=2)
        conf = np.max(input_tensor, axis=2)
    else:
        raise ValueError(f"conf_measure: {conf_measure} not implemented")

    return conf


def compute_exits(conf, threshold):
    # first exit whose confidence is strictly above the threshold, otherwise the final exit
    assert len(conf.shape) == 2
    assert 0 <= threshold <= 1

    mask = (conf > threshold).astype(np.int32)
    exits = np.argmax(mask, axis=1)
    rows_with_no_threshold = np.sum(mask, axis=1) == 0
    exits[rows_with_no_threshold] = mask.shape[1] - 1
    exits = exits + 1

    assert exits.shape == (conf.shape[0],)
    assert np.all(exits >= 1) and np.all(exits <= conf.shape[1])

    return exits


def get_preds_per_exit(logits):
    L = logits.shape[1]
    preds = np.argmax(logits, axis=2)
    return {l: preds[:, l] for l in range(L)}


def get_acc_per_exit(preds, targets):
    L = len(preds)
    return [(targets == preds[l]).sum() / len(targets) for l in range(L)]
