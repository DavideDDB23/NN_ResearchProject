# Read the precomputed logits released with Fast yet Safe.
# Copied from metodj/RC-EENN, img_cls/utils.py, commit 3db9a52.
# Each file is {path}/{model}/{dataset}.p and contains (logits, targets), logits of shape (L, N, C).

import pickle

import numpy as np


def read_img_cls_data(path, model, dataset):
    # returns logits (N, L, C) and integer targets (N,)
    with open(f"{path}/{model}/{dataset}.p", "rb") as f:
        data = pickle.load(f)

    if len(data) == 2:
        logits, targets = data
    else:
        logits, targets, _ = data

    logits = np.transpose(logits, (1, 0, 2))
    targets = targets.astype(int)

    return logits, targets

