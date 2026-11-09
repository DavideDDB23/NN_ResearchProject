# Offline adaptation of the heads on unlabeled data of the new domain (method M8 of the project document).
# Backbone and exit 9 stay frozen and alpha of the pooled heads stays fixed, so the cached features of exits 1..8
# are enough: the linear layer of each head is re-fitted by distillation towards the softmax of exit 9 (the full
# network, which an audit runs anyway). No labels are used.
# With a few thousand images the 1024 x 100 layer of a late head overfits, so the fit starts from the trained
# weights, with an L2 penalty that pulls it back towards them; the strength is chosen for every head on held-out
# adaptation rows (the distillation loss there), from a small grid.

import numpy as np
import torch
import torch.nn.functional as F

from eesavings.export import split_features
from eesavings.models import FEATURE_DIMS

L2_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1]
MAX_ITER = 100


def split_by_exit(features):
    # (N, 4352) float16 or float32 -> list of the 8 float32 feature blocks, one per exit
    return split_features(np.asarray(features, dtype=np.float32))


def head_weights(state_dict):
    # weights and biases of the linear layers of the 8 heads, from a model.pt state_dict
    weights = [state_dict[f"heads.{l}.classifier.weight"].numpy() for l in range(len(FEATURE_DIMS))]
    biases = [state_dict[f"heads.{l}.classifier.bias"].numpy() for l in range(len(FEATURE_DIMS))]
    return weights, biases


def head_logits(features, weights, biases):
    # logits of exits 1..8 from the cached features: (N, 8, C)
    blocks = split_by_exit(features)
    return np.stack([f @ w.T + b for f, w, b in zip(blocks, weights, biases)], axis=1)


def distillation_loss(x, teacher, w, b):
    # KL(softmax of exit 9 || softmax of the head), mean over the rows
    return F.kl_div(F.log_softmax(x @ w.T + b, dim=1), teacher, reduction="batchmean")


def fit_head(x, teacher, w0, b0, l2, max_iter=MAX_ITER):
    # full batch L-BFGS on the distillation loss plus l2 * ||w - w0||^2
    w = torch.tensor(w0, requires_grad=True)
    b = torch.tensor(b0, requires_grad=True)
    w_start = torch.tensor(w0)
    optimizer = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = distillation_loss(x, teacher, w, b) + l2 * ((w - w_start) ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach(), b.detach()


def adapt_heads(train_features, train_exit9, val_features, val_exit9, weights, biases, l2_grid=L2_GRID):
    # re-fits every head on the train rows and keeps, for every head, the L2 strength with the lowest
    # distillation loss on the held-out val rows. Returns new weights, biases and the chosen strengths.
    t_train = F.softmax(torch.as_tensor(np.asarray(train_exit9), dtype=torch.float32), dim=1)
    t_val = F.softmax(torch.as_tensor(np.asarray(val_exit9), dtype=torch.float32), dim=1)
    new_weights, new_biases, chosen = [], [], []
    for x_tr, x_val, w0, b0 in zip(split_by_exit(train_features), split_by_exit(val_features), weights, biases):
        x_tr, x_val = torch.as_tensor(x_tr), torch.as_tensor(x_val)
        best = None
        for l2 in l2_grid:
            w, b = fit_head(x_tr, t_train, w0, b0, l2)
            with torch.no_grad():
                val_loss = distillation_loss(x_val, t_val, w, b).item()
            if best is None or val_loss < best[0]:
                best = (val_loss, w, b, l2)
        new_weights.append(best[1].numpy())
        new_biases.append(best[2].numpy())
        chosen.append(best[3])
    return new_weights, new_biases, chosen
