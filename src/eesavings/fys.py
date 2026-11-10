# Fast yet Safe (Jazbec et al. 2024): risk control of an early exit network with one shared threshold.
# Copied from the original code (metodj/RC-EENN, commit 3db9a52: img_cls/utils.py, rc/risk_control.py and
# rc/bounds.py) with the same function names, so the results match the original exactly
# (tests/test_upstream_equivalence.py). Learn then Test is left out because we never use it.
#
# Conventions:
#   logits (N, L, C), confidences (N, L); exits are numbered 1..L, L is the final exit.
#   An input leaves at the first exit whose confidence is strictly above the threshold lambda.
#   losses (K, N): row k is the k-th lambda of a grid sorted in DESCENDING order (row 0 is lambda = 1, i.e. no
#   early exit); the calibration functions return the index of the chosen lambda in that grid.
#
# Sections: 1) exit rule, 2) losses, 3) concentration bounds, 4) calibration of lambda, 5) losses and exits
# for every lambda of a grid.

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

from eesavings.data.fys_logits import read_img_cls_data


# ---------------- 1) exit rule ----------------

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


# ---------------- 2) losses with respect to the final exit ----------------
# Our main loss is "prediction-consistency": 1 if the exit used predicts a different class from the
# final exit, 0 otherwise. It needs no labels.

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


# ---------------- 3) concentration bounds (Fast yet Safe adapts them from aangelopoulos/rcps) ----------------

def ucb_wsr(x, delta, maxiters=1000, B=1, eps=1e-10):
    # upper confidence bound on the mean of x (Waudby-Smith Ramdas), valid with prob. 1 - delta
    n = x.shape[0]
    muhat = (np.cumsum(x) + 0.5) / (1 + np.array(range(1, n + 1)))
    sigma2hat = (np.cumsum((x - muhat) ** 2) + 0.25) / (1 + np.array(range(1, n + 1)))
    sigma2hat[1:] = sigma2hat[:-1]
    sigma2hat[0] = 0.25
    nu = np.minimum(np.sqrt(2 * np.log(1 / delta) / n / sigma2hat), 1 / B)

    def _Kn(mu):
        return np.max(np.cumsum(np.log(1 - nu * (x - mu)))) + np.log(delta)

    if _Kn(1) < 0:
        return B
    if _Kn(eps) > 0:
        return eps
    return brentq(_Kn, eps, 1 - eps, maxiter=maxiters)


def ucb_hb(risk, delta, n_cal, binary_loss, step=0.01):
    # upper confidence bound from the Hoeffding Bentkus p-value, on a grid of risk levels
    alphas = np.arange(0.01, 1.0 + step, step)[::-1]
    for i in range(len(alphas)):
        if hb_p_value(risk=risk, n=n_cal, alpha=alphas[i], binary_loss=binary_loss) >= delta:
            return alphas[i]
    return 0.0


def hb_p_value(risk, n, alpha=0.05, eps=1e-3, binary_loss=False):
    # p-value of the Hoeffding Bentkus test of "true risk >= alpha"
    if binary_loss:
        p_value = binom.cdf(np.ceil(n * risk), n, alpha)
    else:
        bentkus_p_value = np.e * binom.cdf(np.ceil(n * risk), n, alpha)
        a, b = min(risk, alpha), alpha
        h1 = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
        hoeffding_p_value = np.exp(-n * h1)
        p_value = min(bentkus_p_value, hoeffding_p_value)

    assert 0 - eps <= p_value <= 1 + eps, "p-value must be in [0, 1]: {}".format(p_value)
    return p_value


# ---------------- 4) calibration of the shared threshold ----------------

def rc_main(losses, exits, eps_grid, rcp_types, binary_loss=False, loss_bound=1,
            n_trials=5, n_cal=100, delta=0.1, seed=42):
    # For each trial: random calibration/test split, choose lambda on calibration,
    # measure the risk on test. The efficiency is exits[lambda], whatever the caller
    # passes: in our script it is the mean exit over ALL inputs (as in the original).
    # The third output only holds the choices of the last trial.
    np.random.seed(seed)

    for x in rcp_types:
        assert x in ["naive", "ucb-wsr", "crc"]

    assert losses.shape[0] == exits.shape[0]
    _, N = losses.shape

    test_risk, eff_gains = {r: [] for r in rcp_types}, {r: [] for r in rcp_types}

    for _ in range(n_trials):
        cal_ids = np.random.choice(N, n_cal, replace=False)
        test_ids = np.setdiff1d(np.arange(N), cal_ids)
        cal_losses, test_losses = losses[:, cal_ids], losses[:, test_ids]

        # stage 1: choose lambda on the calibration split
        rcp_lams = {r: [] for r in rcp_types}
        for rcp in rcp_types:
            for eps in eps_grid:
                if rcp == "naive":
                    lam_id = naive_lam(cal_losses, eps)
                elif rcp == "ucb-wsr":
                    lam_id = ucb_lam(cal_losses, eps, delta, ucb_type="wsr", binary_loss=False,
                                     loss_bound=loss_bound)
                elif rcp == "crc":
                    lam_id = crc_lam(cal_losses, eps, loss_bound=loss_bound)
                rcp_lams[rcp].append(lam_id)

        # stage 2: evaluate the chosen lambda on the test split
        for rcp in rcp_types:
            test_risk_e, eff_gains_e = [], []
            for e, eps in enumerate(eps_grid):
                lam_id = rcp_lams[rcp][e]
                test_risk_e.append(test_losses[lam_id].mean())
                eff_gains_e.append(exits[lam_id])
            test_risk[rcp].append(test_risk_e)
            eff_gains[rcp].append(eff_gains_e)

    return test_risk, eff_gains, rcp_lams


def rc_main_fast(losses, exits, eps_grid, rcp_types, loss_bound=1,
                 n_trials=5, n_cal=100, delta=0.1, seed=42):
    # Same as rc_main (same splits, same choices) but faster: the UCB of each lambda does
    # not depend on eps, so we compute it once per split instead of once per eps.
    # It also returns the chosen indices of every trial, lam_ids[rcp] of shape (n_trials, n_eps).
    for rcp in rcp_types:
        assert rcp in ["naive", "ucb-wsr", "crc"], rcp
    assert losses.shape[0] == exits.shape[0]

    np.random.seed(seed)
    K, N = losses.shape

    test_risk = {r: [] for r in rcp_types}
    eff_gains = {r: [] for r in rcp_types}
    lam_ids = {r: [] for r in rcp_types}

    for _ in range(n_trials):
        cal_ids = np.random.choice(N, n_cal, replace=False)
        test_ids = np.setdiff1d(np.arange(N), cal_ids)
        cal_losses = losses[:, cal_ids]
        test_means = losses[:, test_ids].mean(axis=1)

        if "ucb-wsr" in rcp_types:
            ucbs = np.array([ucb_wsr(cal_losses[k], delta=delta, B=loss_bound) for k in range(K)])

        for rcp in rcp_types:
            ids = []
            for eps in eps_grid:
                if rcp == "naive":
                    ids.append(naive_lam(cal_losses, eps))
                elif rcp == "ucb-wsr":
                    ids.append(find_first_gap((ucbs < eps).nonzero()[0]))
                elif rcp == "crc":
                    ids.append(crc_lam(cal_losses, eps, loss_bound=loss_bound))
            test_risk[rcp].append([test_means[i] for i in ids])
            eff_gains[rcp].append([exits[i] for i in ids])
            lam_ids[rcp].append(ids)

    last_lams = {r: lam_ids[r][-1] for r in rcp_types}
    lam_ids = {r: np.array(v) for r, v in lam_ids.items()}
    return test_risk, eff_gains, last_lams, lam_ids


def naive_lam(losses, epsilon):
    # largest index with empirical risk below eps (no guarantee)
    risk = losses.mean(axis=1)
    lams = (risk < epsilon).nonzero()[0]
    if len(lams) == 0:
        return 0
    return lams.max()


def crc_lam(losses, epsilon, loss_bound=1.0):
    # Conformal Risk Control: risk below eps in expectation
    _, n_cal = losses.shape
    risk = losses.mean(axis=1)
    ucb = (n_cal + 1) * epsilon / n_cal - loss_bound / n_cal
    lams = (risk <= ucb).nonzero()[0]
    if len(lams) == 0:
        return 0
    return lams.max()


def ucb_lam(losses, epsilon, delta, ucb_type="wsr", binary_loss=False, loss_bound=1):
    # UCB calibration: smallest lambda such that the UCB of it and of every larger
    # lambda is below eps; risk below eps with probability 1 - delta
    K, n_cal = losses.shape

    ucb = []
    for k in range(K):
        if ucb_type == "wsr":
            ucb.append(ucb_wsr(losses[k], delta=delta, B=loss_bound))
        elif ucb_type == "hb":
            ucb.append(ucb_hb(risk=losses[k].mean(), delta=delta, n_cal=n_cal, binary_loss=binary_loss))
        else:
            raise ValueError(f"Invalid UCB type: {ucb_type}")

    lams = (np.array(ucb) < epsilon).nonzero()[0]
    return find_first_gap(lams)


def find_first_gap(arr):
    # last index of the initial run 0, 1, 2, ... of a sorted array of indices
    if len(arr) == 0:
        return 0
    for i in range(len(arr)):
        if arr[i] != i:
            return max(0, i - 1)
    return len(arr) - 1


# ---------------- 5) losses and exits for every lambda of a grid ----------------
# Adapted from img_cls_main: the logits are passed directly, so the same code works on the ImageNet
# pickles and on our CIFAR networks, and conf (N, L) lets us pass our own exit scores.

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
