# Calibration of the shared exit threshold with risk control.
# rc_main and the *_lam functions are copied from Fast yet Safe
# (metodj/RC-EENN, rc/risk_control.py, commit 3db9a52), so results match the original.
#
# Convention: losses has shape (K, n), row k is the k-th threshold of a grid sorted in
# DESCENDING order (row 0 is lambda = 1, i.e. no early exit). Every function returns
# the index of the chosen threshold in that grid.

import numpy as np

from eesavings.risk.bounds import hb_p_value, ucb_hb, ucb_wsr


def rc_main(losses, exits, eps_grid, rcp_types, binary_loss=False, loss_bound=1,
            n_trials=5, n_cal=100, delta=0.1, seed=42):
    # For each trial: random calibration/test split, choose lambda on calibration,
    # measure the risk on test. The efficiency is exits[lambda], whatever the caller
    # passes: in our script it is the mean exit over ALL inputs (as in the original).
    # The third output only holds the choices of the last trial.
    np.random.seed(seed)

    for x in rcp_types:
        assert x in ["naive", "ltt", "ucb-wsr", "crc"]

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
                elif rcp == "ltt":
                    lam_id = ltt_lam(np.maximum(cal_losses, 0.0), eps, delta, binary_loss)
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
    # "ltt" is left out on purpose (see the note on ltt_lam).
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


def ltt_lam(losses, epsilon, delta, binary_loss):
    # Learn then Test with Hoeffding Bentkus p-values, copied as is and NOT used by us.
    # Careful: it returns the first index with p-value > delta, i.e. the first threshold
    # that is NOT certified (one step too far, same as in the original code).
    # -1 means that every threshold is certified, so the last one (lambda = 0) is chosen.
    K, n_cal = losses.shape
    risk = losses.mean(axis=1)

    p_vals = []
    for k in range(K):
        p_vals.append(hb_p_value(risk=risk[k], n=n_cal, alpha=epsilon, binary_loss=binary_loss).item())

    lams = (np.array(p_vals) > delta).nonzero()[0]
    if len(lams) == 0:
        return -1
    return lams.min()


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
