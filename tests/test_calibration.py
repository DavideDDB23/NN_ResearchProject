import numpy as np

from eesavings.fys import crc_lam, find_first_gap, losses_and_exits, rc_main, ucb_lam, ucb_wsr

LAMBDAS = np.round(np.arange(0.0, 1.001, 0.01)[::-1], 2)


def test_find_first_gap():
    assert find_first_gap(np.array([])) == 0
    assert find_first_gap(np.array([0, 1, 2, 5, 6])) == 2
    assert find_first_gap(np.array([0, 1, 2, 3])) == 3
    assert find_first_gap(np.array([2, 3])) == 0


def test_crc_picks_the_last_threshold_that_satisfies_its_rule(logits_targets):
    logits, targets = logits_targets
    losses, _ = losses_and_exits(logits, targets, LAMBDAS, "prediction-consistency")
    cal = losses[:, :200]
    n = cal.shape[1]
    for eps in [0.05, 0.1, 0.2]:
        k = crc_lam(cal, eps)
        rhs = (n + 1) * eps / n - 1.0 / n
        ok = [j for j in range(len(cal)) if cal[j].mean() <= rhs]
        assert k == (max(ok) if ok else 0)
    # on these logits early exits are possible, so lambda = 1 (index 0) is not enough
    assert crc_lam(cal, 0.1) > 0


def test_ucb_picks_the_last_threshold_before_the_first_failure(logits_targets):
    logits, targets = logits_targets
    losses, _ = losses_and_exits(logits, targets, LAMBDAS, "prediction-consistency")
    cal = losses[:, :200]
    ucbs = np.array([ucb_wsr(cal[j], delta=0.1) for j in range(len(cal))])
    for eps in [0.05, 0.1, 0.2]:
        k = ucb_lam(cal, eps, delta=0.1)
        if ucbs[0] < eps:
            assert np.all(ucbs[:k + 1] < eps)  # k and every larger threshold pass
            assert k == len(ucbs) - 1 or ucbs[k + 1] >= eps  # the next one fails
        else:
            assert k == 0
    assert ucb_lam(cal, 0.1, delta=0.1) > 0


def test_ucb_stops_at_the_first_threshold_that_fails():
    # thresholds 0 and 1 are safe, 2 is not, 3 is safe again: UCB must stop at 1
    safe, unsafe = np.zeros(200), np.ones(200)
    cal = np.array([safe, safe, unsafe, safe])
    assert ucb_lam(cal, 0.1, delta=0.1) == 1


def test_crc_controls_the_test_risk_on_average(logits_targets):
    logits, targets = logits_targets
    losses, exits = losses_and_exits(logits, targets, LAMBDAS, "prediction-consistency")
    eps_grid = np.array([0.05, 0.1, 0.2])
    test_risk, _, _ = rc_main(losses, exits.mean(axis=1), eps_grid, ["crc"], n_trials=50, n_cal=100)
    mean_risk = np.array(test_risk["crc"]).mean(axis=0)
    assert np.all(mean_risk <= eps_grid + 0.01)
