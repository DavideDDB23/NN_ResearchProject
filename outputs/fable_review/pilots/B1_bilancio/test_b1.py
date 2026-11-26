# Tests of pilot B1 (b1_levels.py): the rule with 8 thresholds and the input oracle with eps, on random and hand
# made problems (no logits needed).
#   (a) policy_exits with all thresholds equal to lambda gives the exits of eesavings.fys.compute_exits
#   (b) coordinate_step reaches the minimum of the Lagrangian over every threshold of that exit (brute force)
#   (c) fit_exit_thresholds is feasible and not worse than the best shared threshold
#   (d) oracle_input_eps on a hand made example
#
# Usage (from the root of the repository):
#   python -m pytest outputs/fable_review/pilots/B1_bilancio/test_b1.py -q

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b1_levels import (LAMBDAS, coordinate_step, first_agreeing_exit, fit_exit_path, fit_exit_thresholds,  # noqa: E402
                       oracle_input_eps_exits, policy_exits)
from eesavings.fys import compute_exits  # noqa: E402
from review_common import fractions_of  # noqa: E402

FR = fractions_of("cifar10")


def random_problem(rng, n, ties=False):
    # confidences of exits 1..8 and disagreement with exit 9, more likely at low confidence and at early exits
    conf8 = rng.uniform(0.05, 1.0, size=(n, 8))
    if ties:
        conf8 = np.clip(np.round(conf8, 1), 0.1, 1.0)  # many equal confidences, some exactly 1
    loss9 = np.zeros((n, 9), dtype=bool)
    loss9[:, :8] = rng.uniform(size=(n, 8)) < (1 - conf8) * np.linspace(0.9, 0.2, 8)
    return conf8, loss9


def lagrangian(conf8, loss9, t, mu):
    exits = policy_exits(conf8, t)
    return FR[exits].mean() + mu * loss9[np.arange(len(exits)), exits].mean()


def with_threshold(t, l, value):
    t = t.copy()
    t[l] = value
    return t


def test_policy_exits_matches_compute_exits():
    rng = np.random.default_rng(0)
    conf = np.ones((2000, 9))
    conf[:, :8] = rng.uniform(size=(2000, 8))
    conf[:100, :8] = np.round(conf[:100, :8], 1)  # ties and confidences exactly 0 or 1
    for lam in list(LAMBDAS[::10]) + [1.0, 0.5, 0.0] + list(rng.uniform(size=10)):
        assert np.array_equal(policy_exits(conf[:, :8], np.full(8, lam)) + 1, compute_exits(conf, lam))


@pytest.mark.parametrize("seed", range(20))
def test_coordinate_step_is_exact(seed):
    rng = np.random.default_rng(seed)
    conf8, loss9 = random_problem(rng, 40, ties=seed % 2 == 0)
    t = rng.uniform(0.3, 1.0, size=8)
    mu = [0.2, 1.0, 5.0, 50.0][seed % 4]
    for l in range(8):
        new = coordinate_step(l, conf8, loss9, FR, t, mu)
        # every set of rows that a threshold of exit l can make leave: above each confidence, or all of them
        candidates = np.concatenate([[1.0, 0.0], np.unique(conf8[:, l])])
        brute = min(lagrangian(conf8, loss9, with_threshold(t, l, c), mu) for c in candidates)
        assert lagrangian(conf8, loss9, with_threshold(t, l, new), mu) == pytest.approx(brute, abs=1e-12)
        t[l] = new


def best_shared_cost(conf8, loss9, eps):
    # lowest mean cost of a shared threshold of the grid with mean loss <= eps, through eesavings.fys.compute_exits
    conf = np.concatenate([conf8, np.ones((len(conf8), 1))], axis=1)
    rows, best = np.arange(len(conf8)), np.inf
    for lam in LAMBDAS:
        exits = compute_exits(conf, lam) - 1
        if loss9[rows, exits].mean() <= eps:
            best = min(best, FR[exits].mean())
    return best


@pytest.mark.parametrize("seed", range(8))
def test_fit_exit_thresholds_feasible_and_not_worse(seed):
    rng = np.random.default_rng(100 + seed)
    conf8, loss9 = random_problem(rng, 2000, ties=seed % 4 == 3)
    eps = [0.05, 0.02, 0.1, 0.05][seed % 4]
    t = fit_exit_thresholds(conf8, loss9, FR, eps)
    exits = policy_exits(conf8, t)
    assert loss9[np.arange(len(exits)), exits].mean() <= eps + 1e-12
    assert FR[exits].mean() <= best_shared_cost(conf8, loss9, eps) + 1e-12


@pytest.mark.parametrize("seed", range(6))
def test_fit_exit_path_is_feasible_on_cal(seed):
    # added by Fable: the path is fitted on one set of rows and stopped on another; on the second set the rule
    # returned must have mean loss <= eps, and it must not cost more than sending everybody to exit 9
    rng = np.random.default_rng(200 + seed)
    conf_a, loss_a = random_problem(rng, 1500, ties=seed % 3 == 2)
    conf_c, loss_c = random_problem(rng, 1000, ties=seed % 3 == 2)
    eps = [0.05, 0.02, 0.1][seed % 3]
    t = fit_exit_path(conf_a, loss_a, conf_c, loss_c, FR, eps)
    exits = policy_exits(conf_c, t)
    assert loss_c[np.arange(len(exits)), exits].mean() <= eps + 1e-12
    assert FR[exits].mean() <= FR[8] + 1e-12


def test_oracle_input_eps_hand_made():
    # first agreeing exits (0-based) of 10 rows; exits after the first agreeing one are irrelevant
    first = np.array([0, 3, 8, 1, 3, 0, 8, 2, 5, 1])
    loss9 = np.zeros((10, 9), dtype=bool)
    for row, a in enumerate(first):
        loss9[row, :a] = True
        loss9[row, a + 1:8] = row % 2 == 0
    assert np.array_equal(first_agreeing_exit(loss9), first)
    rows = np.arange(10)

    # eps 0.2: 2 rows move, the two rows whose first agreeing exit is exit 9, in their order
    exits, n_moved = oracle_input_eps_exits(first, FR, 0.2)
    assert n_moved == 2 and np.array_equal(exits, [0, 3, 0, 1, 3, 0, 0, 2, 5, 1])
    assert loss9[rows, exits].mean() == pytest.approx(0.2)
    assert 1 - FR[exits].mean() == pytest.approx(1 - FR[[0, 3, 0, 1, 3, 0, 0, 2, 5, 1]].mean())

    # eps 0.3: then the row with exit 6 (index 5)
    exits, n_moved = oracle_input_eps_exits(first, FR, 0.3)
    assert n_moved == 3 and np.array_equal(exits, [0, 3, 0, 1, 3, 0, 0, 2, 0, 1])

    # eps 0.4: 4 rows; of the two rows with exit 4 (index 3) only the first one moves (stable order)
    exits, n_moved = oracle_input_eps_exits(first, FR, 0.4)
    assert n_moved == 4 and np.array_equal(exits, [0, 0, 0, 1, 3, 0, 0, 2, 0, 1])

    # eps 0.5: 5 rows, both rows with exit 4
    exits, n_moved = oracle_input_eps_exits(first, FR, 0.5)
    assert n_moved == 5 and np.array_equal(exits, [0, 0, 0, 1, 0, 0, 0, 2, 0, 1])

    # eps 1.0: only the 8 rows whose first agreeing exit is not exit 1 can move
    exits, n_moved = oracle_input_eps_exits(first, FR, 1.0)
    assert n_moved == 8 and np.array_equal(exits, np.zeros(10))

