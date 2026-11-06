import numpy as np

from eesavings.savings import (
    RESNET20_CIFAR_CUM_MMACS,
    cost_fractions,
    exit_distribution,
    per_exit_disagreement,
    savings_change_by_exit,
    savings_from_exits,
    uniform_cost_fractions,
)


def test_resnet20_fractions_match_the_project_document():
    f = cost_fractions(RESNET20_CIFAR_CUM_MMACS)
    expected = [0.13, 0.24, 0.36, 0.45, 0.56, 0.68, 0.77, 0.88, 1.00]
    assert np.allclose(np.round(f, 2), expected)


def test_savings_of_known_exits():
    f = cost_fractions(RESNET20_CIFAR_CUM_MMACS)
    assert savings_from_exits(np.array([9, 9, 9]), f) == 0.0
    assert np.isclose(savings_from_exits(np.array([1]), f), 1 - f[0])
    assert np.isclose(savings_from_exits(np.array([[1, 9], [9, 9]]), f)[0], (1 - f[0]) / 2)


def test_uniform_fractions_turn_mean_exit_into_savings():
    exits = np.array([1, 2, 4, 4])
    f = uniform_cost_fractions(4)
    assert np.isclose(savings_from_exits(exits, f), 1 - exits.mean() / 4)


def test_decomposition_sums_to_the_change():
    f = cost_fractions(RESNET20_CIFAR_CUM_MMACS)
    rng = np.random.default_rng(0)
    before = rng.integers(1, 10, size=1000)
    after = np.minimum(before + rng.integers(0, 3, size=1000), 9)
    pi_b, pi_a = exit_distribution(before, 9), exit_distribution(after, 9)
    contrib = savings_change_by_exit(pi_b, pi_a, f)
    assert np.isclose(contrib.sum(), savings_from_exits(before, f) - savings_from_exits(after, f))
    assert contrib[-1] == 0


def test_per_exit_disagreement():
    exits = np.array([1, 1, 2, 3])
    losses = np.array([1.0, 0.0, 1.0, 0.0])
    q = per_exit_disagreement(exits, losses, 4)
    assert q[0] == 0.5 and q[1] == 1.0 and q[2] == 0.0 and np.isnan(q[3])
