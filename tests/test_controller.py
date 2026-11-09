import numpy as np

from eesavings.controller import controller_step, long_run_bound


def run_controller(T, eps, gamma, pi_min, rng, adversary):
    # simulates the controller against a stream: adversary(lam, t) gives the disagreement z_t of the input,
    # after seeing the threshold but before the audit is drawn; audit probabilities vary in [pi_min, 0.5]
    lam, lams, z_all, weighted = 0.5, [], [], []
    for t in range(T):
        z = adversary(lam, t) if lam < 1 else 0  # condition 2: no exit fires when lambda >= 1
        pi = rng.uniform(pi_min, 0.5)  # condition 1: known, decided before the final exit runs
        audited = rng.random() < pi
        lams.append(lam)
        z_all.append(z)
        weighted.append(audited * z / pi)
        lam = controller_step(lam, audited, z, pi, eps, gamma)
    return np.array(lams), np.array(z_all), np.array(weighted)


def worst_case(lam, t):
    # disagrees every time an early exit can fire
    return 1


def test_the_deterministic_part_holds_on_every_path():
    eps, gamma, pi_min, T = 0.05, 0.01, 0.05, 5000
    for seed in range(20):
        lams, _, weighted = run_controller(T, eps, gamma, pi_min, np.random.default_rng(seed), worst_case)
        assert lams.max() <= 1 + gamma / pi_min + 1e-12
        assert weighted.mean() <= eps + (1 + gamma / pi_min) / (gamma * T) + 1e-12


def test_the_bound_holds_with_probability_one_minus_delta():
    eps, gamma, pi_min, T, delta = 0.05, 0.01, 0.05, 5000, 0.1

    def shifting_stream(lam, t):
        # a drift: first rare disagreements, then frequent ones
        return int(rng.random() < (0.02 if t < T // 2 else 0.6))

    violations = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        _, z, _ = run_controller(T, eps, gamma, pi_min, rng, shifting_stream)
        violations += z.mean() > long_run_bound(T, eps, gamma, pi_min, delta)
    assert violations / 200 <= delta


def test_size_of_the_terms_in_the_project_document():
    # pi = 0.05, delta = 0.1: the martingale term is about 3 points after 1e5 inputs and about 1 after 1e6
    for T, expected in [(1e5, 0.030), (1e6, 0.0094)]:
        term = np.sqrt(2 * 0.95 * np.log(10) / (0.05 * T))
        assert abs(term - expected) < 0.001
    assert long_run_bound(1e6, 0.05, 0.01, 0.05, 0.1) < 0.05 + 0.011


def test_long_run_bound_has_all_three_terms():
    # closed form at one point: eps + deterministic term + martingale term (Freedman)
    T, eps, gamma, pi, delta = 1e5, 0.05, 0.01, 0.05, 0.1
    deterministic = (1 + gamma / pi) / (gamma * T)
    martingale = np.sqrt(2 * (1 - pi) * np.log(1 / delta) / (pi * T)) + 2 * np.log(1 / delta) / (3 * T)
    assert np.isclose(long_run_bound(T, eps, gamma, pi, delta), eps + deterministic + martingale)
    assert martingale > 0.029  # the term that dominates for a realistic audit rate
