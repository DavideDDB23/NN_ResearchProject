import numpy as np

from eesavings.fys import hb_p_value, ucb_hb, ucb_wsr


def test_wsr_bound_is_above_the_empirical_mean():
    rng = np.random.default_rng(1)
    for p in [0.02, 0.1, 0.3]:
        x = (rng.random(200) < p).astype(float)
        assert ucb_wsr(x, delta=0.1) >= x.mean()


def test_wsr_bound_grows_with_the_losses():
    x = np.zeros(200)
    previous = ucb_wsr(x, delta=0.1)
    for k in [5, 20, 60]:
        y = x.copy()
        y[:k] = 1.0
        current = ucb_wsr(y, delta=0.1)
        assert current >= previous
        previous = current


def test_hoeffding_bentkus_is_monotone():
    p_low = hb_p_value(risk=0.02, n=100, alpha=0.1)
    p_high = hb_p_value(risk=0.08, n=100, alpha=0.1)
    assert p_low <= p_high
    assert ucb_hb(risk=0.02, delta=0.1, n_cal=100, binary_loss=True) <= ucb_hb(
        risk=0.08, delta=0.1, n_cal=100, binary_loss=True
    )
