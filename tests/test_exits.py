import numpy as np

from eesavings.fys import compute_exits, conf_measures, losses_and_exits

LAMBDAS = np.round(np.arange(0.0, 1.001, 0.01)[::-1], 2)


def test_threshold_one_sends_everything_to_the_final_exit(logits_targets):
    logits, _ = logits_targets
    exits = compute_exits(conf_measures(logits), threshold=1.0)
    assert np.all(exits == logits.shape[1])


def test_exit_rule_uses_a_strict_inequality():
    conf = np.array([[0.5, 0.7, 0.9], [0.7, 0.7, 0.9]])
    exits = compute_exits(conf, threshold=0.7)
    # a confidence equal to the threshold does not exit
    assert exits.tolist() == [3, 3]
    assert compute_exits(conf, threshold=0.69).tolist() == [2, 1]


def test_consistency_loss_is_zero_without_early_exits(logits_targets):
    logits, targets = logits_targets
    losses, exits = losses_and_exits(logits, targets, LAMBDAS, "prediction-consistency")
    assert losses.shape == exits.shape == (len(LAMBDAS), logits.shape[0])
    assert losses[0].sum() == 0  # row 0 is lambda = 1
    # row k+1 has a lower threshold than row k: no input leaves later
    assert np.all(np.diff(exits, axis=0) <= 0)
