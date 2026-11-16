# Small tests of the pure functions of the scripts of the review of block 3 (scripts/6_block3_review/).
import importlib.util
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from conftest import ROOT
from eesavings.analysis import consistency_losses
from eesavings.fys import softmax
from eesavings.models import FEATURE_DIMS
from eesavings.scores import exit_scores

REVIEW = ROOT / "scripts" / "6_block3_review"


def load_review_script(name):
    # the scripts import review_common from their own folder
    sys.path.insert(0, str(REVIEW))
    try:
        spec = importlib.util.spec_from_file_location(name, REVIEW / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(REVIEW))
    return module


def random_logits(n=500, n_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=3, size=(n, 9, n_classes)).astype(np.float32)


def test_the_ensemble_reference_is_the_mean_softmax_of_exits_7_to_9():
    ref = load_review_script("reference_check")
    logits = random_logits()
    ens_logits, scores = ref.with_reference(logits, "ens789")
    assert ens_logits.shape == logits.shape and np.array_equal(ens_logits[:, :8], logits[:, :8])
    mean_softmax = softmax(logits[:, 6:9].astype(np.float64), axis=2).mean(axis=1)
    assert np.allclose(softmax(ens_logits[:, -1].astype(np.float64), axis=1), mean_softmax, atol=1e-6)
    assert np.array_equal(ens_logits[:, -1].argmax(axis=1), mean_softmax.argmax(axis=1))
    assert np.array_equal(scores, exit_scores(logits, "max_softmax"))  # exits 1 to 8 keep their scores
    same_logits, same_scores = ref.with_reference(logits, "exit9")
    assert same_logits is logits and np.array_equal(same_scores, scores)


def test_the_cut_at_exit_8_makes_exit_8_the_final_exit():
    ref = load_review_script("reference_check")
    logits = random_logits()
    cut_logits, cut_scores = ref.with_reference(logits, "exit8")
    scores = exit_scores(logits, "max_softmax")
    assert np.array_equal(cut_logits, logits[:, :8])
    assert np.array_equal(cut_scores[:, :7], scores[:, :7]) and np.all(cut_scores[:, 7] == 1)
    losses, exits = consistency_losses(cut_logits, cut_scores)
    assert exits.max() == 8 and np.all(losses[0] == 0)  # lambda = 1: everything leaves at exit 8, no disagreement


def test_the_two_parts_add_up_to_the_drop_in_both_orders():
    dec = load_review_script("decomposition_orders")
    rng = np.random.default_rng(0)
    out = dec.add_parts(pd.DataFrame(rng.uniform(0, 0.6, size=(50, 4)), columns=["s_pp", "s_qp", "s_pq", "s_qq"]))
    for shift, lam in [("shift_1", "lambda_1"), ("shift_2", "lambda_2"), ("shift_shapley", "lambda_shapley")]:
        assert np.allclose(out[shift] + out[lam], out["drop"])
    # a case by hand: clean 0.5, shifted at the clean lambda 0.4, clean at the shifted lambda 0.45, shifted 0.3
    one = dec.add_parts(pd.DataFrame({"s_pp": [0.5], "s_qp": [0.4], "s_pq": [0.45], "s_qq": [0.3]})).iloc[0]
    assert np.allclose([one["drop"], one.shift_1, one.lambda_1, one.shift_2, one.lambda_2],
                       [0.2, 0.1, 0.1, 0.15, 0.05])
    assert np.isclose(one.shift_shapley, 0.125)


def test_macs_of_one_evaluation_of_a_head():
    cost = load_review_script("adaptation_cost")
    # forward x @ W.T and gradient x.T @ g: N * D * C multiply-adds each
    assert cost.evaluation_macs(3000, 256, 10) == 2 * 3000 * 256 * 10 == 15_360_000
    assert cost.evaluation_macs(3000, 1024, 100) == 10 * cost.evaluation_macs(3000, 1024, 10)


def test_lbfgs_evaluations_are_counted_for_every_head_and_l2():
    cost = load_review_script("adaptation_cost")
    rng = np.random.default_rng(0)
    n, c = 60, 5
    features = rng.normal(size=(n, sum(FEATURE_DIMS))).astype(np.float32)
    weights = [rng.normal(scale=0.01, size=(c, d)).astype(np.float32) for d in FEATURE_DIMS]
    biases = [np.zeros(c, dtype=np.float32) for _ in FEATURE_DIMS]
    teacher = F.softmax(torch.as_tensor(features[:, :c] * 3), dim=1)
    evals = cost.count_evaluations(features, teacher, weights, biases)
    assert len(evals) == len(FEATURE_DIMS) * len(cost.L2_GRID)
    assert all(1 <= e <= 1.25 * cost.MAX_ITER for e in evals)  # max_eval of L-BFGS is 1.25 * max_iter


def test_bootstraps_on_synthetic_data():
    iv = load_review_script("intervals")
    rng = np.random.default_rng(0)
    # a constant table gives a zero width interval; a random one an interval around its mean
    assert np.allclose(iv.table_bootstrap(np.full((6, 15), 0.2), rng, 200), [0.2, 0.2])
    table = rng.random((12, 15))
    lo, hi = iv.table_bootstrap(table, rng, 500)
    assert lo < table.mean() < hi and hi - lo < 0.3
    # hierarchical bootstrap of a ratio of means: M2 always half of clean gives exactly 0.5
    assert np.allclose(iv.hierarchical_bootstrap(np.full((4, 3, 15, 5), 0.3), np.full((4, 3, 5), 0.6), rng, 50),
                       [0.5, 0.5])
    cube, clean = rng.uniform(0.1, 0.3, size=(4, 3, 15, 5)), rng.uniform(0.4, 0.6, size=(4, 3, 5))
    lo, hi = iv.hierarchical_bootstrap(cube, clean, rng, 200)
    assert lo < cube.mean() / clean.mean() < hi
    # week 4 test: 15 corruptions x 3 severities, the cases of 8 corruptions recover a quarter
    g = pd.DataFrame({"corruption": np.repeat(np.arange(15), 3), "rec": np.repeat(np.arange(15) < 8, 3) * 0.5})
    lo, hi = iv.corruption_bootstrap(g, rng, 500)
    assert lo < 8 / 15 < hi and 0 <= lo and hi <= 1
    assert np.allclose(iv.corruption_bootstrap(g.assign(rec=0.3), rng, 50), [1, 1])
