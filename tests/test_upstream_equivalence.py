"""The ported code gives exactly the same numbers as the original Fast yet Safe code."""

import pickle

import numpy as np
import pytest

from conftest import FYS_DATA
from eesavings import fys

LAMBDAS = np.arange(0.0, 1.001, 0.01)[::-1]
EPS = np.arange(0.01, 0.51, 0.05)
RISKS = ["prediction-consistency", "prediction-gt-gap", "confidence-brier", "confidence-brier-top-pred"]


def _same_rc(up_rc, losses, exits, loss_bound):
    kw = dict(eps_grid=EPS, rcp_types=["naive", "ucb-wsr", "crc"], loss_bound=loss_bound,
              n_trials=3, n_cal=100, delta=0.1, seed=7)
    ours = fys.rc_main(losses, exits.mean(axis=1), **kw)
    theirs = up_rc.rc_main(losses, exits.mean(axis=1), **kw)
    for a, b in zip(ours, theirs):
        for key in a:
            assert np.array_equal(np.array(a[key]), np.array(b[key])), key


@pytest.mark.parametrize("risk_type", RISKS)
def test_same_results_on_synthetic_logits(upstream, logits_targets, tmp_path, risk_type):
    up_utils, up_rc = upstream
    logits, targets = logits_targets
    (tmp_path / "toy").mkdir()
    with open(tmp_path / "toy" / "test.p", "wb") as f:
        pickle.dump((np.transpose(logits, (1, 0, 2)), targets), f)

    ours = fys.img_cls_main("toy", "test", str(tmp_path), LAMBDAS, risk_type)
    theirs = up_utils.img_cls_main("toy", "test", str(tmp_path), LAMBDAS, risk_type)
    assert np.array_equal(ours[0], theirs[0]) and np.array_equal(ours[1], theirs[1])

    _same_rc(up_rc, ours[0], ours[1], loss_bound=2 if "brier" in risk_type else 1)


@pytest.mark.parametrize("risk_type", RISKS)
def test_fast_version_matches_the_slow_one(logits_targets, risk_type):
    # rc_main is checked against the original above, so this closes the chain
    logits, targets = logits_targets
    losses, exits = fys.losses_and_exits(logits, targets, LAMBDAS, risk_type)
    kw = dict(eps_grid=EPS, rcp_types=["naive", "ucb-wsr", "crc"], n_trials=3, n_cal=100,
              delta=0.1, seed=3, loss_bound=2 if "brier" in risk_type else 1)
    slow = fys.rc_main(losses, exits.mean(axis=1), **kw)
    fast = fys.rc_main_fast(losses, exits.mean(axis=1), **kw)
    for a, b in zip(slow, fast[:3]):
        for key in a:
            assert np.allclose(np.array(a[key]), np.array(b[key])), key
    # lam_ids has one row per trial: each row gives back the efficiency of that trial
    for rcp in kw["rcp_types"]:
        assert fast[3][rcp].shape == (3, len(EPS))
        assert np.array_equal(fast[3][rcp][-1], fast[2][rcp])
        assert np.allclose(np.array(fast[1][rcp]), exits.mean(axis=1)[fast[3][rcp]])


@pytest.mark.skipif(not (FYS_DATA / "msdnet" / "imagenet.p").exists(), reason="ImageNet logits not downloaded")
def test_same_results_on_msdnet_imagenet(upstream):
    up_utils, up_rc = upstream
    ours = fys.img_cls_main("msdnet", "imagenet", str(FYS_DATA), LAMBDAS, "prediction-consistency")
    theirs = up_utils.img_cls_main("msdnet", "imagenet", str(FYS_DATA), LAMBDAS, "prediction-consistency")
    assert np.array_equal(ours[0], theirs[0]) and np.array_equal(ours[1], theirs[1])
    _same_rc(up_rc, ours[0], ours[1], loss_bound=1)
