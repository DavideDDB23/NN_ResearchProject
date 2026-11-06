import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "RC-EENN"
FYS_DATA = ROOT / "data" / "fys"


def synthetic_logits(n=600, n_exits=4, n_classes=10, seed=0):
    """Logits whose exits get more confident and more accurate with depth, shape (N, L, C)."""
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, n_classes, size=n)
    logits = rng.normal(size=(n, n_exits, n_classes))
    for l in range(n_exits):
        logits[np.arange(n), l, targets] += 0.8 * (l + 1)
    return logits.astype(np.float32), targets


@pytest.fixture
def logits_targets():
    return synthetic_logits()


@pytest.fixture
def upstream():
    """Import the original Fast yet Safe modules from third_party/RC-EENN."""
    if not UPSTREAM.exists():
        pytest.skip("third_party/RC-EENN not cloned")
    sys.path.insert(0, str(UPSTREAM))
    try:
        from img_cls import utils as up_utils
        from rc import risk_control as up_rc
    finally:
        sys.path.remove(str(UPSTREAM))
    return up_utils, up_rc
