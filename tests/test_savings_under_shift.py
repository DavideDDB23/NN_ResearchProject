import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from conftest import ROOT, synthetic_logits
from eesavings.adaptation import adapt_heads, fit_head, head_logits, head_weights
from eesavings.analysis import LAMBDAS, consistency_losses, make_splits, measure, ucb_index
from eesavings.export import confidence_targets
from eesavings.models import FEATURE_DIMS
from eesavings.fys import ucb_lam
from eesavings.savings import cost_fractions, savings_from_exits
from eesavings.scores import SCORES, exit_scores, fit_learned, fit_scores, fit_temperatures, output_statistics

RUN = ROOT / "outputs" / "cifar" / "cifar10" / "joint" / "seed0"


def test_output_statistics():
    logits = np.log(np.array([[0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3]]))
    s = output_statistics(logits)
    assert np.allclose(s[0], [np.log(0.7), 0.5, -(0.7 * np.log(0.7) + 0.2 * np.log(0.2) + 0.1 * np.log(0.1))])
    assert np.allclose(s[1, 1], 0) and np.isclose(s[1, 2], np.log(3))


def test_temperature_scaling_recovers_a_known_temperature():
    # labels drawn from softmax(z / 2): the fitted temperature of the logits z is about 2
    rng = np.random.default_rng(0)
    z = rng.normal(scale=3, size=(20000, 2, 5))
    p = np.exp(z[:, 0] / 2)
    p /= p.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(5, p=row) for row in p])
    assert abs(fit_temperatures(z, labels)[0] - 2) < 0.1


def test_scores_are_probabilities_and_the_last_exit_always_answers():
    logits, targets = synthetic_logits(n=2000, n_exits=4)
    params = fit_scores(logits, targets)
    for name in SCORES:
        s = exit_scores(logits, name, params)
        assert s.shape == (2000, 4) and np.all(s[:, -1] == 1) and np.all((s >= 0) & (s <= 1)), name


def test_learned_score_is_calibrated_on_its_fitting_data():
    # logistic regression with an unpenalized intercept: mean prediction = mean target on the fitting data
    logits, labels = synthetic_logits(n=3000, n_exits=4)
    targets = confidence_targets(logits, labels, "agreement")
    s = exit_scores(logits, "agreement", {"agreement": fit_learned(logits, targets)})
    assert np.allclose(s[:, :3].mean(axis=0), targets.mean(axis=0), atol=5e-3)


def test_splits_are_disjoint_complete_and_fixed():
    splits = make_splits(3)
    for adapt, cal, ev in splits:
        assert (len(adapt), len(cal), len(ev)) == (4000, 3000, 3000)
        assert len(np.unique(np.concatenate([adapt, cal, ev]))) == 10000
    again = make_splits(3)
    assert all(np.array_equal(a, b) for s, t in zip(splits, again) for a, b in zip(s, t))
    assert not np.array_equal(splits[0][1], splits[1][1])


def test_ucb_index_makes_the_same_choice_as_fast_yet_safe():
    # rows of losses with a risk that grows with the index, as along the lambda grid
    rng = np.random.default_rng(0)
    losses = (rng.random((60, 2000)) < np.linspace(0, 0.3, 60)[:, None]).astype(float)
    for eps in [0.01, 0.05, 0.1, 0.2, 0.5]:
        assert ucb_index(losses, eps, 0.1) == ucb_lam(losses, eps, 0.1)


def test_measure():
    f = cost_fractions([1, 2, 3, 4])
    losses = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
    exits = np.array([[4, 4, 4, 4], [1, 2, 4, 4]])
    m = measure(losses, exits, 1, np.arange(4), f, eps=0.3)
    assert np.isclose(m["risk"], 0.25) and np.isclose(m["savings"], savings_from_exits(exits[1], f))
    assert m["valid_savings"] == m["savings"] and m["lambda"] == LAMBDAS[1]
    assert np.allclose([m[f"pi_{l}"] for l in range(1, 5)], [0.25, 0.25, 0, 0.5])
    assert measure(losses, exits, 1, np.arange(4), f, eps=0.1)["valid_savings"] == 0.0


def test_fitting_a_head_on_its_own_predictions_keeps_it():
    # the teacher is the head itself, so the trained weights are already the optimum
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.normal(size=(500, 16)).astype(np.float32))
    w0, b0 = rng.normal(size=(5, 16)).astype(np.float32), rng.normal(size=5).astype(np.float32)
    teacher = F.softmax(x @ torch.as_tensor(w0).T + torch.as_tensor(b0), dim=1)
    w, b = fit_head(x, teacher, w0, b0, l2=1e-2)
    assert np.allclose(w.numpy(), w0, atol=1e-3) and np.allclose(b.numpy(), b0, atol=1e-3)


def test_adaptation_moves_the_heads_towards_exit_9():
    rng = np.random.default_rng(0)
    n, c = 800, 10
    features = rng.normal(size=(n, sum(FEATURE_DIMS))).astype(np.float32)
    weights = [rng.normal(scale=0.01, size=(c, d)).astype(np.float32) for d in FEATURE_DIMS]
    biases = [np.zeros(c, dtype=np.float32) for _ in FEATURE_DIMS]
    exit9 = features[:, :c] * 5  # a teacher that only the head of exit 1 can learn (it uses its features)
    before = (head_logits(features, weights, biases).argmax(axis=2) == exit9.argmax(axis=1)[:, None]).mean(axis=0)
    new_w, new_b, chosen = adapt_heads(features[:600], exit9[:600], features[600:], exit9[600:], weights, biases)
    after = (head_logits(features[600:], new_w, new_b).argmax(axis=2) == exit9[600:].argmax(axis=1)[:, None]).mean(0)
    assert after[0] > before[0] + 0.3 and len(chosen) == 8


@pytest.mark.skipif(not (RUN / "preds").exists(), reason="exported outputs not present")
def test_cached_features_and_head_weights_give_the_saved_logits():
    # the feature cache is aligned with the heads: re-computing the logits of exits 1..8 gives the saved ones
    weights, biases = head_weights(torch.load(RUN / "model.pt", map_location="cpu", weights_only=True))
    features = np.load(RUN / "preds" / "test_features.npy", mmap_mode="r")[:500]
    saved = np.load(RUN / "preds" / "test_logits.npy", mmap_mode="r")[:500, :8]
    recomputed = head_logits(features, weights, biases)
    assert np.allclose(recomputed, saved, atol=2e-2, rtol=1e-2)  # features are float16
    assert np.mean(recomputed.argmax(axis=2) == saved.argmax(axis=2)) > 0.999


def test_each_exit_gets_its_own_temperature():
    # exit 1 sampled with temperature 2, exit 2 with temperature 0.5: fitted and applied exit by exit
    rng = np.random.default_rng(1)
    z = rng.normal(scale=3, size=(20000, 3, 5))
    labels = np.zeros((20000, 2), dtype=int)
    for l, t in enumerate([2.0, 0.5]):
        p = np.exp(z[:, l] / t)
        p /= p.sum(axis=1, keepdims=True)
        labels[:, l] = [rng.choice(5, p=row) for row in p]
    temps = [fit_temperatures(z[:, [l, 2]], labels[:, l])[0] for l in range(2)]
    assert abs(temps[0] - 2) < 0.1 and abs(temps[1] - 0.5) < 0.05
    s = exit_scores(z, "temperature", {"temperature": np.array(temps)})
    for l in range(2):
        p = np.exp(z[:, l] / temps[l] - (z[:, l] / temps[l]).max(axis=1, keepdims=True))
        assert np.allclose(s[:, l], (p / p.sum(axis=1, keepdims=True)).max(axis=1))


def load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "5_savings_under_shift" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_savings_rows_use_the_right_rows_and_thresholds(tmp_path, monkeypatch):
    # a small fake network: which rows calibrate, which rows measure, and M1 reuses the clean lambda of its split
    shift, figures = load_script("savings_under_shift"), load_script("savings_figures")
    monkeypatch.setattr(shift, "CORRUPTIONS", ["fog"])
    monkeypatch.setattr(shift, "SCORES", ["max_softmax"])
    monkeypatch.setattr(shift, "EPSILONS", [0.05])
    preds = tmp_path / "preds"
    (preds / "cifar_c").mkdir(parents=True)
    def weak_logits(n, seed):
        # exits that get better with depth, but none of them always right (the learned scores need both classes)
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 10, size=n)
        logits = rng.normal(size=(n, 9, 10))
        logits[np.arange(n), :, labels] += 0.25 * np.arange(1, 10)
        return logits.astype(np.float32), labels

    val_logits, val_labels = weak_logits(2000, 0)
    test_logits, _ = weak_logits(10000, 1)
    noisy = np.stack([test_logits + np.random.default_rng(s).normal(scale=0.5 * s, size=test_logits.shape)
                      for s in range(1, 6)]).astype(np.float32)
    for name, array in [("val_logits", val_logits), ("val_labels", val_labels), ("test_logits", test_logits)]:
        np.save(preds / f"{name}.npy", array)
    np.save(preds / "cifar_c" / "fog_logits.npy", noisy)

    rows = []
    fractions = cost_fractions(np.arange(1, 10))
    splits = make_splits(2)
    shift.calibration_rows(lambda m, s, e, c, sev, t, r: rows.append(
        {"dataset": "cifar10", "regime": "joint", "seed": 0, "method": m, "score": s, "eps": e, "corruption": c,
         "severity": sev, "split": t, **r}), str(preds), splits, fractions)
    df = pd.DataFrame(rows)

    clean = df[df.method == "clean"].set_index("split")
    for _, r in df[df.method == "M1"].iterrows():
        assert r["lambda"] == clean.loc[r.split, "lambda"]
    for _, r in df[(df.method == "M2")].iterrows():
        _, cal, ev = splits[r.split]
        logits = noisy[r.severity - 1]
        losses, exits = consistency_losses(logits, exit_scores(logits, "max_softmax"))
        k = ucb_index(losses[:, cal], 0.05, 0.1)
        assert r["lambda"] == LAMBDAS[k] and np.isclose(r.risk, losses[k, ev].mean())
        assert np.isclose(r.savings, savings_from_exits(exits[k, ev], fractions))

    # the decomposition over the exits sums to clean minus M2 (with the costs of the real ResNet20 of cifar10)
    dec = figures.decomposition(df)
    real = cost_fractions(figures.RESNET20_EXIT_MACS["cifar10"], figures.RESNET20_FULL_MACS["cifar10"])
    pi = [f"pi_{l}" for l in range(1, 10)]
    m2 = df[(df.method == "M2") & (df.severity >= 3)]
    s_m2 = (m2[pi].to_numpy() * (1 - real)).sum(axis=1).mean()
    s_clean = (clean[pi].to_numpy() * (1 - real)).sum(axis=1).mean()
    assert np.isclose(dec["drop"].sum(), s_clean - s_m2)
