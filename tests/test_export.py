import json

import numpy as np
import pytest
import torch

from conftest import ROOT
from eesavings.data.cifar import ALL_CORRUPTIONS, NUM_CLASSES, to_tensor_batch
from eesavings.export import confidence_targets, fit_confidence, predict, set_confidence, split_features
from eesavings.models import FEATURE_DIMS, MultiExitResNet20
from eesavings.train import load_model

# exported runs (outputs/cifar/<dataset>/<regime>/seed<k>/preds), checked if present
EXPORTED_RUNS = sorted(p.parent for p in (ROOT / "outputs" / "cifar").glob("*/*/seed*/conf_label.pt"))


def test_confidence_targets():
    logits = np.zeros((2, 9, 3))
    logits[0, :, 1] = 1  # every exit says class 1
    logits[1, :4, 0] = 1  # exits 1..4 say 0, the others 2
    logits[1, 4:, 2] = 1
    labels = np.array([1, 2])
    assert confidence_targets(logits, labels, "label").tolist() == [[1] * 8, [0] * 4 + [1] * 4]
    assert confidence_targets(logits, labels, "agreement").tolist() == [[1] * 8, [0] * 4 + [1] * 4]
    labels = np.array([0, 0])
    assert confidence_targets(logits, labels, "label").tolist() == [[0] * 8, [1] * 4 + [0] * 4]


def test_fitted_units_reproduce_the_logistic_regressions():
    # features from a random network, targets from a known linear rule plus noise
    torch.manual_seed(0)
    model = MultiExitResNet20(10).eval()
    images = (np.random.default_rng(0).random((400, 32, 32, 3)) * 255).astype(np.uint8)
    _, _, features = predict(model, images, "cifar10", "cpu", return_features=True)
    rng = np.random.default_rng(1)
    targets = []
    for f in split_features(features):
        score = (f - f.mean(axis=0)) @ rng.normal(size=f.shape[1])
        targets.append((score + score.std() * rng.normal(size=len(f)) > 0).astype(int))
    targets = np.stack(targets, axis=1)
    weights, biases = fit_confidence(features, targets)
    set_confidence(model, weights, biases)

    _, conf, _ = predict(model, images, "cifar10", "cpu", return_features=True)
    for l, f in enumerate(split_features(features)):
        expected = 1 / (1 + np.exp(-(f @ weights[l] + biases[l])))
        assert np.allclose(conf[:, l], expected, atol=1e-5), l
    # logistic regression with an intercept: the mean prediction equals the mean target on the fitting data
    assert np.allclose(conf[:, :8].mean(axis=0), targets.mean(axis=0), atol=2e-3)
    assert np.all(conf[:, 8] == 1)


@pytest.mark.parametrize("run_dir", EXPORTED_RUNS, ids=[str(p.relative_to(ROOT)) for p in EXPORTED_RUNS])
def test_saved_outputs_match_a_new_forward(run_dir):
    with open(run_dir / "args.json") as f:
        dataset = json.load(f)["dataset"]
    model = load_model(MultiExitResNet20(NUM_CLASSES[dataset]), str(run_dir / "model.pt")).eval()
    saved_conf = torch.load(run_dir / "conf_label.pt", weights_only=True)
    set_confidence(model, [w.numpy() for w in saved_conf["weights"]], saved_conf["biases"].numpy())
    preds = run_dir / "preds"
    rng = np.random.default_rng(0)

    def check(images, logits, conf):
        with torch.no_grad():
            z, c = model(to_tensor_batch(images, dataset))
        # the files were written on MPS, here we recompute on CPU: small numerical differences are expected
        assert np.allclose(z.numpy(), logits, atol=1e-3, rtol=1e-3)
        assert np.allclose(c.numpy(), conf, atol=1e-3)

    from eesavings.data.cifar import cifar_arrays, load_cifar_c
    test_images, test_labels = cifar_arrays(dataset, "test", str(ROOT / "data"))
    idx = rng.choice(10000, size=20, replace=False)
    check(test_images[idx], np.load(preds / "test_logits.npy")[idx], np.load(preds / "test_conf.npy")[idx])
    assert np.array_equal(np.load(preds / "test_labels.npy"), test_labels)

    for corruption in rng.choice(ALL_CORRUPTIONS, size=3, replace=False):
        images, _ = load_cifar_c(dataset, corruption, str(ROOT / "data"))
        logits = np.load(preds / "cifar_c" / f"{corruption}_logits.npy", mmap_mode="r")
        conf = np.load(preds / "cifar_c" / f"{corruption}_conf.npy", mmap_mode="r")
        s, i = rng.integers(0, 5, size=10), rng.integers(0, 10000, size=10)
        check(images[s, i], logits[s, i], conf[s, i])


@pytest.mark.parametrize("run_dir", EXPORTED_RUNS, ids=[str(p.relative_to(ROOT)) for p in EXPORTED_RUNS])
def test_val_confidence_matches_val_accuracy(run_dir):
    # on the val set (where the units were fitted) the mean confidence of each exit is its accuracy
    preds = run_dir / "preds"
    logits, conf = np.load(preds / "val_logits.npy"), np.load(preds / "val_conf.npy")
    labels = np.load(preds / "val_labels.npy")
    acc = (logits.argmax(axis=2) == labels[:, None]).mean(axis=0)
    assert np.allclose(conf[:, :8].mean(axis=0), acc[:8], atol=5e-3)
    assert np.load(preds / "val_features.npy", mmap_mode="r").shape == (5000, sum(FEATURE_DIMS))
