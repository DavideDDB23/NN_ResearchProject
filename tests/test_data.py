import numpy as np
import pytest
from torchvision import datasets

from conftest import ROOT
from eesavings.data.cifar import (ALL_CORRUPTIONS, CIFAR_C_FOLDER, MEAN, STD, cifar_arrays, cifar_c_row,
                                  cifar_datasets, load_cifar_c, to_tensor_batch, train_val_indices)

DATA = ROOT / "data"
DATASETS = ["cifar10", "cifar100"]
CLASSES = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
FOLDERS = {"cifar10": "cifar-10-batches-py", "cifar100": "cifar-100-python"}


def need_cifar(dataset):
    if not (DATA / dataset / FOLDERS[dataset]).exists():
        pytest.skip(f"{dataset} not downloaded")


def need_cifar_c(dataset):
    need_cifar(dataset)
    if not (DATA / CIFAR_C_FOLDER[dataset] / "labels.npy").exists():
        pytest.skip(f"{CIFAR_C_FOLDER[dataset]} not downloaded")


def test_train_val_split_is_disjoint_complete_and_fixed():
    train_idx, val_idx = train_val_indices()
    assert len(train_idx) == 45000 and len(val_idx) == 5000
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(np.union1d(train_idx, val_idx)) == 50000
    again = train_val_indices()
    assert np.array_equal(train_idx, again[0]) and np.array_equal(val_idx, again[1])


def test_cifar_c_row():
    assert cifar_c_row(1, 0) == 0 and cifar_c_row(1, 9999) == 9999
    assert cifar_c_row(3, 5) == 20005 and cifar_c_row(5, 9999) == 49999


@pytest.mark.parametrize("dataset", DATASETS)
def test_mean_and_std_of_the_training_images(dataset):
    need_cifar(dataset)
    images = CLASSES[dataset](DATA / dataset, train=True).data / 255.0
    assert np.allclose(images.mean(axis=(0, 1, 2)), MEAN[dataset], atol=1e-4)
    assert np.allclose(images.std(axis=(0, 1, 2)), STD[dataset], atol=1e-4)


@pytest.mark.parametrize("dataset", DATASETS)
def test_val_images_are_not_in_the_training_subset(dataset):
    need_cifar(dataset)
    train_set, val_set, test_set = cifar_datasets(dataset, str(DATA), download=False)
    assert len(train_set) == 45000 and len(val_set) == 5000 and len(test_set) == 10000
    assert len(np.intersect1d(train_set.indices, val_set.indices)) == 0
    # the arrays used by the export are the same images as the val subset
    images, labels = cifar_arrays(dataset, "val", str(DATA))
    full = CLASSES[dataset](DATA / dataset, train=True)
    assert np.array_equal(images, full.data[val_set.indices])
    assert np.array_equal(labels, np.array(full.targets)[val_set.indices])


@pytest.mark.parametrize("dataset", DATASETS)
def test_numpy_path_gives_the_same_tensor_as_torchvision(dataset):
    need_cifar(dataset)
    _, _, test_set = cifar_datasets(dataset, str(DATA), download=False)
    for i in [0, 17, 9999]:
        x_torchvision = test_set[i][0]
        x_numpy = to_tensor_batch(test_set.data[i:i + 1], dataset)[0]
        assert np.allclose(x_torchvision.numpy(), x_numpy.numpy(), atol=1e-6)


@pytest.mark.parametrize("dataset", DATASETS)
def test_cifar_c_files_and_labels(dataset):
    need_cifar_c(dataset)
    test_labels = np.array(CLASSES[dataset](DATA / dataset, train=False).targets)
    labels = np.load(DATA / CIFAR_C_FOLDER[dataset] / "labels.npy")
    assert labels.shape == (50000,)
    assert np.array_equal(labels.reshape(5, 10000), np.tile(test_labels, (5, 1)))  # test labels repeated 5 times
    for corruption in ALL_CORRUPTIONS:
        raw = np.load(DATA / CIFAR_C_FOLDER[dataset] / f"{corruption}.npy", mmap_mode="r")
        assert raw.shape == (50000, 32, 32, 3) and raw.dtype == np.uint8, corruption


@pytest.mark.parametrize("dataset", DATASETS)
def test_cifar_c_images_follow_the_order_of_the_test_set(dataset):
    # labels.npy is shared by all corruptions, so it cannot reveal a file in a different order:
    # check the images. Each corrupted image must be closer (correlation) to its own clean image than to the
    # other clean images, on 200 random indices. Chance level is 1/200: we ask for 90% at severity 1 and 50% at
    # severity 5, where fog and contrast wash out the images (about 75% there on the real data).
    min_matched = {1: 0.9, 5: 0.5}
    need_cifar_c(dataset)
    clean, _ = cifar_arrays(dataset, "test", str(DATA))
    idx = np.random.default_rng(0).choice(10000, size=200, replace=False)

    def standardize(images):
        a = images.reshape(len(images), -1).astype(float)
        a -= a.mean(axis=1, keepdims=True)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)

    reference = standardize(clean[idx])
    for corruption in ALL_CORRUPTIONS:
        images, _ = load_cifar_c(dataset, corruption, str(DATA))
        for severity in [1, 5]:
            similarity = standardize(images[severity - 1, idx]) @ reference.T
            matched = np.mean(similarity.argmax(axis=1) == np.arange(len(idx)))
            assert matched >= min_matched[severity], (corruption, severity, matched)
