# CIFAR10 / CIFAR100 and their corrupted versions CIFAR-10-C / CIFAR-100-C (Hendrycks and Dietterich 2019).
#
# Splits:
#   - train 45,000 and val 5,000, taken from the CIFAR training set with a fixed seed
#     (He et al. 2016 use the same 45k / 5k split). The val set is never used to train the network:
#     it is used to fit the confidence units and, later, the temperatures.
#   - test 10,000, kept whole. The calibration / evaluation split is chosen at analysis time.
# CIFAR-C: each corruption file holds the 10,000 test images for severity 1, then severity 2, ... up to 5,
# in the order of the test set, so row (s - 1) * 10000 + i is test image i at severity s.

import os

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}

# per channel mean and std of the 50,000 training images (checked in tests/test_data.py)
MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
STD = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}

N_TRAIN_FULL = 50000
N_VAL = 5000
N_TEST = 10000
SPLIT_SEED = 0

# the 15 corruptions of the benchmark, then the 4 extra ones that Hendrycks and Dietterich keep for validation
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
EXTRA_CORRUPTIONS = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]
ALL_CORRUPTIONS = CORRUPTIONS + EXTRA_CORRUPTIONS
SEVERITIES = [1, 2, 3, 4, 5]

CIFAR_C_FOLDER = {"cifar10": "CIFAR-10-C", "cifar100": "CIFAR-100-C"}


def train_val_indices(n=N_TRAIN_FULL, n_val=N_VAL, seed=SPLIT_SEED):
    # fixed random split of the training set, the same for every run
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def cifar_c_row(severity, index):
    # row of a CIFAR-C file that holds test image `index` at `severity` (1..5)
    return (severity - 1) * N_TEST + index


def train_transform(dataset):
    # standard CIFAR augmentation (He et al. 2016): random crop with 4 pixels of padding and horizontal flip
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN[dataset], STD[dataset]),
    ])


def eval_transform(dataset):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN[dataset], STD[dataset]),
    ])


def _cifar_class(dataset):
    return datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100


def cifar_datasets(dataset, data_dir="data", download=True):
    # train (with augmentation), val and test (without) as torch datasets
    cls = _cifar_class(dataset)
    root = os.path.join(data_dir, dataset)
    train_full = cls(root, train=True, transform=train_transform(dataset), download=download)
    train_full_eval = cls(root, train=True, transform=eval_transform(dataset), download=download)
    test = cls(root, train=False, transform=eval_transform(dataset), download=download)
    train_idx, val_idx = train_val_indices()
    return Subset(train_full, train_idx), Subset(train_full_eval, val_idx), test


def cifar_arrays(dataset, split, data_dir="data"):
    # raw uint8 images (N, 32, 32, 3) and labels (N,) of "val" or "test", used by the export
    cls = _cifar_class(dataset)
    root = os.path.join(data_dir, dataset)
    if split == "test":
        d = cls(root, train=False, download=True)
        return d.data, np.array(d.targets)
    d = cls(root, train=True, download=True)
    _, val_idx = train_val_indices()
    return d.data[val_idx], np.array(d.targets)[val_idx]


def load_cifar_c(dataset, corruption, data_dir="data"):
    # images (5, 10000, 32, 32, 3) uint8 (memory mapped) and the labels of the 10,000 test images
    folder = os.path.join(data_dir, CIFAR_C_FOLDER[dataset])
    images = np.load(os.path.join(folder, f"{corruption}.npy"), mmap_mode="r")
    labels = np.load(os.path.join(folder, "labels.npy"))
    return images.reshape(len(SEVERITIES), N_TEST, 32, 32, 3), labels[:N_TEST].astype(int)


def to_tensor_batch(images, dataset):
    # uint8 images (B, 32, 32, 3) -> normalized float tensor (B, 3, 32, 32), same result as eval_transform
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(MEAN[dataset]).view(1, 3, 1, 1)
    std = torch.tensor(STD[dataset]).view(1, 3, 1, 1)
    return (x - mean) / std
