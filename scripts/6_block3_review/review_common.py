# Helpers shared by the scripts of the review of block 3 (scripts/6_block3_review/), taken from the pilot scripts of
# the review (outputs/fable_review/parte1/common.py). The scripts only READ outputs/cifar and
# outputs/savings_under_shift; what they write goes in outputs/block3_review/ (CSV files and a .log with what they
# print). Paths are relative to the root of the repository, where the scripts are run from.
# The UCB choice of lambda is eesavings.analysis.ucb_index_fast (the same choice as ucb_index, faster).

import os
import sys

import numpy as np

from eesavings.data.cifar import CORRUPTIONS
from eesavings.savings import RESNET20_EXIT_MACS, RESNET20_FULL_MACS, cost_fractions

CIFAR = "outputs/cifar"
RESULTS = "outputs/savings_under_shift/results.csv"
OUT = "outputs/block3_review"
DATASETS = ["cifar10", "cifar100"]
REGIMES = ["joint", "disjoint", "mixed", "joint_sd"]
SEEDS = [0, 1, 2]
FAMILIES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
FAMILY_OF = {c: f for f, cs in FAMILIES.items() for c in cs}
assert sorted(FAMILY_OF) == sorted(CORRUPTIONS)


def run_dir(dataset, regime, seed):
    return os.path.join(CIFAR, dataset, regime, f"seed{seed}")


def all_runs():
    return [(d, r, s) for d in DATASETS for r in REGIMES for s in SEEDS]


def load_logits(dataset, regime, seed, corruption=None, severity=None):
    # (10000, 9, C) float32 logits of the clean test set, or of one corruption and severity (1..5)
    preds = os.path.join(run_dir(dataset, regime, seed), "preds")
    if corruption is None:
        return np.load(os.path.join(preds, "test_logits.npy"))
    return np.asarray(np.load(os.path.join(preds, "cifar_c", f"{corruption}_logits.npy"), mmap_mode="r")[severity - 1])


def load_labels(dataset, regime, seed):
    return np.load(os.path.join(run_dir(dataset, regime, seed), "preds", "test_labels.npy"))


def fractions_of(dataset):
    return cost_fractions(RESNET20_EXIT_MACS[dataset], RESNET20_FULL_MACS[dataset])


class Tee:
    # what is printed goes both to the terminal and to a log file
    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = open(path, "w")

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def start_log(name, out_dir=OUT):
    # from here on, print also writes to <out_dir>/<name>.log
    os.makedirs(out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(out_dir, f"{name}.log"))
