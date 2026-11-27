# Phase D, entry D.01: logits of the 12 CIFAR10 networks on CIFAR-10.1 v6 (Recht et al. 2018), with the same predict()
# of the export of block 2 (scripts/4_export/export_outputs.py), saved outside outputs/cifar/ (never overwritten).
# First it checks the pipeline: the same code on the CIFAR-10 test set must give the saved test logits.
# Output: outputs/fable_review/fase_d/cifar101/<regime>_seed<s>_logits.npy (2000, 9, 10) float32, and labels.npy.
# Written by Opus 5.5, 15 November 2026. Usage (from the repository root):
#   python outputs/fable_review/fase_d/export_cifar101.py > outputs/fable_review/fase_d/export_cifar101.log

import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from eesavings.data.cifar import cifar_arrays  # noqa: E402
from eesavings.export import predict  # noqa: E402
from eesavings.models import MultiExitResNet20  # noqa: E402
from eesavings.train import load_model  # noqa: E402

DATA = "outputs/fable_review/fase_d/data"
OUT = "outputs/fable_review/fase_d/cifar101"
RUNS = [(r, s) for r in ["joint", "disjoint", "mixed", "joint_sd"] for s in [0, 1, 2]]


def model_of(regime, seed):
    run_dir = f"outputs/cifar/cifar10/{regime}/seed{seed}"
    return load_model(MultiExitResNet20(10), os.path.join(run_dir, "model.pt")).eval(), run_dir


def main():
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    images = np.load(os.path.join(DATA, "cifar10.1_v6_data.npy"), allow_pickle=False)
    labels = np.load(os.path.join(DATA, "cifar10.1_v6_labels.npy"), allow_pickle=False).astype(int)
    np.save(os.path.join(OUT, "labels.npy"), labels)

    # check of the pipeline on one network: CIFAR-10 test logits recomputed against the saved ones
    model, run_dir = model_of("joint", 0)
    test_images, test_labels = cifar_arrays("cifar10", "test", "data")
    logits, _ = predict(model, test_images[:2000], "cifar10", "cpu")
    saved = np.load(os.path.join(run_dir, "preds", "test_logits.npy"), mmap_mode="r")[:2000]
    print(f"check on cifar10/joint/seed0, first 2,000 test images: largest |logit difference| "
          f"{float(np.abs(logits - saved).max()):.2e}; same argmax at every exit: "
          f"{float((logits.argmax(-1) == saved.argmax(-1)).mean()):.4f}")

    for regime, seed in RUNS:
        model, run_dir = model_of(regime, seed)
        logits, _ = predict(model, images, "cifar10", "cpu")
        np.save(os.path.join(OUT, f"{regime}_seed{seed}_logits.npy"), logits.astype(np.float32))
        test_acc = np.load(os.path.join(run_dir, "preds", "test_logits.npy"), mmap_mode="r")[:, 8].argmax(-1)
        test_acc = float((test_acc == np.load(os.path.join(run_dir, "preds", "test_labels.npy"))).mean())
        print(f"{regime} seed{seed}: exit 9 accuracy CIFAR-10 test {test_acc:.4f}, CIFAR-10.1 "
              f"{float((logits[:, 8].argmax(-1) == labels).mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
