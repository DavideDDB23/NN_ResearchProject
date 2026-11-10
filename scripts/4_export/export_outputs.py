# Run a trained network on val, test and the 19 corruptions of CIFAR-C, and save its outputs in <run-dir>/preds/.
# Steps: val (logits and features) -> fit the confidence units with target "label" on val -> test and CIFAR-C
# -> metrics.json with the accuracy of every exit.
#
# Files (numpy, not compressed, they can be opened with np.load(..., mmap_mode="r")):
#   preds/{val,test}_logits.npy (N, 9, C) float32, _conf.npy (N, 9) float32, _features.npy (N, 4352) float16,
#   preds/{val,test}_labels.npy (N,)
#   preds/cifar_c/<corruption>_logits.npy (5, 10000, 9, C), _conf.npy (5, 10000, 9),
#   and with --features also _features.npy (5, 10000, 4352) float16; preds/cifar_c/labels.npy (10000,)
#   conf_label.pt: weights of the 8 confidence units (target "label")
#
# Usage: python scripts/4_export/export_outputs.py --run-dir outputs/cifar/cifar100/joint/seed0 [--features]

import argparse
import json
import os

import numpy as np
import torch

from eesavings.data.cifar import ALL_CORRUPTIONS, N_TEST, NUM_CLASSES, SEVERITIES, cifar_arrays, load_cifar_c
from eesavings.export import confidence_targets, fit_confidence, predict, set_confidence
from eesavings.models import N_EXITS, MultiExitResNet20, exit_costs
from eesavings.train import load_model


def accuracy_from_logits(logits, labels):
    # accuracy of every exit, from saved logits (..., 9, C) and labels (...)
    return (logits.argmax(axis=-1) == labels[..., None]).mean(axis=-2).round(4).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--features", action="store_true", help="also save the features of CIFAR-C (8.3 GB)")
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "args.json")) as f:
        dataset = json.load(f)["dataset"]
    model = load_model(MultiExitResNet20(NUM_CLASSES[dataset]), os.path.join(args.run_dir, "model.pt"))
    model = model.to(args.device).eval()
    out = os.path.join(args.run_dir, "preds")
    os.makedirs(os.path.join(out, "cifar_c"), exist_ok=True)
    metrics = {}

    # val: fit the confidence units, then save the outputs with the fitted units
    images, labels = cifar_arrays(dataset, "val", args.data_dir)
    logits, _, features = predict(model, images, dataset, args.device, return_features=True)
    weights, biases = fit_confidence(features, confidence_targets(logits, labels, "label"))
    set_confidence(model, weights, biases)
    torch.save({"weights": [torch.tensor(w) for w in weights], "biases": torch.tensor(biases)},
               os.path.join(args.run_dir, "conf_label.pt"))

    for split in ["val", "test"]:
        images, labels = cifar_arrays(dataset, split, args.data_dir)
        logits, conf, features = predict(model, images, dataset, args.device, return_features=True)
        np.save(os.path.join(out, f"{split}_logits.npy"), logits)
        np.save(os.path.join(out, f"{split}_conf.npy"), conf)
        np.save(os.path.join(out, f"{split}_features.npy"), features.astype(np.float16))
        np.save(os.path.join(out, f"{split}_labels.npy"), labels)
        metrics[split] = accuracy_from_logits(logits, labels)
        if split == "val":
            metrics["val_mean_conf"] = conf.mean(axis=0).round(4).tolist()
        print(split, "accuracy per exit", metrics[split], flush=True)

    # CIFAR-C: one corruption at a time, all 5 severities together
    metrics["cifar_c"] = {}
    for corruption in ALL_CORRUPTIONS:
        images, labels = load_cifar_c(dataset, corruption, args.data_dir)
        flat = images.reshape(-1, 32, 32, 3)
        result = predict(model, flat, dataset, args.device, return_features=args.features)
        shape = (len(SEVERITIES), N_TEST)
        logits = result[0].reshape(*shape, N_EXITS, -1)
        np.save(os.path.join(out, "cifar_c", f"{corruption}_logits.npy"), logits)
        np.save(os.path.join(out, "cifar_c", f"{corruption}_conf.npy"), result[1].reshape(*shape, N_EXITS))
        if args.features:
            np.save(os.path.join(out, "cifar_c", f"{corruption}_features.npy"),
                    result[2].astype(np.float16).reshape(*shape, -1))
        metrics["cifar_c"][corruption] = accuracy_from_logits(logits, np.tile(labels, (len(SEVERITIES), 1)))
        print(corruption, "final exit accuracy by severity", [a[-1] for a in metrics["cifar_c"][corruption]],
              flush=True)
    np.save(os.path.join(out, "cifar_c", "labels.npy"), labels)

    costs, full_cost = exit_costs(MultiExitResNet20(NUM_CLASSES[dataset]))
    metrics["exit_macs"] = costs
    metrics["full_macs"] = full_cost
    with open(os.path.join(args.run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print("saved in", out)


if __name__ == "__main__":
    main()
