# Train the multi exit ResNet20 with one training regime, on one dataset, with one seed.
# --regime backbone trains only phase 1, saved in <out-dir>/<dataset>/backbone/seed<k>/ and reused by
# disjoint and mixed of the same seed (they train it themselves if it is not there yet).
#
# Usage:
#   python scripts/3_training/train_multiexit.py --dataset cifar100 --regime mixed --seed 0
#   python scripts/3_training/train_multiexit.py --dataset cifar10 --regime joint --seed 0 --epochs 2 --limit 512 \
#       --out-dir /tmp/x

import argparse
import json
import os
import subprocess

import torch
from torch.utils.data import DataLoader, Subset

from eesavings.data.cifar import NUM_CLASSES, cifar_datasets
from eesavings.models import MultiExitResNet20
from eesavings.train import REGIMES, train_backbone, train_regime


def git_commit():
    # commit of the code that runs ("-dirty" if there are uncommitted changes), or "unknown" outside a git checkout
    try:
        result = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=40"],
                                cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
    except OSError:  # git is not installed
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--regime", choices=REGIMES + ["backbone"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=160, help="epochs of one phase (joint gets twice as many)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="outputs/cifar")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N train and val images (quick checks)")
    args = parser.parse_args()

    backbone_dir = os.path.join(args.out_dir, args.dataset, "backbone", f"seed{args.seed}")
    run_dir = backbone_dir if args.regime == "backbone" else os.path.join(
        args.out_dir, args.dataset, args.regime, f"seed{args.seed}")

    train_set, val_set, _ = cifar_datasets(args.dataset, args.data_dir)
    if args.limit is not None:
        train_set = Subset(train_set, range(args.limit))
        val_set = Subset(val_set, range(min(args.limit, len(val_set))))

    # the seed fixes the initialization and the order of the batches (the splits never change)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True  # faster convolutions for fixed input sizes (not bitwise reproducible)
    generator = torch.Generator().manual_seed(args.seed)
    loader_args = dict(num_workers=args.workers, persistent_workers=args.workers > 0,
                       pin_memory=args.device == "cuda")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_set, batch_size=500, shuffle=False, **loader_args)

    def make_model():
        return MultiExitResNet20(NUM_CLASSES[args.dataset])

    # record the arguments and where the run happened (code version, torch, GPU), only for a run that trains:
    # a finished run keeps the args.json it was trained with. Disjoint and mixed may also train the backbone.
    info = dict(vars(args), git_commit=git_commit(), torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if args.device == "cuda" else args.device)
    dirs = [(run_dir, args.regime)]
    if args.regime in ("disjoint", "mixed"):
        dirs.append((backbone_dir, "backbone"))
    for d, regime in dirs:
        if not os.path.exists(os.path.join(d, "model.pt")):
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "args.json"), "w") as f:
                json.dump(dict(info, regime=regime), f, indent=2)

    if args.regime == "backbone":
        train_backbone(make_model, train_loader, val_loader, args.epochs, args.device, backbone_dir, seed=args.seed)
    else:
        train_regime(make_model, args.regime, train_loader, val_loader, args.epochs, args.device, run_dir,
                     backbone_dir, seed=args.seed)
    print("saved in", run_dir)


if __name__ == "__main__":  # needed on macOS, where the DataLoader workers start a new python process
    main()
