# Run the trainings of the block (datasets x seeds x regimes), with --jobs processes on the same GPU.
# The backbone (phase 1) of each dataset and seed is a job of its own: disjoint and mixed of that seed start only
# when it is on disk, so they share it. Finished runs are skipped by train_multiexit.py, so after an interruption
# (for example a Colab disconnection) the same command continues where it stopped; an unfinished phase restarts
# from its first epoch.
#
# Usage (from the repository root):
#   python scripts/run_trainings.py --seeds 0 --jobs 3 --out-dir /content/drive/MyDrive/NN_ResearchProject/outputs/cifar

import argparse
import os
import subprocess
import sys
import time

REGIMES = ["joint", "joint_sd", "disjoint", "mixed"]
TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_multiexit.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--jobs", type=int, default=1, help="trainings running at the same time")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--workers", type=int, default=4, help="data loader workers of each training")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="outputs/cifar")
    args = parser.parse_args()

    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # seed by seed; inside a seed the backbones first, so that disjoint and mixed can start as soon as possible
    pending = []
    for seed in args.seeds:
        pending += [(dataset, seed, "backbone") for dataset in args.datasets]
        pending += [(dataset, seed, regime) for dataset in args.datasets for regime in REGIMES]
    running, failed = {}, []

    def backbone_ready(dataset, seed):
        return os.path.exists(os.path.join(args.out_dir, dataset, "backbone", f"seed{seed}", "model.pt"))

    while pending or running:
        # start the jobs that can start, up to --jobs at a time
        for job in list(pending):
            dataset, seed, regime = job
            if regime in ("disjoint", "mixed"):
                if (dataset, seed, "backbone") in failed:
                    pending.remove(job)
                    failed.append(job)
                    continue
                if not backbone_ready(dataset, seed):
                    continue
            if len(running) >= args.jobs:
                break
            pending.remove(job)
            log = open(os.path.join(log_dir, f"{dataset}_{regime}_seed{seed}.log"), "a")
            cmd = [sys.executable, TRAIN_SCRIPT, "--dataset", dataset, "--regime", regime, "--seed", str(seed),
                   "--epochs", str(args.epochs), "--workers", str(args.workers),
                   "--data-dir", args.data_dir, "--out-dir", args.out_dir]
            running[job] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
            print(time.strftime("%H:%M"), "start", dataset, regime, f"seed{seed}", flush=True)

        # collect the jobs that ended
        for job, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[job]
                status = "end" if proc.returncode == 0 else f"FAILED (see {log.name})"
                print(time.strftime("%H:%M"), status, *job, flush=True)
                if proc.returncode != 0:
                    failed.append(job)
        time.sleep(10)

    print("failed jobs:", failed if failed else "none")


if __name__ == "__main__":
    main()
