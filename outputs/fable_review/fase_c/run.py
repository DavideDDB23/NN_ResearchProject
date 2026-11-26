# Phase C runner (written by Opus, 15 November 2026; design in outputs/fable_review/quaderno.md, entry C.01).
# For every network and shifted domain Q, it runs the methods of the project document on a stream and writes, per
# method and repetition, a summary of windows around the shift, and the curves averaged over the repetitions.
#   streams   clean:  clean for --t0 + --t1 inputs (to choose gamma without looking at shifted data)
#             jump:   clean for --t0 inputs, then Q for --t1
#             return: clean for --t0, Q for --t1, clean again for --t2
#             steps:  clean for --t0, then the corruption of Q at severity 1, 2, 3, 4, 5, --t1/5 inputs each
#             mixed:  clean for --t0, then Q and two corruptions of other families at the same severity, --t1/3 each
#                     (steps and mixed added on 15 November, entry C.07; the adaptation phase stays one-shot)
#   methods   M1 fixed clean lambda; M2 fixed lambda of each segment calibrated on its calibration pool (oracle);
#             M3 M1 plus a fallback to the full network when an e-detector on the audited disagreements fires;
#             M4 controller; M5, M6, M7 controller plus one adaptation phase triggered at the shift (oracle trigger)
#             with the rows that reached exit 9 alone, the audits, or both; M7f the same with every input audited
#             during the phase; M7c continuous adaptation (no phase); M7t M7 with a trigger on lambda; M8 the maps
#             fitted on 3,000 calibration rows of Q and lambda on the other 2,000 (oracle with adaptation); M8c
#             the same maps swapped in at the shift, with the controller (restarted at the lambda of M8).
#             M1, M2 and M8 make no audits (offline references); M7nr is M7 without the restart after the fit.
#             M7w and M7cw: M7 and M7c with the maps fitted with weights 1/q (entry C.09). M7wg: M7w with a
#             label-free guard that removes the maps when the domain changes (entry C.21).
#             A suffix "@<gamma>" sets the step of the controller for one method (for the choice of gamma).
# Lambda of M1 and start of every controller: UCB on the 5,000 clean calibration rows.
# Output in --out: summary.csv (one row per network, domain, method, repetition, window), curves/<...>.npz (the
# mean over the repetitions of cost, gross savings, z, accuracy, lambda and audits, every 100 inputs), events.csv.
#
# Usage (from the repository root):
#   python outputs/fable_review/fase_c/run.py --out DIR --stream jump --methods M1 M2 M4 M7 M8
#          [--runs cifar10/joint/seed0 ...] [--domains gaussian_noise:3 ...] [--reps 10] [--pi 0.05]
#          [--gamma 5e-4] [--pi-a 0.25] [--n-target 3000] [--t0 10000] [--t1 30000] [--t2 15000] [--workers 3]

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "scripts/6_block3_review")
from sim import EPS, N_POOL, Stream, fit_diag, pools, simulate, ucb_lambda  # noqa: E402
from review_common import fractions_of, load_labels, load_logits  # noqa: E402

BLOCK = 100            # resolution of the curves
CIFAR101 = "outputs/fable_review/fase_d/cifar101"  # logits of CIFAR-10.1 (entry D.01, export_cifar101.py)
WORST = 1000           # window of the worst risk


def load_domains(dataset, regime, seed, names):
    # logits and labels of the stream pool and logits of the calibration pool, for every domain name
    cal_rows, stream_rows = pools()
    labels = load_labels(dataset, regime, seed)
    base = {"logits": {}, "labels": {}, "cal": {}}
    for name in names:
        if name == "cifar101":
            # CIFAR-10.1 (entry D.01): 2,000 images, half to calibrate and half as the stream pool, tiled to the size
            # of the other pools so that the stream draws uniformly from its 1,000 images
            logits = np.load(f"{CIFAR101}/{regime}_seed{seed}_logits.npy")
            lab = np.load(f"{CIFAR101}/labels.npy")
            perm = np.random.default_rng(0).permutation(len(logits))
            cal, pool = perm[:len(perm) // 2], perm[len(perm) // 2:]
            tiled = np.tile(pool, int(np.ceil(N_POOL / len(pool))))[:N_POOL]
            base["logits"][name], base["labels"][name], base["cal"][name] = logits[tiled], lab[tiled], logits[cal]
            continue
        if name == "clean":
            logits = load_logits(dataset, regime, seed)
        else:
            corruption, severity = name.rsplit(":", 1)
            logits = load_logits(dataset, regime, seed, corruption, int(severity))
        base["logits"][name], base["labels"][name], base["cal"][name] = \
            logits[stream_rows], labels[stream_rows], logits[cal_rows]
    return base


# the 15 corruptions in the order of src/eesavings/data/cifar.py (CORRUPTIONS); the mixed stream follows a corruption
# with the ones 5 and 10 places later, so the three come from different families (noise, blur, weather, digital)
CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
               "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
               "jpeg_compression"]


def segments_of(args, q):
    if args.stream == "clean":
        return [("clean", args.t0 + args.t1)]
    if args.stream == "jump":
        return [("clean", args.t0), (q, args.t1)]
    if args.stream == "return":
        return [("clean", args.t0), (q, args.t1), ("clean", args.t2)]
    if args.stream == "steps":
        c = q.rsplit(":", 1)[0]
        return [("clean", args.t0)] + [(f"{c}:{s}", args.t1 // 5) for s in range(1, 6)]
    if args.stream == "mixed":
        c, s = q.rsplit(":", 1)
        i = CORRUPTIONS.index(c)
        return [("clean", args.t0)] + [(f"{CORRUPTIONS[(i + k) % 15]}:{s}", args.t1 // 3) for k in (0, 5, 10)]
    raise ValueError(args.stream)


def method_spec(name, args, refs, segs):
    # the dict that simulate() takes, and the gamma of the method
    base, gamma = name.split("@")[0], float(name.split("@")[1]) if "@" in name else args.gamma
    lam1 = refs["lam1"]
    t_shift = segs[0][1] if len(segs) > 1 else None
    # restart_holdout (entry C.04): a quarter of the observed rows is kept out of the training of the maps and used
    # only to restart lambda; the phase ends when n_target training rows are left (about 4,000 observed rows)
    phase = {"trigger": "oracle", "t_start": t_shift, "pi_a": args.pi_a, "n_target": args.n_target, "restart": True,
             "restart_holdout": args.restart_holdout, "restart_margin": args.restart_margin,
             "restart_test": args.restart_test}
    if base == "M7nr":  # M7 without the restart of the controller after the fit
        return {"controller": True, "phase": {**phase, "rows": "both", "restart": False, "restart_holdout": 0.0}}, gamma
    if base == "M1":
        spec = {"lam_segments": [lam1] * len(segs)}
    elif base == "M2":
        spec = {"lam_segments": [refs["lam2"][q] for q, _ in segs]}
    elif base == "M8":
        spec = {"lam_segments": [refs["lam8"][q] for q, _ in segs], "maps_segments": [refs["maps8"][q] for q, _ in segs]}
    elif base == "M3":
        spec = {"lam_segments": [lam1] * len(segs), "monitor": {"betas": args.betas, "threshold": args.threshold}}
    elif base == "M4":
        spec = {"controller": True}
    elif base == "M8c":  # the oracle maps of the new domain, swapped in at the shift, with the controller of M7
        q = segs[1][0]
        spec = {"controller": True, "swap": {"t": t_shift, "maps": refs["maps8"][q], "lam": refs["lam8"][q]}}
    elif base in ("M5", "M6", "M7"):
        rows = {"M5": "final", "M6": "audit", "M7": "both"}[base]
        spec = {"controller": True, "phase": {**phase, "rows": rows, "pi_a": args.pi if base == "M5" else args.pi_a}}
    elif base == "M7f":
        spec = {"controller": True, "phase": {**phase, "rows": "both", "pi_a": 1.0}}
    elif base == "M7w":  # M7 with the maps fitted with weights 1/q (entry C.09)
        spec = {"controller": True, "phase": {**phase, "rows": "both", "weighted": True}}
    elif base == "M7wg":  # M7w plus the guard that removes the maps when the domain changes (entry C.21)
        if args.guard_kappa is None:
            raise ValueError("M7wg needs --guard-kappa (chosen on clean streams by guard_calibrate.py)")
        spec = {"controller": True, "phase": {**phase, "rows": "both", "weighted": True},
                "guard": {"kappa": args.guard_kappa, "window": args.guard_window}}
    elif base == "M7t":
        spec = {"controller": True, "phase": {"trigger": "lambda", "kappa": args.kappa, "window": args.window,
                                              "t_min": args.window, "pi_a": args.pi_a, "n_target": args.n_target,
                                              "rows": "both", "restart": True,
                                              "restart_holdout": args.restart_holdout, "restart_margin": args.restart_margin,
             "restart_test": args.restart_test}}
    elif base == "M7cr":  # continuous adaptation that restarts lambda after every fit (repeated restarts drift
        # above or below eps, entry C.04: kept only to show it)
        spec = {"controller": True, "continuous": {"k_refit": args.k_refit, "n_target": args.n_target,
                                                   "n_min": args.n_target // 4, "restart": True,
                                                   "restart_holdout": args.restart_holdout, "restart_margin": args.restart_margin,
             "restart_test": 0.0}}
    elif base in ("M7c", "M7cw"):  # continuous adaptation: maps fitted again every k_refit inputs, lambda restarted
        # only when the test before the restart says the current one is off; M7cw with the weighted fit (entry C.09)
        spec = {"controller": True, "continuous": {"k_refit": args.k_refit, "n_target": args.n_target,
                                                   "n_min": args.n_target // 4, "restart": True,
                                                   "restart_holdout": args.restart_holdout,
                                                   "restart_margin": args.restart_margin, "restart_test": args.restart_test,
                                                   "weighted": base == "M7cw"}}
    else:
        raise ValueError(name)
    return spec, gamma


def references(base, names):
    # offline thresholds and maps of every domain, from its calibration pool (5,000 rows)
    refs = {"lam1": ucb_lambda(base["cal"]["clean"]), "lam2": {}, "lam8": {}, "maps8": {}}
    for n in names:
        cal = base["cal"][n]
        refs["lam2"][n] = ucb_lambda(cal)
        if n == "clean":
            refs["lam8"][n], refs["maps8"][n] = refs["lam1"], None
        elif len(cal) < 4000:  # CIFAR-10.1 (entry D.01): too few calibration rows for M8, which is not run there
            refs["lam8"][n], refs["maps8"][n] = None, None
        else:
            scale, bias, _ = fit_diag(cal[:3000])
            refs["lam8"][n], refs["maps8"][n] = ucb_lambda(cal[3000:], scale, bias), (scale, bias)
    return refs


def windows_of(args, segs):
    t0, t1 = args.t0, args.t1
    if args.stream == "clean":
        n = t0 + t1
        return {"all": (0, n), "first_half": (0, n // 2), "second_half": (n // 2, n)}
    w = {"all": (0, t0 + t1 + (args.t2 if args.stream == "return" else 0)), "first_5k": (0, 5000),
         "pre": (t0 - 5000, t0), "post_0_1k": (t0, t0 + 1000), "post_1k_5k": (t0 + 1000, t0 + 5000),
         "post_5k_10k": (t0 + 5000, t0 + 10000), "post_10k_20k": (t0 + 10000, t0 + 20000),
         "post_last_10k": (t0 + t1 - 10000, t0 + t1), "post_all": (t0, t0 + t1)}
    if args.stream == "return":
        w.update({"back_0_5k": (t0 + t1, t0 + t1 + 5000), "back_all": (t0 + t1, t0 + t1 + args.t2)})
    if args.stream in ("steps", "mixed"):  # one window per shifted segment, and the last 5,000 inputs
        start = t0
        for k, (_, n) in enumerate(segs[1:], 1):
            w[f"seg{k}"] = (start, start + n)
            start += n
        w["last_5k"] = (t0 + t1 - 5000, t0 + t1)
    return w


def summarize(out, wins, t_shift):
    rows = []
    R = out["cost"].shape[0]
    for r in range(R):
        for wname, (a, b) in wins.items():
            rows.append({"rep": r, "window": wname, "net": 1.0 - out["cost"][r, a:b].mean(),
                         "gross": out["gross"][r, a:b].mean(), "risk": out["z"][r, a:b].mean(),
                         "acc": out["correct"][r, a:b].mean(), "audit": out["audit"][r, a:b].mean(),
                         "lam": out["lam"][r, a:b].mean(), "exit9": (out["exit"][r, a:b] == 8).mean()})
        z = out["z"][r, t_shift:] if t_shift is not None else out["z"][r]
        mov = np.convolve(z, np.ones(WORST) / WORST, mode="valid")
        rows.append({"rep": r, "window": "worst_1k_post" if t_shift is not None else "worst_1k_all",
                     "risk": mov.max()})
    return rows


def curves(out):
    T = out["cost"].shape[1]
    n = T // BLOCK
    return {k: v[:, :n * BLOCK].reshape(v.shape[0], n, BLOCK).mean(axis=(0, 2)) for k, v in out.items()}


def one_job(job):
    (dataset, regime, seed), q, args = job
    start = time.time()
    fr = fractions_of(dataset)
    segs = segments_of(args, q)
    names = []
    for n, _ in segs:
        if n not in names:
            names.append(n)
    base = load_domains(dataset, regime, seed, names)
    refs = references(base, names)
    stream = Stream(segs, args.reps, seed=args.seed)
    t_shift = segs[0][1] if len(segs) > 1 else None
    key = {"dataset": dataset, "regime": regime, "seed": seed, "domain": q, "stream": args.stream}
    summary, events = [], []
    for m in args.methods:
        spec, gamma = method_spec(m, args, refs, segs)
        out, ev = simulate(stream, base, fr, dataset, spec, refs["lam1"], args.pi, gamma, seed=args.seed + 1)
        for row in summarize(out, windows_of(args, segs), t_shift):
            summary.append({**key, "method": m, "gamma": gamma, "pi": args.pi, **row})
        events += [{**key, "method": m, **e} for e in ev]
        path = os.path.join(args.out, "curves", f"{dataset}_{regime}_seed{seed}_{q.replace(':', '_')}_{m}.npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **curves(out))
    name = f"{dataset}_{regime}_seed{seed}_{q.replace(':', '_')}"
    for sub, rows in [("summary", summary), ("events", events)]:
        path = os.path.join(args.out, sub, f"{name}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
    refs_row = {**key, "lam1": refs["lam1"], **{f"lam2_{n}": v for n, v in refs["lam2"].items()},
                **{f"lam8_{n}": v for n, v in refs["lam8"].items()}}
    path = os.path.join(args.out, "refs", f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(refs_row, f)
    print(name, f"done in {time.time() - start:.1f} s", flush=True)
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--stream", default="jump", choices=["clean", "jump", "return", "steps", "mixed"])
    parser.add_argument("--methods", nargs="+", default=["M1", "M2", "M4", "M7", "M8"])
    parser.add_argument("--runs", nargs="+", default=["cifar10/joint/seed0"])
    parser.add_argument("--domains", nargs="+", default=["gaussian_noise:3"])
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--pi", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=5e-4)
    parser.add_argument("--pi-a", type=float, default=0.25)
    parser.add_argument("--n-target", type=int, default=3000)
    parser.add_argument("--restart-test", type=float, default=2.0,
                        help="restart lambda only if the risk of the new heads at the current lambda is more than this "
                             "many standard errors away from eps (0: always)")
    parser.add_argument("--restart-margin", type=float, default=0.0,
                        help="standard errors of the risk estimate kept as a margin in the restart of lambda")
    parser.add_argument("--restart-holdout", type=float, default=0.25,
                        help="share of the observed rows kept for the restart of lambda (0: the training rows, the "
                             "first version, optimistic)")
    parser.add_argument("--guard-kappa", type=float, default=None,
                        help="threshold in total variation of the guard of M7wg (entry C.21), one per dataset")
    parser.add_argument("--guard-window", type=int, default=1000)
    parser.add_argument("--k-refit", type=int, default=5000)
    parser.add_argument("--kappa", type=float, default=0.03)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--threshold", type=float, default=1e4)
    parser.add_argument("--t0", type=int, default=10000)
    parser.add_argument("--t1", type=int, default=30000)
    parser.add_argument("--t2", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    runs = [(r.split("/")[0], r.split("/")[1], int(r.split("/")[2].removeprefix("seed"))) for r in args.runs]
    domains = ["clean"] if args.stream == "clean" else args.domains
    jobs = []
    for run in runs:
        for q in domains:
            name = f"{run[0]}_{run[1]}_seed{run[2]}_{q.replace(':', '_')}"
            if args.force or not os.path.exists(os.path.join(args.out, "summary", f"{name}.csv")):
                jobs.append((run, q, args))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    print(len(jobs), "jobs", flush=True)
    start = time.time()
    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(one_job, jobs, chunksize=1):
            pass
    print(f"done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
