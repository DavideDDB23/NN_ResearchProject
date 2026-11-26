# Reading of the variants of Phase C (15 November 2026): outputs/fable_review/fase_c/variants/ (8 networks of seed 0,
# 15 corruptions at severity 3, the methods M4, M3, M5, M6, M7, M7f, M7c, M7t).
#   1. net savings, gross savings and risk by window, by dataset and method
#   2. gain over M4 in the last 10,000 inputs (paired, with a bootstrap over networks and corruptions), and the
#      fraction recovered; H5: M5 and M6 against M7
#   3. the adaptation: when the phase starts and ends, rows and audits used, training compute, restart of lambda
#   4. payback against M4 (break-even of the cumulative net compute since the shift)
#   5. M3: when the monitor falls back to the full network, and what it costs
# With several folders a method is taken from the last one that has it (variants_v2 after variants: the methods with
# adaptation rerun with the fixed restart, entry C.04).
# It only prints. Usage: python outputs/fable_review/pilots/verifica_fable/c_variants_read.py dir [dir ...]

import glob
import os
import sys

import numpy as np
import pandas as pd

DIRS = sys.argv[1:] or ["outputs/fable_review/fase_c/variants"]
DIR = DIRS[0]


SRC = {}  # (method, dataset) -> index of the folder its summary comes from


def load(kind):
    # every (method, dataset) from the last folder that has it (a folder may cover one dataset only while its job runs)
    frames = [pd.read_csv(f).assign(src=k) for k, d in enumerate(DIRS) for f in sorted(glob.glob(f"{d}/{kind}/*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    last = df.groupby(["method", "dataset"]).src.max()
    if kind == "summary":
        SRC.update(last.to_dict())
    pick = pd.Series(list(zip(df.method, df.dataset))).map(last).to_numpy()
    return df[df.src.to_numpy() == pick].drop(columns="src")


def curve_path(base_name, method):
    # only from the folder the summary of the method comes from: an older folder may hold a flawed version of the
    # method for a dataset the newer folder does not cover (M6 of CIFAR100 in variants, entry C.04)
    k = SRC.get((method, base_name.split("_")[0]))
    if k is None:
        return None
    path = f"{DIRS[k]}/curves/{base_name}_{method}.npz"
    return path if os.path.exists(path) else None


def present(g, m):
    # the method was run for this dataset (after the unstack a method of another dataset is an all-NaN column)
    return m in g and g[m].notna().any()
BLOCK, REPS = 100, 2000
METHODS = ["M4", "M3", "M5", "M6", "M7", "M7f", "M7c", "M7t", "M7w", "M7cw"]  # M7w, M7cw: entry C.09


def boot(cells, reps=REPS, seed=0):
    a = cells.to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        i, j = rng.integers(a.shape[0], size=a.shape[0]), rng.integers(a.shape[1], size=a.shape[1])
        out[r] = np.nanmean(a[np.ix_(i, j)])
    return np.percentile(out, [2.5, 97.5])


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    s = load("summary")
    args = pd.read_json(f"{DIR}/args.json", typ="series")
    t0, t1 = int(args.t0), int(args.t1)
    methods = [m for m in METHODS if m in s.method.unique()]
    print(f"{DIRS}: {s[['dataset', 'regime', 'seed']].drop_duplicates().shape[0]} networks, {s.domain.nunique()} "
          f"domains, methods {methods}")
    print("   repetitions by dataset and method:", s.groupby(["dataset", "method"]).rep.nunique().to_dict())
    print("   folder of each method:", {f"{m} {ds}": os.path.basename(DIRS[k]) for (m, ds), k in SRC.items()})

    print("\n1. By window (means over networks, corruptions, repetitions)")
    wins = ["pre", "post_0_1k", "post_1k_5k", "post_5k_10k", "post_10k_20k", "post_last_10k", "post_all"]
    for col in ["net", "risk", "audit"]:
        t = s[s.window.isin(wins)].groupby(["dataset", "method", "window"])[col].mean().unstack("window")
        print(f"\n   {col}\n", t[wins].reindex(methods, level="method").round(4).to_string())
    w = s[s.window == "worst_1k_post"].groupby(["dataset", "method"]).risk.mean().unstack("method")[methods]
    print("\n   worst window of 1,000 inputs after the shift\n", w.round(4).to_string())

    print("\n2. Last 10,000 inputs: gain over M4, paired; fraction recovered (S_0 = M4 before the shift)")
    last = s[s.window == "post_last_10k"].groupby(["dataset", "regime", "domain", "method"]).net.mean().unstack("method")
    pre = s[(s.window == "pre") & (s.method == "M4")].groupby(["dataset", "regime", "domain"]).net.mean()
    for dataset, g in last.groupby(level="dataset"):
        drop = (pre.loc[dataset] - g.M4.droplevel(0)).mean()
        print(f"   {dataset}: S_0 {pre.loc[dataset].mean():.4f}, S_1 (M4) {g.M4.mean():.4f}, drop {drop:.4f}")
        for m in methods:
            if m == "M4" or not present(g, m):
                continue
            d = (g[m] - g.M4).droplevel(0)
            lo, hi = boot(d.unstack("domain"))
            print(f"      {m:4s} S_2 {g[m].mean():.4f}, gain {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], fraction recovered "
                  f"{d.mean() / drop:.3f}")
        for a, b in [("M5", "M7"), ("M6", "M7"), ("M7f", "M7"), ("M7c", "M7"), ("M7t", "M7"), ("M7w", "M7"),
                     ("M7w", "M7f"), ("M7cw", "M7c")]:
            if present(g, a) and present(g, b):
                ga, gb = (g[a] - g.M4).mean(), (g[b] - g.M4).mean()
                print(f"      {a} recovers {ga / gb:.2f} of what {b} recovers")

    print("\n3. The adaptation (events): start and end of the phase, rows, audits, training, restart of lambda")
    ev = load("events")
    fits = ev[ev.kind == "fit"].copy()
    starts = ev[ev.kind == "phase_start"].groupby(["dataset", "regime", "domain", "method", "rep"]).t.min()
    fits = fits.merge(starts.rename("t_start").reset_index(), on=["dataset", "regime", "domain", "method", "rep"],
                      how="left")
    fits["since_shift"] = fits.t - t0
    fits["phase_length"] = fits.t - fits.t_start
    first = fits.sort_values("t").groupby(["dataset", "regime", "domain", "method", "rep"]).head(1)
    print(first.groupby(["dataset", "method"])[["t_start", "since_shift", "phase_length", "n_rows", "n_audit",
                                                "n_observed", "train_fwd", "lam_before", "lam_after"]]
          .mean().round(1).to_string())
    nfit = fits.groupby(["dataset", "method"]).size() / fits.groupby(["dataset", "method"]).rep.nunique()
    print("   fits per (network, corruption):", (nfit / s.domain.nunique() / 4).round(2).to_dict())
    print("   M7t: share of repetitions where the trigger fired, by dataset:",
          (ev[(ev.method == "M7t") & (ev.kind == "phase_start")].groupby("dataset").size()
           / (s[s.method == "M7t"].groupby("dataset")[["regime", "domain", "rep"]].apply(
               lambda x: len(x.drop_duplicates())))).round(3).to_dict())
    pre_trig = ev[(ev.method == "M7t") & (ev.kind == "phase_start") & (ev.t < t0)]
    print(f"   M7t: phases started before the shift (false alarms on clean data): {len(pre_trig)}")

    print("\n4. Payback against M4 (break-even of the cumulative net compute since the shift)")
    rows = []
    bases = sorted({os.path.basename(f)[:-len("_M4.npz")] for d in DIRS for f in glob.glob(f"{d}/curves/*_M4.npz")})
    for base in bases:
        g = base
        c4 = np.load(curve_path(base, "M4"))["cost"]
        for m in methods:
            path = curve_path(base, m)
            if m == "M4" or path is None:
                continue
            diff = np.cumsum((c4 - np.load(path)["cost"])[t0 // BLOCK:]) * BLOCK
            neg = np.flatnonzero(diff < 0)
            h = np.inf if diff[-1] <= 0 else ((neg[-1] + 1) * BLOCK if len(neg) else 0)
            rows.append({"dataset": os.path.basename(g).split("_")[0], "method": m, "payback": h,
                         "net_gain_stream": diff[-1] / t1})
    p = pd.DataFrame(rows)
    print(p.groupby(["dataset", "method"]).agg(
        median=("payback", "median"), share_paid=("payback", lambda x: np.isfinite(x).mean()),
        net_gain_over_stream=("net_gain_stream", "mean")).reindex(methods, level="method").round(3).to_string())

    print("\n5. M3: when the monitor falls back (inputs after the shift), and the share that never falls back")
    fb = ev[(ev.method == "M3") & (ev.kind == "fallback")].copy()
    if len(fb):
        fb["after"] = np.where(fb.t >= 0, fb.t - t0, np.nan)
        print(fb.groupby("dataset").after.describe().round(0).to_string())
        print("   falls back before the shift:", fb[(fb.t >= 0) & (fb.t < t0)].groupby("dataset").size().to_dict(),
              "; never:", fb[fb.t < 0].groupby("dataset").size().to_dict())


if __name__ == "__main__":
    main()
