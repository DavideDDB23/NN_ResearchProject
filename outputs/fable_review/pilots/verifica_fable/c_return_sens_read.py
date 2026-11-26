# Readings of two small jobs of Phase C (15 November 2026).
#   return  streams clean -> shifted -> clean (10,000, 15,000, 15,000 inputs): savings and risk after the return to
#           the clean domain, against before the shift (forgetting), for M4, M7, M7c, M8c
#   sens    the same stream as the main job with another pi or gamma: for every folder and method, net savings and
#           risk in the last 10,000 inputs, worst window, transient, and the gain of M7 over M4 with its payback
#   segments  streams with several shifted segments (steps of severity, mixed corruptions; entry C.07): savings and
#           risk by segment, the difference from M4 with a bootstrap, and when the adaptations happen
# It only prints. Usage (from the repository root):
#   python outputs/fable_review/pilots/verifica_fable/c_return_sens_read.py return outputs/fable_review/fase_c/return
#   python outputs/fable_review/pilots/verifica_fable/c_return_sens_read.py sens DIR [DIR ...]

import glob
import os
import sys

import numpy as np
import pandas as pd

EPS, BLOCK = 0.05, 100


def load(d):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{d}/summary/*.csv"))], ignore_index=True)


def read_return(d):
    s = load(d)
    a = pd.read_json(f"{d}/args.json", typ="series")
    print(f"{d}: {s[['dataset', 'regime', 'seed']].drop_duplicates().shape[0]} networks, {s.domain.nunique()} domains, "
          f"{s.rep.nunique()} repetitions; stream {a.t0} clean, {a.t1} shifted, {a.t2} clean")
    wins = ["pre", "post_0_1k", "post_1k_5k", "post_last_10k", "back_0_5k", "back_all", "all"]
    for col in ["net", "risk", "lam"]:
        t = s[s.window.isin(wins)].groupby(["dataset", "method", "window"])[col].mean().unstack("window")[wins]
        print(f"\n   {col}\n", t.round(4).to_string())
    print("\n   forgetting: net savings before the shift minus after the return (first 5,000 and all 15,000 inputs)")
    p = s[s.window.isin(["pre", "back_0_5k", "back_all"])].groupby(["dataset", "method", "window"]).net.mean() \
        .unstack("window")
    p["loss_first_5k"], p["loss_all"] = p.pre - p.back_0_5k, p.pre - p.back_all
    print(p.round(4).to_string())
    print("\n   return to clean: window of 1,000 inputs after the return, inputs until the savings (mean over "
          "repetitions) are back within 0.01 of those before the shift")
    rows = []
    t_back = int(a.t0 + a.t1)
    for f in sorted(glob.glob(f"{d}/curves/*.npz")):
        name = os.path.basename(f)[:-4]
        method = name.rsplit("_", 1)[1]
        c = np.load(f)
        net = 1 - np.convolve(c["cost"], np.ones(10) / 10, mode="valid")
        s0 = net[(int(a.t0) - 5000) // BLOCK:int(a.t0) // BLOCK - 10].mean()
        after = net[t_back // BLOCK:]
        ok = np.flatnonzero(after >= s0 - 0.01)
        rows.append({"dataset": name.split("_")[0], "method": method,
                     "back_in": (ok[0] * BLOCK + 1000) if len(ok) else np.nan})
    r = pd.DataFrame(rows)
    print(r.groupby(["dataset", "method"]).back_in.agg(["median", "mean", lambda x: x.isna().mean()])
          .rename(columns={"<lambda_0>": "never"}).round(0).to_string())

    # entry C.06: the last 5,000 inputs of the stream, when every method has settled on the clean domain again, and
    # the first 1,000 inputs after the return; from the curves (means over repetitions of blocks of 100 inputs)
    print("\n   last 5,000 inputs of the stream and first 1,000 after the return (curves averaged over repetitions); "
          "difference from M4 of the same network and corruption")
    rows = []
    for f in sorted(glob.glob(f"{d}/curves/*.npz")):
        name = os.path.basename(f)[:-4]
        case, method = name.rsplit("_", 1)
        c = np.load(f)
        rows.append({"dataset": name.split("_")[0], "case": case, "method": method,
                     "net_last_5k": 1 - c["cost"][-5000 // BLOCK:].mean(), "risk_last_5k": c["z"][-5000 // BLOCK:].mean(),
                     "risk_back_0_1k": c["z"][t_back // BLOCK:t_back // BLOCK + 10].mean()})
    r = pd.DataFrame(rows)
    m4 = r[r.method == "M4"].set_index("case")
    r["net_last_5k_minus_M4"] = r.net_last_5k - r.case.map(m4.net_last_5k)
    print(r.groupby(["dataset", "method"])[["net_last_5k", "net_last_5k_minus_M4", "risk_last_5k", "risk_back_0_1k"]]
          .mean().round(4).to_string())

    # entry C.10: after the return the adapted methods have a second transient of the risk, as after the shift
    print("\n   transient after the return: inputs until the window of 1,000 inputs (mean over repetitions) is back "
          "below eps + 0.01, and the same after the shift; never = not within the segment")
    rows = []
    for f in sorted(glob.glob(f"{d}/curves/*.npz")):
        name = os.path.basename(f)[:-4]
        mov = np.convolve(np.load(f)["z"], np.ones(10) / 10, mode="valid")
        row = {"dataset": name.split("_")[0], "method": name.rsplit("_", 1)[1]}
        # the same definition as the transient of c_read.py: the end of the first window of 1,000 inputs, among those
        # inside the segment, whose mean risk is below eps + 0.01 (1,000 if it is the first one)
        for label, a_, b_ in [("after_shift", int(a.t0), t_back), ("after_return", t_back, int(a.t0 + a.t1 + a.t2))]:
            below = np.flatnonzero(mov[a_ // BLOCK:(b_ // BLOCK) - 9] < EPS + 0.01)
            row[label] = (below[0] * BLOCK + 1000) if len(below) else np.nan
        rows.append(row)
    r = pd.DataFrame(rows)
    print(r.groupby(["dataset", "method"]).agg(
        after_shift_median=("after_shift", "median"), after_shift_never=("after_shift", lambda x: x.isna().mean()),
        after_return_median=("after_return", "median"),
        after_return_never=("after_return", lambda x: x.isna().mean())).round(3).to_string())


def boot(cells, reps=2000, seed=0):
    # two-way bootstrap (networks x corruptions) of the mean of a table of cells
    a = cells.to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        i, j = rng.integers(a.shape[0], size=a.shape[0]), rng.integers(a.shape[1], size=a.shape[1])
        out[r] = np.nanmean(a[np.ix_(i, j)])
    return np.percentile(out, [2.5, 97.5])


def read_segments(d):
    # entry C.07: streams with several shifted segments (steps of severity 1 to 5, or three corruptions in a row)
    s = load(d)
    a = pd.read_json(f"{d}/args.json", typ="series")
    t0 = int(a.t0)
    segs = sorted([w for w in s.window.unique() if w.startswith("seg")], key=lambda w: int(w[3:]))
    print(f"{d}: stream {a.stream}, {s[['dataset', 'regime', 'seed']].drop_duplicates().shape[0]} networks, "
          f"{s.domain.nunique()} domains, repetitions by dataset {s.groupby('dataset').rep.nunique().to_dict()}")
    wins = ["pre"] + segs + ["post_all", "last_5k"]
    for col in ["net", "risk"]:
        t = s[s.window.isin(wins)].groupby(["dataset", "method", "window"])[col].mean().unstack("window")[wins]
        print(f"\n   {col}\n", t.round(4).to_string())
    w = s[s.window == "worst_1k_post"].groupby(["dataset", "method"]).risk.mean().unstack("method")
    print("\n   worst window of 1,000 inputs after the first shift\n", w.round(4).to_string())

    print("\n   net savings minus M4, paired by network, corruption and repetition (two-way bootstrap)")
    key = ["dataset", "regime", "seed", "domain", "rep", "window"]
    j = s[s.method != "M4"].merge(s[s.method == "M4"][key + ["net"]], on=key, suffixes=("", "_4"))
    j["d"] = j.net - j.net_4
    for (dataset, method), g in j.groupby(["dataset", "method"]):
        parts = []
        for wname in segs + ["post_all", "last_5k"]:
            x = g[g.window == wname].groupby(["regime", "domain"]).d.mean().unstack("domain")
            lo, hi = boot(x)
            parts.append(f"{wname} {np.nanmean(x.to_numpy()):+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"   {dataset} {method}: " + "; ".join(parts))

    print("\n   adaptations: phase starts (M7, M7t) and fits, by segment of the stream")
    ev = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{d}/events/*.csv"))], ignore_index=True)
    n_seg = len(segs)
    length = (int(a.t1) // n_seg)
    ev["segment"] = np.where(ev.t < t0, 0, (ev.t - t0) // length + 1)
    cases = s[["dataset", "method", "regime", "domain", "rep"]].drop_duplicates().groupby(["dataset", "method"]).size()
    for kind in ["phase_start", "fit"]:
        e = ev[ev.kind == kind].groupby(["dataset", "method", "segment"]).size().unstack("segment", fill_value=0)
        print(f"   {kind}, per run of a method (network, corruption, repetition), by segment (0 = clean):")
        print((e.div(cases.reindex(e.index), axis=0)).round(3).to_string())
    f = ev[(ev.kind == "fit") & ev.lam_before.notna()]
    f = f.assign(restarted=(f.lam_before - f.lam_after).abs() > 1e-12)
    print("   share of fits followed by a restart of lambda, by segment:")
    print(f.groupby(["dataset", "method", "segment"]).restarted.mean().unstack("segment").round(3).to_string())


def read_sens(dirs):
    out = []
    for d in dirs:
        s = load(d)
        a = pd.read_json(f"{d}/args.json", typ="series")
        t0 = int(a.t0)
        for (dataset, method), g in s.groupby(["dataset", "method"]):
            last = g[g.window == "post_last_10k"]
            gamma = float(method.split("@")[1]) if "@" in method else float(a.gamma)
            row = {"folder": os.path.basename(d), "dataset": dataset, "method": method.split("@")[0],
                   "pi": float(a.pi), "gamma": gamma, "net_last": last.net.mean(), "risk_last": last.risk.mean(),
                   "pre_net": g[g.window == "pre"].net.mean(), "pre_risk": g[g.window == "pre"].risk.mean(),
                   "worst_1k": g[g.window == "worst_1k_post"].risk.mean(),
                   "risk_all": g[g.window == "all"].risk.mean() if "all" in g.window.values else np.nan}
            trans = []
            for f in glob.glob(f"{d}/curves/{dataset}_*_{method}.npz"):
                mov = np.convolve(np.load(f)["z"], np.ones(10) / 10, mode="valid")
                below = np.flatnonzero(mov[t0 // BLOCK:] < EPS + 0.01)
                trans.append(below[0] * BLOCK + 1000 if len(below) else np.nan)
            row["transient_median"] = np.nanmedian(trans) if trans else np.nan
            out.append(row)
    t = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    print(t.sort_values(["dataset", "method", "pi", "gamma"]).round(4).to_string(index=False))
    print("\n   gain of M7 over M4 (same pi and gamma) in the last 10,000 inputs")
    m4 = t[t.method == "M4"].set_index(["dataset", "pi", "gamma"]).net_last
    m7 = t[t.method == "M7"].set_index(["dataset", "pi", "gamma"]).net_last
    print((m7 - m4.reindex(m7.index)).round(4).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    if sys.argv[1] == "return":
        read_return(sys.argv[2])
    elif sys.argv[1] == "segments":
        read_segments(sys.argv[2])
    else:
        read_sens(sys.argv[2:])
