# Reading of the main job of Phase C (15 November 2026): one or more folders outputs/fable_review/fase_c/<dir>/
# (summary, curves, events). With several folders, a method present in more than one is taken from the last given
# (main_v2 after main: M4 and M7 with the fixed restart, entry C.04), and the earlier copies are compared with it.
#   0. reproducibility: a method run in two folders must give the same numbers
#   1. windows around the shift, by dataset x severity and method (means over networks, corruptions, repetitions)
#   2. RQ2: the controller (M4) against the oracle threshold (M2) once stabilized, and the transient
#   3. RQ3: S_0, S_1 (M4), S_2 (M7), the gain and the fraction recovered, with a two-way bootstrap over networks and
#      corruptions; M8c and M8 next to them
#   4. payback of M7 against M4: the cumulative net compute from the shift, from the curves averaged over repetitions
#   5. the guarantee: long-run risk over the whole stream against the Freedman bound plus the restart term
# It only prints. Usage (from the repository root):
#   python outputs/fable_review/pilots/verifica_fable/c_read.py outputs/fable_review/fase_c/main [.../main_v2]

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from eesavings.controller import long_run_bound  # noqa: E402

DIRS = sys.argv[1:] or ["outputs/fable_review/fase_c/main"]
# the adapted method of sections 2 to 5 (entry D.01: M7w in the confirmation on the extra corruptions)
ADAPT = os.environ.get("ADAPT", "M7")
EPS, BLOCK, REPS = 0.05, 100, 2000
CASE = ["dataset", "regime", "seed", "domain"]


def table(df, col):
    d = df.assign(network=df.dataset + "/" + df.regime + "/" + df.seed.astype(str))
    return d.groupby(["network", "domain"])[col].mean().unstack("domain")


def boot(num, den=None, reps=REPS, seed=0):
    a = num.to_numpy()
    b = None if den is None else den.reindex(index=num.index, columns=num.columns).to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        i, j = rng.integers(a.shape[0], size=a.shape[0]), rng.integers(a.shape[1], size=a.shape[1])
        m = np.nanmean(a[np.ix_(i, j)])
        out[r] = m if b is None else m / np.nanmean(b[np.ix_(i, j)])
    return np.percentile(out, [2.5, 97.5])


def fmt(m, lo, hi, p=4):
    return f"{m:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


def load(kind):
    frames = []
    for k, d in enumerate(DIRS):
        for f in sorted(glob.glob(f"{d}/{kind}/*.csv")):
            frames.append(pd.read_csv(f).assign(src=k))
    return pd.concat(frames, ignore_index=True)


def latest(df):
    # keep, for every method, only the rows of the last folder that has it
    last = df.groupby("method").src.max()
    return df[df.src == df.method.map(last)].drop(columns="src")


def curve(name, method):
    # the curves of one (network, domain) and method, from the last folder that has them
    for d in reversed(DIRS):
        path = f"{d}/curves/{name}_{method}.npz"
        if os.path.exists(path):
            return np.load(path)
    return None


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    raw = load("summary")
    args = pd.read_json(f"{DIRS[0]}/args.json", typ="series")
    t0, t1, gamma, pi = int(args.t0), int(args.t1), float(args.gamma), float(args.pi)

    print("0. Reproducibility: methods run in more than one folder")
    key = CASE + ["method", "rep", "window"]
    for m, g in raw.groupby("method"):
        if g.src.nunique() < 2:
            continue
        a, b = g[g.src == g.src.min()], g[g.src == g.src.max()]
        j = a.merge(b, on=key, suffixes=("_a", "_b"))
        common = sorted(set(j.window) - {"all", "first_5k"})
        j = j[j.window.isin(common)]
        diffs = {c: float(np.nanmax(np.abs(j[c + "_a"] - j[c + "_b"]))) for c in ["net", "gross", "risk"]}
        print(f"   {m}: {len(j)} rows in common, largest differences {diffs}")

    s = latest(raw)
    s["severity"] = s.domain.str.split(":").str[1].astype(int)
    s["corruption"] = s.domain.str.split(":").str[0]
    print(f"\n{DIRS}: {s[['dataset', 'regime', 'seed']].drop_duplicates().shape[0]} networks, {s.domain.nunique()} "
          f"domains, methods {sorted(s.method.unique())}, gamma {gamma}, pi {pi}")

    print("\n1. Net savings, gross savings and risk by window (means over networks, corruptions, repetitions)")
    wins = ["pre", "post_0_1k", "post_1k_5k", "post_5k_10k", "post_10k_20k", "post_last_10k"]
    for col in ["net", "gross", "risk"]:
        t = s[s.window.isin(wins)].groupby(["dataset", "severity", "method", "window"])[col].mean().unstack("window")
        print(f"\n   {col}\n", t[wins].round(4).to_string())
    w = s[s.window == "worst_1k_post"].groupby(["dataset", "severity", "method"]).risk.mean().unstack("method")
    print("\n   worst window of 1,000 inputs after the shift\n", w.round(4).to_string())

    last = s[s.window == "post_last_10k"].groupby(CASE + ["severity", "method"])[
        ["net", "gross", "risk", "acc"]].mean().reset_index()
    pre = s[s.window == "pre"].groupby(CASE + ["severity", "method"])[["net", "gross", "risk", "acc"]].mean().reset_index()

    def col(df, method, c):
        return df[df.method == method].set_index(CASE)[c]

    print("\n2. RQ2: M4 (net, last 10,000 inputs) against M2 (no audits, so net = gross), by dataset x severity")
    for (dataset, severity), g in last.groupby(["dataset", "severity"]):
        d = (col(g, "M4", "net") - col(g, "M2", "net")).rename("d").reset_index()
        dg = (col(g, "M4", "gross") - col(g, "M2", "gross")).rename("d").reset_index()
        lo, hi = boot(table(d, "d"))
        print(f"   {dataset} severity {severity}: M4 net {col(g, 'M4', 'net').mean():.4f}, M2 "
              f"{col(g, 'M2', 'net').mean():.4f}, difference {fmt(d.d.mean(), lo, hi)}; in gross savings "
              f"{dg.d.mean():+.4f}; risk M4 {col(g, 'M4', 'risk').mean():.4f}, M2 {col(g, 'M2', 'risk').mean():.4f}")

    names = sorted({f"{r.dataset}_{r.regime}_seed{r.seed}_{r.domain.replace(':', '_')}"
                    for r in s[CASE].drop_duplicates().itertuples()})
    print("\n   transient: inputs after the shift until the window of 1,000 inputs (mean over repetitions) falls below "
          "eps + 0.01")
    trans = []
    for name in names:
        for method in ("M1", "M4", ADAPT, "M8c"):
            c = curve(name, method)
            if c is None:
                continue
            mov = np.convolve(c["z"], np.ones(10) / 10, mode="valid")
            below = np.flatnonzero(mov[t0 // BLOCK:] < EPS + 0.01)
            trans.append({"dataset": name.split("_")[0], "severity": int(name.rsplit("_", 1)[1]), "method": method,
                          "transient": (below[0] * BLOCK + 1000) if len(below) else np.nan})
    tr = pd.DataFrame(trans)
    print(tr.groupby(["dataset", "severity", "method"]).transient.agg(
        ["median", "mean", lambda x: x.isna().mean()]).rename(columns={"<lambda_0>": "never"}).round(0).to_string())

    print("\n3. RQ3: S_0 (M4, last 5,000 clean inputs), S_1 (M4) and S_2 (M7) over the last 10,000 shifted inputs")
    for (dataset, severity), g in last.groupby(["dataset", "severity"]):
        p = pre[(pre.dataset == dataset) & (pre.severity == severity)]
        s0, s1, s2 = col(p, "M4", "net"), col(g, "M4", "net"), col(g, ADAPT, "net")
        gain = (s2 - s1).rename("gain").reset_index()
        drop = (s0 - s1).rename("drop").reset_index()
        lo, hi = boot(table(gain, "gain"))
        flo, fhi = boot(table(gain, "gain"), table(drop, "drop"))
        line = (f"   {dataset} severity {severity}: S_0 {s0.mean():.4f}, S_1 {s1.mean():.4f}, S_2 {s2.mean():.4f}; "
                f"S_2 - S_1 {fmt(gain.gain.mean(), lo, hi)}; fraction recovered {gain.gain.mean() / drop['drop'].mean():.3f} "
                f"[{flo:.3f}, {fhi:.3f}]")
        for ref in ["M8c", "M8", "M2"]:
            if ref in g.method.values:
                line += f"; {ref} {col(g, ref, 'net').mean():.4f}"
        print(line)
        print(f"      accuracy last 10k: M4 {col(g, 'M4', 'acc').mean():.4f}, {ADAPT} {col(g, ADAPT, 'acc').mean():.4f}; "
              f"risk M4 {col(g, 'M4', 'risk').mean():.4f}, {ADAPT} {col(g, ADAPT, 'risk').mean():.4f}")

    print("\n4. Payback of M7 against M4 (break-even of the cumulative net compute from the shift, curves averaged over "
          "repetitions)")
    pay = []
    for name in names:
        c7, c4 = curve(name, ADAPT), curve(name, "M4")
        if c7 is None or c4 is None:
            continue
        diff = np.cumsum((c4["cost"] - c7["cost"])[t0 // BLOCK:]) * BLOCK  # positive: M7 has saved more since the shift
        neg = np.flatnonzero(diff < 0)
        h = np.nan if diff[-1] <= 0 else ((neg[-1] + 1) * BLOCK if len(neg) else 0)
        pay.append({"dataset": name.split("_")[0], "severity": int(name.rsplit("_", 1)[1]), "payback": h,
                    "net_gain_end": diff[-1] / t1})
    pay = pd.DataFrame(pay)
    pay["payback_inf"] = pay.payback.fillna(np.inf)
    print(pay.groupby(["dataset", "severity"]).agg(
        median_all=("payback_inf", "median"), median_paid=("payback", "median"),
        q25_all=("payback_inf", lambda x: np.quantile(x, 0.25)), q75_all=("payback_inf", lambda x: np.quantile(x, 0.75)),
        share_paid=("payback", lambda x: x.notna().mean()),
        mean_net_gain_over_stream=("net_gain_end", "mean")).round(3).to_string())
    print("   (median_all counts as infinite a case not paid back by the end of the stream; net gain over the stream: "
          "the cumulative net compute saved by M7 against M4 over the post-shift segment, divided by its length)")

    a = s[s.window == "all"].copy()
    if len(a):
        print("\n5. Guarantee: mean risk over the whole stream, per repetition, against the bound (delta 0.1)")
        ev = latest(load("events"))
        T = t0 + t1
        fits = ev[ev.kind == "fit"] if "kind" in ev else ev.iloc[:0]
        a["restart"] = 0.0
        if len(fits):
            fits = fits.assign(drop=np.maximum(0, fits.lam_before - fits.lam_after))
            extra = fits.groupby(CASE + ["method", "rep"])["drop"].sum().rename("restart_sum").reset_index()
            a = a.merge(extra, on=CASE + ["method", "rep"], how="left")
            a["restart"] = a.restart_sum.fillna(0)
        a["bound"] = long_run_bound(T, EPS, gamma, pi, 0.1) + a.restart / (gamma * T)
        for (dataset, m), g in a[a.method.isin(["M4", ADAPT, "M8c"])].groupby(["dataset", "method"]):
            print(f"   {dataset} {m}: mean risk over {T} inputs {g.risk.mean():.4f} (max {g.risk.max():.4f}); bound "
                  f"{g.bound.mean():.4f}; under the bound {(g.risk <= g.bound).mean():.3f}; under eps + 0.01 "
                  f"{(g.risk <= EPS + 0.01).mean():.3f}; under eps + 0.005 {(g.risk <= EPS + 0.005).mean():.3f}")


if __name__ == "__main__":
    main()
