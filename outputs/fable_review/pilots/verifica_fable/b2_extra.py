# Fable's reading of pilot B2 (15 November 2026), from the rows written by the two jobs, with separate code (it does
# not use b2_summary.py): the gains over M2 with an interval, and the comparisons the hypotheses need.
# Interval: bootstrap over the 15 corruptions (the two networks of the dataset stay together), 2,000 resamples.
#   1. gain of every configuration over m2, by dataset x severity (split 0), and the share of the gain of M8b kept
#   2. the row sets: head and logit, grid and k, from 4,000 rows down to 500
#   3. free against match (paired)
#   4. BN then heads against heads (D2, paired), and BN alone, by family
#   5. agreement against accuracy of the heads (D3) and the certificate on the accuracy lost
#   6. splits 0 and 1 side by side for the configurations that have both
# It only prints. Usage (from the repository root): python outputs/fable_review/pilots/verifica_fable/b2_extra.py

import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/6_block3_review")
from review_common import FAMILY_OF  # noqa: E402

B2 = "outputs/fable_review/pilots/B2_adattamento"
KEY = ["dataset", "regime", "corruption", "severity", "split"]
HG = "head_grid_n4000"
REPS = 2000


def read(sub):
    files = sorted(glob.glob(f"{B2}/{sub}/*.csv"))
    return pd.concat([pd.read_csv(f, dtype={"l2": str}) for f in files], ignore_index=True) if files else pd.DataFrame()


def boot(values, corruptions, reps=REPS, seed=0):
    # values: one number per case; corruptions: the corruption of every case. Mean and 95% interval when the
    # corruptions are resampled with replacement (every case of a drawn corruption goes with it)
    s = pd.Series(np.asarray(values, dtype=float)).groupby(np.asarray(corruptions)).agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(s), size=(reps, len(s)))
    means = s["sum"].to_numpy()[idx].sum(axis=1) / s["size"].to_numpy()[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(values)), lo, hi


def fmt(m, lo, hi):
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    hd, bn = read("runs_heads"), read("runs_bn")
    rows = pd.concat([hd, bn[bn.config != "source"]] if len(bn) else [hd], ignore_index=True)
    base = hd[hd.config == "m2"].set_index(KEY)
    for col in ["savings", "risk", "acc_system", "savings_acc02", "savings_acc05"]:
        rows[col + "_m2"] = base[col].reindex(pd.MultiIndex.from_frame(rows[KEY])).to_numpy()
    rows = rows.dropna(subset=["savings_m2"])
    rows["gain"] = rows.savings - rows.savings_m2
    rows["family"] = rows.corruption.map(FAMILY_OF)
    s0 = rows[rows.split == 0]
    print(f"{len(hd)} rows of job B, {len(bn)} of job A; cases on split 0: "
          f"{s0[s0.config == 'm2'].groupby(['dataset', 'severity']).size().to_dict()}")

    print("\n1. Gain over m2 on split 0 (mean and 95% interval over the corruptions) and share of the gain of M8b")
    for (dataset, severity), g in s0.groupby(["dataset", "severity"]):
        ref = g[g.config == HG].gain.mean()
        print(f"\n   {dataset} severity {severity}: m2 savings {g[g.config == 'm2'].savings.mean():.4f}, "
              f"M8b (head_grid_n4000) gain {ref:+.4f}")
        out = []
        for config, c in g[g.config != "m2"].groupby("config", sort=False):
            m, lo, hi = boot(c.gain, c.corruption)
            out.append({"config": config, "cases": len(c), "gain": fmt(m, lo, hi), "kept": m / ref,
                        "positive": (c.gain > 0).mean(), "above_eps": (c.risk > 0.05).mean(),
                        "acc_system": c.acc_system.mean() - c.acc_system_m2.mean(),
                        "rows_paid": c.n_rows_paid.mean(), "train_fwd": c.train_fwd.mean()})
        print(pd.DataFrame(out).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n2. Share of the gain of M8b kept, by row set (split 0, severity 3 and 5 together in the columns)")
    for dataset, g in s0.groupby("dataset"):
        t = {}
        for severity, gs in g.groupby("severity"):
            ref = gs[gs.config == HG].gain.mean()
            for config, c in gs.groupby("config"):
                parts = config.split("_")
                if parts[0] in ("head", "logit") and len(parts) == 3:
                    t[(f"{parts[0]}_{parts[1]}", severity, parts[2])] = c.gain.mean() / ref
        t = pd.Series(t).unstack([1, 2])
        order = [(s, r) for s in (3, 5) for r in ["n4000", "n2000", "n1000", "n500", "match", "free"] if (s, r) in t]
        print(dataset, "\n", t[order].round(2).to_string())

    print("\n3. Free rows against as many random rows (gain of free minus gain of match, paired, split 0)")
    for (dataset, severity), g in s0.groupby(["dataset", "severity"]):
        line = []
        for what in ["head_grid", "head_k10", "head_k20", "logit_grid", "logit_k10", "logit_k20"]:
            f, m = g[g.config == f"{what}_free"].set_index(KEY), g[g.config == f"{what}_match"].set_index(KEY)
            if not len(f) or not len(m):
                continue
            d = (f.gain - m.gain.reindex(f.index)).dropna()
            mean, lo, hi = boot(d.to_numpy(), d.index.get_level_values("corruption"))
            line.append(f"{what}: free {f.gain.mean():+.4f}, match {m.gain.mean():+.4f}, difference {fmt(mean, lo, hi)}")
        rows_free = g[g.config == "head_k10_free"].n_rows_set.mean()
        print(f"   {dataset} severity {severity} (free rows {rows_free:.0f}):\n      " + "\n      ".join(line))

    print("\n4. BN then heads against heads alone (D2), and BN alone, split 0")
    for (dataset, severity), g in s0.groupby(["dataset", "severity"]):
        a, b = g[g.config == "c_head_grid_n4000"].set_index(KEY), g[g.config == HG].set_index(KEY)
        if len(a):
            d = (a.gain - b.gain.reindex(a.index)).dropna()
            mean, lo, hi = boot(d.to_numpy(), d.index.get_level_values("corruption"))
            fam = d.groupby(d.index.get_level_values("corruption").map(FAMILY_OF)).mean().round(4).to_dict()
            print(f"   {dataset} severity {severity}: c_head_grid_n4000 - head_grid_n4000 = {fmt(mean, lo, hi)}; by "
                  f"family {fam}; system accuracy {a.acc_system.mean():.4f} against {b.acc_system.mean():.4f}")
        for config in ["bn_n4000", "bn_n500", "bn_traffic"]:
            c = g[g.config == config]
            if len(c):
                mean, lo, hi = boot(c.gain, c.corruption)
                print(f"      {config}: gain {fmt(mean, lo, hi)}, exit 9 accuracy "
                      f"{(c.acc_exit9 - c.acc_exit9_src).mean():+.4f}, by family "
                      f"{c.groupby('family').gain.mean().round(4).to_dict()}")

    print("\n5. D3 on M8b (head_grid_n4000), split 0: the certificate on the accuracy lost")
    for (dataset, severity), g in s0[s0.config == HG].groupby(["dataset", "severity"]):
        parts = []
        for tag in ["acc02", "acc05"]:
            d = g[f"savings_{tag}"] - g[f"savings_{tag}_m2"]
            mean, lo, hi = boot(d, g.corruption)
            parts.append(f"gain in savings_{tag} {fmt(mean, lo, hi)} (m2 at {g[f'savings_{tag}_m2'].mean():.3f})")
        mean, lo, hi = boot(g.gain, g.corruption)
        print(f"   {dataset} severity {severity}: gain in certified savings {fmt(mean, lo, hi)}; " + "; ".join(parts))
    heads = read("heads_heads")
    if len(heads):
        h = heads[(heads.config == HG) & (heads.split == 0)]
        h = h.assign(family=h.corruption.map(FAMILY_OF))
        t = h.groupby(["dataset", "severity"])[["acc_src", "acc", "agree_src", "agree", "new_agree",
                                                "new_agree_ref_wrong"]].mean()
        t["d_acc"], t["d_agree"] = t.acc - t.acc_src, t.agree - t.agree_src
        t["ratio"] = t.d_agree / t.d_acc
        t["copied_errors"] = t.new_agree_ref_wrong / t.new_agree
        print("   heads of M8b, mean over the 8 heads:\n", t.round(4).to_string())
        n = h[h.family == "noise"].groupby(["dataset", "severity"])[["acc_src", "acc", "agree_src", "agree",
                                                                    "new_agree", "new_agree_ref_wrong"]].mean()
        n["ratio"] = (n.agree - n.agree_src) / (n.acc - n.acc_src)
        n["copied_errors"] = n.new_agree_ref_wrong / n.new_agree
        print("   noise only:\n", n.round(4).to_string())

    both = rows[rows.config.isin(set(rows[rows.split == 1].config))]
    if (both.split == 1).any():
        print("\n6. Gain on split 0 and on split 1, same cases otherwise (configurations present on both)")
        t = both.groupby(["dataset", "severity", "config", "split"]).gain.mean().unstack("split")
        t["difference"] = t[1] - t[0]
        print(t.round(4).to_string())


if __name__ == "__main__":
    main()
