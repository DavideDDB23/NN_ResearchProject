# Reading of the guard of M7wg (entry C.21, 15 November 2026): when the detector fires, relative to the changes of
# domain of each stream, and what M7wg gives against M7w and M4.
#   jump:   a fire after the phase is a false alarm (the domain does not change again); M7wg against M7w of
#           variants_w (same draws), last 10,000 inputs
#   return: share of runs that fire after the return and how fast; risk and savings after the return
#   mixed:  share of runs that fire at the first change after the phase (input 20,000) and how fast
# It only prints. Usage (from the repository root):
#   python outputs/fable_review/pilots/verifica_fable/c_guard_read.py

import glob

import numpy as np
import pandas as pd

FC = "outputs/fable_review/fase_c"
KEY = ["dataset", "regime", "seed", "domain", "rep"]


def load(folder, kind):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{FC}/{folder}/{kind}/*.csv"))], ignore_index=True)


def fires(folder, change):
    ev = load(folder, "events")
    runs = load(folder, "summary")
    runs = runs[runs.method == "M7wg"][KEY].drop_duplicates()
    g = ev[(ev.method == "M7wg") & (ev.kind == "guard_fire")][KEY + ["t"]]
    fit = ev[(ev.method == "M7wg") & (ev.kind == "fit")][KEY + ["t"]].rename(columns={"t": "t_fit"})
    j = runs.merge(fit, on=KEY, how="left").merge(g, on=KEY, how="left")
    j["after_change"] = j.t - change
    out = {}
    for dataset, d in j.groupby("dataset"):
        fired = d.t.notna()
        before = fired & (d.t < change)
        after = fired & (d.t >= change)
        out[dataset] = {"runs": len(d), "fired_before_change": round(before.mean(), 3),
                        "fired_after_change": round(after.mean(), 3),
                        "inputs_after_change_median": float(d.after_change[after].median()) if after.any() else None,
                        "inputs_after_change_p90": float(d.after_change[after].quantile(0.9)) if after.any() else None,
                        "never": round((~fired).mean(), 3)}
    return out


def main():
    pd.set_option("display.width", 250)
    print("1. Jump stream (10,000 clean, 30,000 shifted): any fire after the phase is a false alarm")
    for dataset, r in fires("guard_jump", 10 ** 9).items():
        print(f"   {dataset}: runs {r['runs']}, false alarms after the phase {r['fired_before_change']}")
    s = load("guard_jump", "summary")
    w = load("variants_w", "summary")
    last = pd.concat([s[s.method.isin(["M4", "M7wg"])], w[w.method == "M7w"]])
    last = last[last.window.isin(["post_last_10k", "post_all"])]
    t = last.groupby(["dataset", "method", "window"])[["net", "risk"]].mean().unstack("window")
    print(t.round(4).to_string())

    print("\n2. Return stream (10,000 clean, 15,000 shifted, 15,000 clean): fires relative to the return (input 25,000)")
    for dataset, r in fires("guard_return", 25000).items():
        print(f"   {dataset}: {r}")

    print("\n3. Mixed stream (three corruptions, 10,000 inputs each): fires relative to the first change after the "
          "phase (input 20,000)")
    for dataset, r in fires("guard_mixed", 20000).items():
        print(f"   {dataset}: {r}")


if __name__ == "__main__":
    main()
