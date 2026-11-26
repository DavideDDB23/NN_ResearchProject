# Check W1 of entry C.09 (15 November 2026): a method run in two folders with the same networks, domains, stream and
# repetitions must give the same numbers (M4 and M7 of variants_w against variants_v2, after the weighted fit was added
# to sim.py; the path without weights must not have changed).
# It only prints. Usage: python outputs/fable_review/pilots/verifica_fable/c_same_methods.py DIR_A DIR_B M4 M7

import glob
import sys

import numpy as np
import pandas as pd

a_dir, b_dir, methods = sys.argv[1], sys.argv[2], sys.argv[3:]
key = ["dataset", "regime", "seed", "domain", "method", "rep", "window"]


def load(d):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{d}/summary/*.csv"))], ignore_index=True)


a, b = load(a_dir), load(b_dir)
for m in methods:
    j = a[a.method == m].merge(b[b.method == m], on=key, suffixes=("_a", "_b"))
    diffs = {c: float(np.nanmax(np.abs(j[c + "_a"] - j[c + "_b"]))) for c in ["net", "gross", "risk", "lam"]}
    print(f"{m}: {len(j)} rows in common (of {int((b.method == m).sum())} in {b_dir}), largest differences {diffs}")
