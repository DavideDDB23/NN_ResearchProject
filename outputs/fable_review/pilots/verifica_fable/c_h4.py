# H4 of the project document on the main job of Phase C (15 November 2026): where the compute of the adaptation
# comes from. For every (network, domain), from the curves averaged over the repetitions and from the events:
#   extra audits   the compute of the audits of M7 minus that of M4 since the shift (cost - (1 - gross) is the audit
#                  compute of a block, plus the training when it happens)
#   training       the compute of the fit of the maps (events), in full forwards
#   gain           the gross savings of M7 minus those of M4 since the shift
# It only prints. Usage: python outputs/fable_review/pilots/verifica_fable/c_h4.py [outputs/fable_review/fase_c/main]

import glob
import os
import sys

import numpy as np
import pandas as pd

DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs/fable_review/fase_c/main"
BLOCK = 100


def main():
    args = pd.read_json(f"{DIR}/args.json", typ="series")
    t0 = int(args.t0)
    ev = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{DIR}/events/*.csv"))], ignore_index=True)
    fits = ev[(ev.kind == "fit") & (ev.method == "M7")]
    train = fits.groupby(["dataset", "regime", "seed", "domain"]).train_fwd.sum() / \
        fits.groupby(["dataset", "regime", "seed", "domain"]).rep.nunique()
    rows = []
    for f in sorted(glob.glob(f"{DIR}/curves/*_M7.npz")):
        g = f[:-len("_M7.npz")]
        c7, c4 = np.load(f), np.load(g + "_M4.npz")
        a = slice(t0 // BLOCK, None)
        audit_and_train_7 = ((c7["cost"] - (1 - c7["gross"]))[a]).sum() * BLOCK
        audit_4 = ((c4["cost"] - (1 - c4["gross"]))[a]).sum() * BLOCK
        name = os.path.basename(g)
        dataset, rest = name.split("_", 1)
        rows.append({"dataset": dataset, "name": name, "audit_and_train_7": audit_and_train_7, "audit_4": audit_4,
                     "gross_gain": (c7["gross"][a] - c4["gross"][a]).sum() * BLOCK})
    d = pd.DataFrame(rows)
    keys = train.rename("train_fwd").reset_index()
    keys["name"] = keys.dataset + "_" + keys.regime + "_seed" + keys.seed.astype(str) + "_" + \
        keys.domain.str.replace(":", "_")
    d = d.merge(keys[["name", "train_fwd"]], on="name", how="left")
    d["extra_audit"] = d.audit_and_train_7 - d.audit_4 - d.train_fwd
    d["severity"] = d.name.str.rsplit("_", n=1).str[1].astype(int)
    g = d.groupby(["dataset", "severity"])[["extra_audit", "train_fwd", "gross_gain"]].mean()
    g["share_training"] = g.train_fwd / (g.train_fwd + g.extra_audit)
    print("Since the shift (30,000 inputs), in full forwards, mean over networks and corruptions:")
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
