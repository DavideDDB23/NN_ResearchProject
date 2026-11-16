# Review of block 3: is exit 9 a good reference for the consistency risk? From the pilots
# outputs/fable_review/parte1/reference_exit9.py and reference_mechanical.py (same logic; the two loops over the
# networks are one here, run in parallel processes). For every network, on clean data and on the 15 standard
# corruptions at severity 3 and 5 (max softmax, eps 0.05, delta 0.1, the 5 splits of results.csv):
#   A) accuracy of exit 8, of exit 9 and of an ensemble of exits 7 to 9 (mean of their softmax);
#   B) at the lambda certified by M2 (results.csv, eval rows of each split): who is right when the exit used and
#      exit 9 disagree, and the accuracy lost by the early exit system;
#   C) the same M2 analysis with two other references, on the same splits and with the same UCB:
#        "ens789"  the final prediction is the ensemble of exits 7 to 9 (no extra cost: exit 9 pays all heads anyway)
#        "exit8"   the network is cut at exit 8, which becomes the final exit (cost of exit 8, about 12% less)
#      Savings are always against the plain ResNet20, so the three references can be compared.
#   D) the mechanical part of the gain of the ensemble (exits 7 and 8 vote in the reference they are compared with):
#        1) clean, at the lambda certified against exit 9: the risk against exit 9 and against the ensemble, split
#           between the inputs that leave at exits 1-6 and at exits 7-8;
#        2) a control without the overlap: only exits 1 to 6 may fire, so no exit that can be used votes in the
#           reference. M2 savings with exit 9 and with the ensemble as the reference, clean and severity 3.
# Labels are used only to report accuracies, never to calibrate.
# Output in outputs/block3_review/: reference_accuracy.csv, reference_disagreements.csv, reference_alternatives.csv,
# reference_mechanical.csv (D1), reference_control.csv (D2) and reference_check.log.
#
# Usage: python scripts/6_block3_review/reference_check.py [--workers 3]

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from eesavings.analysis import LAMBDAS, consistency_losses, make_splits, ucb_index_fast
from eesavings.data.cifar import CORRUPTIONS
from eesavings.fys import compute_exits, softmax
from eesavings.scores import exit_scores
from review_common import OUT, RESULTS, all_runs, fractions_of, load_labels, load_logits, start_log

EPS, DELTA = 0.05, 0.1


def with_reference(logits, reference):
    # logits and scores of the system whose final exit is `reference`
    scores = exit_scores(logits, "max_softmax")
    if reference == "exit9":
        return logits, scores
    if reference == "ens789":
        ens = np.log(softmax(logits[:, 6:9].astype(np.float64), axis=2).mean(axis=1) + 1e-12)
        return np.concatenate([logits[:, :8], ens[:, None].astype(np.float32)], axis=1), scores
    if reference == "exit8":
        cut = scores[:, :8].copy()
        cut[:, -1] = 1.0
        return logits[:, :8], cut
    raise ValueError(reference)


def disagreement_row(scores, preds, labels, lam, ev):
    # B) at threshold lam, on the eval rows: risk against exit 9 and who is right in the disagreements
    exits = compute_exits(scores[ev], lam)
    used = preds[ev, exits - 1]
    final, y = preds[ev, -1], labels[ev]
    dis = used != final
    return {"lambda": lam, "risk": dis.mean(), "acc_final": (final == y).mean(), "acc_system": (used == y).mean(),
            "exit_right": (dis & (used == y)).mean(), "final_right": (dis & (final == y)).mean(),
            "both_wrong": (dis & (used != y) & (final != y)).mean()}


def one_network(job):
    dataset, regime, seed, saved = job
    fractions = fractions_of(dataset)
    labels = load_labels(dataset, regime, seed)
    splits = make_splits(5)
    lam_saved = saved[saved.method.isin(["clean", "M2"])].set_index(["corruption", "severity", "split"])["lambda"]
    lam_clean = saved[saved.method == "clean"].set_index("split")["lambda"]
    info = {"dataset": dataset, "regime": regime, "seed": seed}
    acc_rows, dis_rows, alt_rows, part, ctrl = [], [], [], [], []
    for corruption, severity in [("none", 0)] + [(c, s) for c in CORRUPTIONS for s in (3, 5)]:
        logits = load_logits(dataset, regime, seed, None if severity == 0 else corruption, severity)
        preds = logits.argmax(axis=2)
        dom = {**info, "corruption": corruption, "severity": severity}

        # A) accuracies
        ens = softmax(logits[:, 6:9].astype(np.float64), axis=2).mean(axis=1)
        acc = (preds == labels[:, None]).mean(axis=0)
        acc_rows.append({**dom, **{f"acc_{l + 1}": a for l, a in enumerate(acc)},
                         "acc_ens789": (ens.argmax(axis=1) == labels).mean(),
                         "agree_8_9": (preds[:, 7] == preds[:, 8]).mean()})

        # B) who is right in the disagreements at the certified lambda (exit 9 as reference)
        scores = exit_scores(logits, "max_softmax")
        for t, (_, _, ev) in enumerate(splits):
            dis_rows.append({**dom, "split": t,
                             **disagreement_row(scores, preds, labels, lam_saved[(corruption, severity, t)], ev)})

        # C) M2 with other references, same splits and same UCB
        for reference in ["exit9", "ens789", "exit8"]:
            ref_logits, ref_scores = with_reference(logits, reference)
            f = fractions[:ref_logits.shape[1]]
            losses, exits = consistency_losses(ref_logits, ref_scores)
            ref_preds = ref_logits.argmax(axis=2)
            for t, (_, cal, ev) in enumerate(splits):
                k = ucb_index_fast(losses[:, cal], EPS, DELTA)
                used = ref_preds[ev, exits[k, ev] - 1]
                alt_rows.append({**dom, "reference": reference, "split": t, "lambda": LAMBDAS[k],
                                 "risk": losses[k, ev].mean(), "savings": 1.0 - f[exits[k, ev] - 1].mean(),
                                 "savings_no_exit": 1.0 - f[-1],
                                 "acc_reference": (ref_preds[ev, -1] == labels[ev]).mean(),
                                 "acc_system": (used == labels[ev]).mean()})

        # D1) clean only: where the risk against exit 9 and against the ensemble comes from (all 10,000 rows)
        if severity == 0:
            ens_pred = ens.argmax(axis=1)
            for t in range(5):
                exits = compute_exits(scores, lam_clean[t])
                used = preds[np.arange(len(preds)), exits - 1]
                late, early = (exits == 7) | (exits == 8), exits <= 6
                part.append({**info, "split": t, "share_78": late.mean(),
                             "r9_16": (early & (used != preds[:, 8])).mean(),
                             "rens_16": (early & (used != ens_pred)).mean(),
                             "r9_78": (late & (used != preds[:, 8])).mean(),
                             "rens_78": (late & (used != ens_pred)).mean()})

        # D2) clean and severity 3: the control where only exits 1-6 may fire
        if severity in (0, 3):
            restricted = scores.copy()
            restricted[:, 6:8] = 0.0
            for reference in ["exit9", "ens789"]:
                ref_logits = with_reference(logits, reference)[0]
                ref_preds = ref_logits.argmax(axis=2)
                for name, sc in [("all exits", scores), ("exits 1-6 only", restricted)]:
                    losses, exits = consistency_losses(ref_logits, sc)
                    for t, (_, cal, ev) in enumerate(splits):
                        k = ucb_index_fast(losses[:, cal], EPS, DELTA)
                        used = ref_preds[ev, exits[k, ev] - 1]
                        ctrl.append({**info, "severity": severity, "reference": reference, "exits": name,
                                     "split": t, "savings": 1 - fractions[exits[k, ev] - 1].mean(),
                                     "acc_system": (used == labels[ev]).mean()})
    print(dataset, regime, seed, "done", flush=True)
    return acc_rows, dis_rows, alt_rows, part, ctrl


def print_reference(acc, dis, alt):
    print("\nA) accuracy of exit 8, exit 9 and the ensemble of exits 7-9 (mean over seeds and corruptions)")
    a = acc.groupby(["dataset", "regime", "severity"])[["acc_8", "acc_9", "acc_ens789", "agree_8_9"]].mean()
    a["9_minus_8"] = a.acc_9 - a.acc_8
    a["ens_minus_9"] = a.acc_ens789 - a.acc_9
    print(a.round(3).to_string())
    print("\nexit 9 minus exit 8 per seed, clean:")
    c = acc[acc.severity == 0].assign(diff=lambda x: x.acc_9 - x.acc_8)
    print(c.pivot_table(index=["dataset", "regime"], columns="seed", values="diff").round(4).to_string())
    print("networks (of 24) where exit 8 is at least as accurate as exit 9, by severity:",
          acc.assign(ok=acc.acc_8 >= acc.acc_9).groupby(["dataset", "regime", "seed", "severity"]).ok.mean()
          .gt(0.5).groupby("severity").sum().to_dict())

    print("\nB) disagreements at the lambda certified by M2 (shares of ALL eval inputs; risk = their sum)")
    b = dis.groupby(["dataset", "regime", "severity"])[["risk", "exit_right", "final_right", "both_wrong", "acc_final",
                                                        "acc_system"]].mean()
    b["acc_lost"] = b.acc_final - b.acc_system
    for col in ["exit_right", "final_right", "both_wrong"]:
        b[col + "_share"] = b[col] / b.risk
    print(b.round(3).to_string())

    print("\nC) M2 with other references (savings against the plain ResNet20)")
    c = alt.groupby(["dataset", "regime", "reference", "severity"])[["savings", "risk", "acc_reference", "acc_system",
                                                                    "savings_no_exit"]].mean().unstack("severity")
    print(c.round(3).to_string())
    s = alt.groupby(["dataset", "reference", "severity"]).savings.mean().unstack("severity")
    s["left_3"], s["left_5"] = s[3] / s[0], s[5] / s[0]
    print("\nshare of the clean savings left, by reference (mean over regimes and seeds):\n", s.round(3).to_string())
    # the same share net of the savings that the cut alone gives (exit8 saves about 0.115 with no early exit at all)
    n = alt.assign(net=alt.savings - alt.savings_no_exit).groupby(["dataset", "reference", "severity"]).net.mean()
    n = n.unstack("severity")
    n["left_3"], n["left_5"] = n[3] / n[0], n[5] / n[0]
    print("\nsame, counting only the savings due to early exits (savings minus savings with no exit):\n",
          n.round(3).to_string())


def print_mechanical(part, ctrl):
    p = part.drop(columns="split").groupby(["dataset", "regime"]).mean().drop(columns="seed")
    p["drop_total"] = (p.r9_16 + p.r9_78) - (p.rens_16 + p.rens_78)
    p["share_of_drop_from_78"] = (p.r9_78 - p.rens_78) / p.drop_total
    print("\nD1) clean, lambda certified against exit 9: risk against exit 9 and against the ensemble, by where the "
          "input leaves")
    print(p.round(4).to_string())
    print("mean share of the drop that comes from exits 7-8, joint regimes only:",
          p[p.index.get_level_values("regime") != "disjoint"].groupby("dataset").share_of_drop_from_78.mean()
          .round(3).to_dict())
    t = ctrl.groupby(["dataset", "exits", "reference", "severity"])[["savings", "acc_system"]].mean()
    t = t.unstack("severity")
    print("\nD2) M2 savings and accuracy of the system, by reference, with all exits and with exits 1-6 only (mean "
          "over regimes and seeds)")
    print(t.round(4).to_string())
    g = ctrl.groupby(["dataset", "exits", "reference", "severity"]).savings.mean().unstack("reference")
    g["gain_of_ensemble"] = g.ens789 - g.exit9
    print("\ngain of the ensemble reference:\n", g.round(4).to_string())
    j = ctrl[ctrl.regime != "disjoint"].groupby(["dataset", "exits", "reference", "severity"]).savings.mean()
    j = j.unstack("reference")
    print("\nsame, joint regimes only:\n", (j.ens789 - j.exit9).round(4).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--out-dir", default=OUT)
    args = parser.parse_args()
    start_log("reference_check", args.out_dir)
    pd.set_option("display.width", 220)
    saved = pd.read_csv(args.results, dtype={"l2": str})
    saved = saved[(saved.score == "max_softmax") & np.isclose(saved.eps, EPS)]
    jobs = [(d, r, s, saved[(saved.dataset == d) & (saved.regime == r) & (saved.seed == s)]) for d, r, s in all_runs()]
    with Pool(args.workers) as pool:
        results = pool.map(one_network, jobs, chunksize=1)
    names = ["accuracy", "disagreements", "alternatives", "mechanical", "control"]
    tables = [pd.DataFrame([row for result in results for row in result[i]]) for i in range(len(names))]
    for name, table in zip(names, tables):
        table.to_csv(os.path.join(args.out_dir, f"reference_{name}.csv"), index=False)
    print_reference(*tables[:3])
    print_mechanical(*tables[3:])


if __name__ == "__main__":
    main()
