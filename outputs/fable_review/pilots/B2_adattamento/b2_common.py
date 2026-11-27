# Pilot B2 (an adaptation phase that pays back its compute): what the two jobs share.
#   measure_config   the certificate (consistency risk, max softmax score, UCB on the cal rows), the equal audit
#                    budget, the accuracy, the certificate on the accuracy lost, the per head agreement and the curves
#                    of one configuration, all measured on the eval rows (SPECIFICA.md, section 4)
#   m2_choice        the M2 lambda on the cal rows of the source logits and the exit of every adapt row under it
#   row_sets         the sets of adapt rows that the adaptation may use (section 6)
#   accounting       the compute of the adaptation in full forwards and the net savings over a stream (section 7)
#   outputs          one CSV / NPZ per (network, corruption), written atomically, merged by (severity, split)
# Imported by b2_bn.py, b2_heads.py, b2_summary.py and test_b2.py (they add this folder to sys.path).
# Run nothing here: python outputs/fable_review/pilots/B2_adattamento/b2_bn.py (or b2_heads.py) from the repo root.

import os
import sys

import numpy as np
import pandas as pd

from eesavings.analysis import LAMBDAS, consistency_losses, make_splits, measure, ucb_index_fast
from eesavings.fys import compute_exits
from eesavings.savings import savings_from_exits
from eesavings.scores import exit_scores

sys.path.insert(0, "scripts/6_block3_review")
from review_common import CIFAR, FAMILY_OF, fractions_of  # noqa: E402

PILOT = "outputs/fable_review/pilots/B2_adattamento"
RUNS = [f"{CIFAR}/{d}/{r}/seed0" for d in ["cifar10", "cifar100"] for r in ["joint", "joint_sd"]]
EPS, DELTA = 0.05, 0.1
EPS_ACC = [0.02, 0.05]           # certificate on the accuracy lost
SEVERITIES = [3, 5]
SPLITS = make_splits(5)          # (adapt 4,000, cal 3,000, eval 3,000) index arrays; the pilot uses splits 0 and 1
N_CAL = 3000
MAX_EQB_ROWS = 2000              # equal audit budget only when the paid rows leave at least 1,000 cal rows
HORIZONS = [10000, 100000, 1000000]
N_COLLECT = 4000                 # inputs that run with M2 while the free rows are collected


# ---------------- runs and cases ----------------

def resolve_run(run):
    # accepts outputs/cifar/cifar10/joint/seed0 or cifar10/joint/seed0
    run = run.rstrip("/")
    return run if os.path.isdir(run) else os.path.join(CIFAR, run)


def run_info(run_dir):
    dataset, regime, seed = run_dir.rstrip("/").split(os.sep)[-3:]
    return {"dataset": dataset, "regime": regime, "seed": int(seed[4:])}


def run_name(run_dir):
    i = run_info(run_dir)
    return f"{i['dataset']}_{i['regime']}_seed{i['seed']}"


def case_rows(split):
    # adapt, cal, eval of a split and the rows that are measured: cal first, then eval
    adapt, cal, ev = SPLITS[split]
    return adapt, cal, ev, np.concatenate([cal, ev])


def c9_of(dataset):
    # cost of a full forward with all the heads, in units of the plain network (above 1)
    return fractions_of(dataset)[-1]


# ---------------- measurement ----------------

def m2_choice(src_all, adapt, cal):
    # lambda index of M2 (UCB on the cal rows of the source logits) and the exit (1..9) of every adapt row at that
    # lambda: the same exit as row k of consistency_losses
    cal_logits = src_all[cal]
    losses, _ = consistency_losses(cal_logits, exit_scores(cal_logits, "max_softmax"))
    k = ucb_index_fast(losses, EPS, DELTA)
    exit_src = compute_exits(exit_scores(src_all[adapt], "max_softmax"), LAMBDAS[k])
    return k, exit_src


def equal_budget(losses, exits, ev, fr, n_rows_paid, eps=EPS, delta=DELTA):
    # the same 3,000 audited rows, n_rows_paid of them spent on the adaptation: lambda calibrated on the others
    if not 0 < n_rows_paid <= MAX_EQB_ROWS:
        return {"k_eqb": np.nan, "savings_eqb": np.nan, "risk_eqb": np.nan, "n_cal_eqb": np.nan}
    n_cal = N_CAL - n_rows_paid
    k = ucb_index_fast(losses[:, :n_cal], eps, delta)
    m = measure(losses, exits, k, ev, fr, eps)
    return {"k_eqb": k, "savings_eqb": m["savings"], "risk_eqb": float(m["risk"]), "n_cal_eqb": n_cal}


def head_records(preds, preds_src, y):
    # agreement of every head with exit 9, in the configuration and in the source network (eval rows)
    ref, ref_src = preds[:, -1], preds_src[:, -1]
    records = []
    for l in range(preds.shape[1] - 1):
        agree, agree_src = preds[:, l] == ref, preds_src[:, l] == ref_src
        new_agree = agree & ~agree_src
        records.append({"head": l + 1, "acc": np.mean(preds[:, l] == y), "agree": agree.mean(),
                        "acc_src": np.mean(preds_src[:, l] == y), "agree_src": agree_src.mean(),
                        "new_agree": new_agree.mean(), "new_agree_ref_wrong": np.mean(new_agree & (ref != y)),
                        "lost_agree": np.mean(agree_src & ~agree)})
    return [{k: float(v) if k != "head" else v for k, v in r.items()} for r in records]


def measure_config(new, src, labels, fr, n_rows_paid, eps=EPS, delta=DELTA):
    # new, src: logits (6000, 9, C) of the configuration and of the source network on concatenate([cal, ev]);
    # labels of the same rows. Exit 9 of new is the reference. Returns (scalars, 8 head records, curves).
    ev = np.arange(N_CAL, len(new))
    losses, exits = consistency_losses(new, exit_scores(new, "max_softmax"))
    k = ucb_index_fast(losses[:, :N_CAL], eps, delta)
    out = {"k": k, **measure(losses, exits, k, ev, fr, eps)}
    out["risk"] = float(out["risk"])
    out.update(equal_budget(losses, exits, ev, fr, n_rows_paid, eps, delta))

    # accuracy, and the three kinds of disagreement that make the risk at k
    preds, preds_src = new.argmax(axis=2), src.argmax(axis=2)
    used = np.take_along_axis(preds, exits.T - 1, axis=1).T      # (K, N): class given by the exit used
    right_used = used == labels                                   # (K, N)
    right_ref = preds[:, -1] == labels                            # (N,)
    y = labels[ev]
    out["acc_system"] = right_used[k, ev].mean()
    out["acc_exit9"] = right_ref[ev].mean()
    out["acc_exit9_src"] = np.mean(preds_src[ev, -1] == y)
    dis = used[k, ev] != preds[ev, -1]
    out["dis_exit_right"] = np.mean(dis & right_used[k, ev])
    out["dis_ref_right"] = np.mean(dis & right_ref[ev])
    out["dis_both_wrong"] = np.mean(dis & ~right_used[k, ev] & ~right_ref[ev])

    # certificate on the accuracy lost: g = 1[exit used wrong] - 1[exit 9 wrong], rescaled to [0, 1]
    x = (right_ref.astype(np.float64) - right_used + 1) / 2
    for eps_acc in EPS_ACC:
        tag = f"acc{round(eps_acc * 100):02d}"
        k_acc = ucb_index_fast(x[:, :N_CAL], (eps_acc + 1) / 2, delta)
        out[f"k_{tag}"] = k_acc
        out[f"savings_{tag}"] = savings_from_exits(exits[k_acc, ev], fr)
        out[f"acc_loss_{tag}"] = out["acc_exit9"] - right_used[k_acc, ev].mean()

    curves = {"cost": np.asarray(fr)[exits[:, ev] - 1].mean(axis=1).astype(np.float32),
              "acc": right_used[:, ev].mean(axis=1).astype(np.float32),
              "risk": losses[:, ev].mean(axis=1).astype(np.float32)}
    scalars = {key: (int(v) if key.startswith("k") and not np.isnan(v) else float(v)) for key, v in out.items()}
    return scalars, head_records(preds[ev], preds_src[ev], y), curves


# ---------------- row sets ----------------

def row_sets(adapt, exit_src_adapt):
    # index arrays into the 10,000 rows, in the order of the split. free: the adapt rows that reach exit 9 under M2
    # (their features and exit 9 are computed anyway); match: as many adapt rows, taken in order
    free = adapt[np.asarray(exit_src_adapt) == 9]
    return {"n4000": adapt, "n2000": adapt[:2000], "n1000": adapt[:1000], "n500": adapt[:500],
            "free": free, "match": adapt[:len(free)]}


def fit_held(rows):
    # the first three quarters fit, the last quarter chooses L2 (3,000 + 1,000 for n4000, as M8b)
    cut = (3 * len(rows)) // 4
    return rows[:cut], rows[cut:]


# ---------------- accounting (full forwards of the plain network) ----------------

def train_macs(evals, n_fit, dims, n_classes):
    # every L-BFGS evaluation is a forward and a backward of a linear map on n_fit rows: 2 * N * D * C MACs
    return int(sum(e * 2 * n_fit * d * n_classes for e, d in zip(evals, dims)))


def adaptation_rows_cost(n_rows_paid, c9, s_m2):
    # the paid rows run the whole network with all heads: incremental over M2, and in full
    return n_rows_paid * (c9 - (1 - s_m2)), n_rows_paid * c9


def payback(rows, train_fwd, gain):
    # inputs of the domain after which the gain has paid back the extra compute
    return (rows + train_fwd) / gain if gain > 0 else np.inf


def cost_kind(config):
    if config in ("m2", "source"):
        return "m2"
    if config == "bn_traffic":
        return "traffic"
    if config.endswith("_free"):
        return "free"
    return "paid"


def stream_cost(kind, horizon, c9, s_m2, s=None, n_rows_paid=0, train_fwd=0.0, extra_fwd=0.0):
    # compute (full forwards) of a stream of `horizon` inputs of the domain, the 3,000 audited cal rows included.
    #   m2       3,000 calibration audits, then M2
    #   paid     the adaptation first (n_rows_paid audits and the training), then ONE calibration, then the
    #            configuration: the story "the shift is detected, adapt, calibrate"
    #   free     M2 is calibrated and runs for N_COLLECT inputs while the rows that reach exit 9 are collected; then
    #            the training, and the new heads evaluated on the stored features of the 3,000 cal rows (extra_fwd):
    #            the backbone has not changed, so those rows need no new forward
    #   traffic  (bn_traffic, corrected by Fable after the note of the executor) as free, but the statistics need the
    #            stored inputs run again up to where they stopped (extra_fwd), and a change of the backbone makes the
    #            stored cal rows useless: 3,000 more audits for a second calibration
    #   eqb      the same 3,000 audits shared between adaptation and calibration
    if kind == "m2":
        return N_CAL * c9 + (horizon - N_CAL) * (1 - s_m2)
    if kind == "paid":
        return (N_CAL + n_rows_paid) * c9 + train_fwd + (horizon - N_CAL - n_rows_paid) * (1 - s)
    if kind == "free":
        return (N_CAL * c9 + N_COLLECT * (1 - s_m2) + train_fwd + extra_fwd
                + (horizon - N_CAL - N_COLLECT) * (1 - s))
    if kind == "traffic":
        return (2 * N_CAL * c9 + N_COLLECT * (1 - s_m2) + train_fwd + extra_fwd
                + (horizon - 2 * N_CAL - N_COLLECT) * (1 - s))
    if kind == "eqb":
        return N_CAL * c9 + train_fwd + (horizon - N_CAL) * (1 - s)
    raise ValueError(kind)


def net_savings(cost, horizon):
    return 1 - cost / horizon


# ---------------- outputs ----------------

def out_paths(out_dir, job, name):
    # job "bn" or "heads"
    return {"runs": os.path.join(out_dir, f"runs_{job}", f"{name}.csv"),
            "heads": os.path.join(out_dir, f"heads_{job}", f"{name}.csv"),
            "curves": os.path.join(out_dir, f"curves_{job}", f"{name}.npz")}


def todo_pairs(paths, severities, splits, force):
    # (severity, split) pairs still to compute for one (network, corruption)
    wanted = [(s, t) for s in severities for t in splits]
    if force or not os.path.exists(paths["runs"]):
        return wanted
    done = set(map(tuple, pd.read_csv(paths["runs"], usecols=["severity", "split"]).drop_duplicates().values))
    return [p for p in wanted if p not in done]


def atomic_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def atomic_npy(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        np.save(f, array)
    os.replace(path + ".tmp", path)


def keep_other_pairs(df, pairs):
    pairs = set(pairs)
    keep = [(s, t) not in pairs for s, t in zip(df["severity"], df["split"])]
    return df[np.array(keep, dtype=bool)]


def write_outputs(paths, runs, heads, curves, pairs):
    # merges the new rows with the file (rows of the recomputed (severity, split) pairs are replaced)
    for key, rows in [("runs", runs), ("heads", heads)]:
        new = pd.DataFrame(rows)
        if os.path.exists(paths[key]):
            new = pd.concat([keep_other_pairs(pd.read_csv(paths[key], dtype={"l2": str}), pairs), new],
                            ignore_index=True)
        atomic_csv(new, paths[key])
    arrays = {name: np.asarray([c[name] for c in curves]) for name in ["config", "severity", "split"]}
    arrays.update({name: np.stack([c[name] for c in curves]) for name in ["cost", "acc", "risk"]})
    if os.path.exists(paths["curves"]):
        old = dict(np.load(paths["curves"]))
        keep = keep_other_pairs(pd.DataFrame({"severity": old["severity"], "split": old["split"]}), pairs).index
        arrays = {name: np.concatenate([old[name][keep], arrays[name]]) for name in arrays}
    os.makedirs(os.path.dirname(paths["curves"]), exist_ok=True)
    with open(paths["curves"] + ".tmp", "wb") as f:
        np.savez(f, **arrays)
    os.replace(paths["curves"] + ".tmp", paths["curves"])


class Collector:
    # rows of runs_*.csv, heads_*.csv and curves_*.npz for one (network, corruption)
    def __init__(self, info, corruption):
        self.base = {**info, "corruption": corruption}
        self.runs, self.heads, self.curves = [], [], []

    def add(self, severity, split, config, measured, **extra):
        scalars, heads, curves = measured
        key = {**self.base, "severity": severity, "split": split, "config": config}
        self.runs.append({**key, **extra, **scalars})
        self.heads.extend({**key, **h} for h in heads)
        self.curves.append({"config": config, "severity": severity, "split": split, **curves})


def family_of(corruption):
    return FAMILY_OF[corruption]

