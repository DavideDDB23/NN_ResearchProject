# Pilot B1 of the review: the compute budget of the savings under shift. On the saved logits of the multi exit
# ResNet20 (max softmax score, eps 0.05, delta 0.1) it measures, for every network, domain (clean and the 15
# corruptions at severity 1, 3, 5) and split of analysis.make_splits(5), several levels of compute savings:
#   ucb_3000, ucb_1000, ucb_250   shared lambda calibrated with UCB (WSR) on the first n rows of cal (M2)
#   crc_3000, crc_250, crc_100    shared lambda calibrated with CRC on the first n rows of cal
#   oracle_shared                 shared lambda with the largest savings and mean loss <= eps on fit (adapt + cal)
#   acc_abs, acc_rel              shared lambda with the largest savings that loses at most 0.05 of accuracy on fit
#                                 (absolute, or relative to the accuracy of exit 9)
#   oracle_exit_thresholds        8 thresholds, one per exit, fitted on fit (fit_exit_thresholds below)
#   oracle_input                  every input leaves at the first exit that agrees with exit 9 (risk 0)
#   oracle_input_eps              as oracle_input, with floor(eps * 3000) rows moved to exit 1
# Added by Fable after the trial run (14 November): the 8 thresholds of oracle_exit_thresholds fill the risk budget
# on the 7,000 fit rows and on fresh rows they are above it (0.054 to 0.056 on eval in the trial, against 0.047 to
# 0.049 of oracle_shared), so that pair is not compared at the same risk. The fair pair, where ONE number is chosen
# on the cal rows in both cases:
#   shared_cal                    shared lambda of the grid with the largest savings and mean loss <= eps on cal
#   shared_cal_fine               the same, searched also between that lambda and the next one of the grid
#   exit_path_cal                 8 thresholds along a path fitted on adapt only (fit_exit_path below), stopped on cal
# The fair gain of the thresholds per exit is exit_path_cal - shared_cal_fine: both fill the budget on cal.
# Every level is measured on the eval rows of the split. Specification: SPECIFICA.md in this folder.
# Reads outputs/cifar only. Writes in --out-dir: runs/<network>.pkl (one file per network, networks already done are
# skipped unless --force), then b1_levels.csv, b1_frontier.csv and b1_exits.csv.
#
# Usage (from the root of the repository):
#   python outputs/fable_review/pilots/B1_bilancio/b1_levels.py [--workers 3] [--runs cifar10/joint/seed0 ...]
#       [--severities 0 1 3 5] [--corruptions frost ...] [--out-dir DIR] [--force]

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "scripts", "6_block3_review"))

from eesavings.analysis import LAMBDAS, consistency_losses, make_splits, ucb_index_fast  # noqa: E402
from eesavings.data.cifar import CORRUPTIONS  # noqa: E402
from eesavings.fys import crc_lam  # noqa: E402
from eesavings.scores import exit_scores  # noqa: E402
from review_common import all_runs, fractions_of, load_labels, load_logits, run_dir  # noqa: E402

EPS, DELTA = 0.05, 0.1
ACC_DROP = 0.05
UCB_NS = [3000, 1000, 250]
CRC_NS = [3000, 250, 100]
MUS = np.geomspace(50, 0.2, 25)  # Lagrange multipliers of the loss, from the largest to the smallest
N_BISECT = 8
N_PATH = 40  # multipliers of the path of fit_exit_path (same range as MUS, finer)
N_LINE = 200  # points of the last segment of that path tried on cal
MAX_SWEEPS = 10
TOL = 1e-12  # rounding of the comparisons with eps


# ---------------- the rule with 8 thresholds (section 5 of the specification) ----------------
# conf8 (n, 8): confidences of exits 1..8; loss9 (n, 9): 1 where the exit disagrees with exit 9; fr (9,): cost of
# every exit. Exits are 0-based here: 8 is exit 9.

def policy_exits(conf8, t):
    # exit of every row: the first l with conf8[:, l] > t[l], otherwise exit 9
    above = conf8 > t
    return np.where(above.any(axis=1), above.argmax(axis=1), conf8.shape[1])


def rule_cost_risk(exits, loss9, fr):
    # mean cost and mean loss of the rows when row i leaves at exits[i]
    return fr[exits].mean(), loss9[np.arange(len(exits)), exits].mean()


def is_feasible(risk, eps):
    return risk <= eps + TOL


def exit_orders(conf8):
    # per exit, computed once: rows sorted by confidence descending, the allowed prefix lengths (0, n, and the cuts
    # between two different confidences) and the threshold of each (1.0, 0.0, and the midpoints)
    n, out = len(conf8), []
    for l in range(conf8.shape[1]):
        order = np.argsort(-conf8[:, l], kind="stable")
        cs = conf8[order, l]
        inner = np.flatnonzero(cs[:-1] > cs[1:]) + 1
        cuts = np.concatenate([[0], inner, [n]])
        thresholds = np.concatenate([[1.0], (cs[inner - 1] + cs[inner]) / 2, [0.0]])
        out.append((order, cuts, thresholds))
    return out


def next_exits(above, l):
    # exit each row would use from the exits after l (the first j > l above its threshold, otherwise exit 9)
    if l + 1 == above.shape[1]:
        return np.full(len(above), above.shape[1])
    after = above[:, l + 1:]
    return np.where(after.any(axis=1), l + 1 + after.argmax(axis=1), above.shape[1])


def coordinate_step(l, conf8, loss9, fr, t, mu, orders=None, cost=None):
    # exact best t[l] for J = mean cost + mu * mean loss, with the other thresholds fixed. A row that reaches exit l
    # and leaves there changes J by g = cost at l - cost at the exit it would use otherwise; the rows that leave at l
    # are a prefix of the rows sorted by conf8[:, l]: take the prefix with the smallest sum of g
    orders = exit_orders(conf8) if orders is None else orders
    cost = fr[None, :] + mu * loss9 if cost is None else cost
    above = conf8 > t
    reach = ~above[:, :l].any(axis=1)
    d = next_exits(above, l)
    rows = np.arange(len(conf8))
    g = np.where(reach, cost[:, l] - cost[rows, d], 0.0)
    order, cuts, thresholds = orders[l]
    csum = np.concatenate([[0.0], np.cumsum(g[order])])
    return thresholds[np.argmin(csum[cuts])]  # ties: the fewest rows leaving


def lagrangian_descent(conf8, loss9, fr, t0, mu, orders=None):
    # sweeps l = 7, 6, ..., 0 of coordinate_step until no threshold changes, at most MAX_SWEEPS sweeps
    orders = exit_orders(conf8) if orders is None else orders
    cost = fr[None, :] + mu * loss9
    t = np.array(t0, dtype=float)
    for _ in range(MAX_SWEEPS):
        changed = False
        for l in range(len(t) - 1, -1, -1):
            new = coordinate_step(l, conf8, loss9, fr, t, mu, orders, cost)
            changed |= new != t[l]
            t[l] = new
        if not changed:
            break
    return t


def best_shared_index(conf8, loss9, fr, eps):
    # index in LAMBDAS of the shared threshold with the lowest mean cost and mean loss <= eps (ties: smallest index)
    costs, risks = zip(*[rule_cost_risk(policy_exits(conf8, np.full(conf8.shape[1], lam)), loss9, fr)
                         for lam in LAMBDAS])
    return int(np.argmin(np.where(is_feasible(np.array(risks), eps), costs, np.inf)))


def greedy_lowering(conf8, loss9, fr, eps, t, orders):
    # last pass: one threshold at a time, lowered to the cut with the lowest mean cost that keeps mean loss <= eps
    # (more rows leave at l, so the cost can only go down), until no threshold changes
    n, rows = len(conf8), np.arange(len(conf8))
    t = t.copy()
    for _ in range(MAX_SWEEPS):
        changed = False
        for l in range(len(t) - 1, -1, -1):
            base = t.copy()
            base[l] = 1.0  # nobody leaves at l (as the empty prefix of exit_orders)
            base_cost, base_risk = rule_cost_risk(policy_exits(conf8, base), loss9, fr)
            above = conf8 > base
            reach = ~above[:, :l].any(axis=1)
            d = next_exits(above, l)
            order, cuts, thresholds = orders[l]
            dcost = np.where(reach, fr[l] - fr[d], 0.0)[order]
            drisk = np.where(reach, loss9[:, l].astype(float) - loss9[rows, d], 0.0)[order]
            cost = base_cost + np.concatenate([[0.0], np.cumsum(dcost)])[cuts] / n
            risk = base_risk + np.concatenate([[0.0], np.cumsum(drisk)])[cuts] / n
            current_cost, _ = rule_cost_risk(policy_exits(conf8, t), loss9, fr)
            ok = is_feasible(risk, eps)
            i = int(np.argmin(np.where(ok, cost, np.inf)))
            if ok[i] and cost[i] < current_cost - 1e-9:
                t[l] = thresholds[i]
                changed = True
        if not changed:
            break
    return t


def fit_exit_thresholds(conf8, loss9, fr, eps, lam_shared=None, greedy=True):
    # 8 thresholds with mean loss <= eps and the lowest mean cost found on these rows: a heuristic, so a lower bound
    # of the best rule of the class. Path of mu from MUS (each descent starts from the previous solution), then
    # bisection in log(mu) between the smallest feasible mu and the next one, then the greedy pass
    orders = exit_orders(conf8)
    if lam_shared is None:
        lam_shared = LAMBDAS[best_shared_index(conf8, loss9, fr, eps)]
    t_shared = np.full(conf8.shape[1], float(lam_shared))
    shared_cost, shared_risk = rule_cost_risk(policy_exits(conf8, t_shared), loss9, fr)
    assert is_feasible(shared_risk, eps), "the shared start must be feasible"
    best = [shared_cost, t_shared]

    def solve(t0, mu):
        t = lagrangian_descent(conf8, loss9, fr, t0, mu, orders)
        cost, risk = rule_cost_risk(policy_exits(conf8, t), loss9, fr)
        if is_feasible(risk, eps) and cost < best[0]:
            best[:] = [cost, t]
        return t, is_feasible(risk, eps)

    path, t = [], t_shared
    for mu in MUS:
        t, ok = solve(t, mu)
        path.append((mu, t, ok))
    feasible = [i for i, (_, _, ok) in enumerate(path) if ok]
    if feasible and feasible[-1] + 1 < len(path):
        mu_hi, t_hi = path[feasible[-1]][:2]
        mu_lo = path[feasible[-1] + 1][0]
        for _ in range(N_BISECT):
            mu = np.sqrt(mu_hi * mu_lo)
            t, ok = solve(t_hi, mu)
            if ok:
                mu_hi, t_hi = mu, t
            else:
                mu_lo = mu
    t = greedy_lowering(conf8, loss9, fr, eps, best[1], orders) if greedy else best[1]
    cost, risk = rule_cost_risk(policy_exits(conf8, t), loss9, fr)
    assert is_feasible(risk, eps) and cost <= shared_cost + TOL, "invariants of fit_exit_thresholds"
    return t


def fit_exit_path(conf_a, loss_a, conf_c, loss_c, fr, eps):
    # the rule with 8 thresholds compared fairly with a shared threshold. The thresholds follow the Lagrangian path
    # fitted on the adapt rows only (conf_a, loss_a: one solution per multiplier, each descent starting from the
    # previous one, the first from the best shared threshold on adapt). The cal rows (conf_c, loss_c) choose only
    # where to stop along the path: the solution with the lowest mean cost and mean loss <= eps on cal, refined by
    # bisection between the last multiplier that is feasible on cal and the next one. One number is chosen on cal, as
    # for a shared lambda, so the 8 thresholds do not push the risk on fresh rows above eps. No greedy pass. If no
    # point of the path is feasible on cal, nobody leaves early (all thresholds 1).
    orders = exit_orders(conf_a)
    n_exits = conf_a.shape[1]
    best = [fr[n_exits], np.ones(n_exits)]

    def solve(t0, mu):
        t = lagrangian_descent(conf_a, loss_a, fr, t0, mu, orders)
        cost, risk = rule_cost_risk(policy_exits(conf_c, t), loss_c, fr)
        if is_feasible(risk, eps) and cost < best[0]:
            best[:] = [cost, t]
        return t, is_feasible(risk, eps)

    path, t = [], np.full(n_exits, float(LAMBDAS[best_shared_index(conf_a, loss_a, fr, eps)]))
    for mu in np.geomspace(MUS[0], MUS[-1], N_PATH):
        t, ok = solve(t, mu)
        path.append((mu, t, ok))
    feasible = [i for i, (_, _, ok) in enumerate(path) if ok]
    last = feasible[-1] if feasible else -1
    # t_hi: the last solution of the path that is feasible on cal (nobody leaves early if there is none);
    # t_lo: the next one (everybody leaves at exit 1 if the path is over)
    t_hi = path[last][1] if last >= 0 else np.ones(n_exits)
    t_lo = path[last + 1][1] if last + 1 < len(path) else np.zeros(n_exits)
    if 0 <= last < len(path) - 1:
        mu_hi, mu_lo = path[last][0], path[last + 1][0]
        for _ in range(N_BISECT):
            mu = np.sqrt(mu_hi * mu_lo)
            t, ok = solve(t_hi, mu)
            if ok:
                mu_hi, t_hi = mu, t
            else:
                mu_lo, t_lo = mu, t
    # the solutions jump from one multiplier to the next, and the jump can leave part of the budget unused: go on
    # along the segment from t_hi to t_lo and keep its cheapest point that is feasible on cal. It is still one
    # position along one path fitted on adapt
    rows = np.arange(len(conf_c))
    for alpha in np.linspace(0, 1, N_LINE + 1)[1:]:
        t = (1 - alpha) * t_hi + alpha * t_lo
        exits = policy_exits(conf_c, t)
        cost, risk = fr[exits].mean(), loss_c[rows, exits].mean()
        if is_feasible(risk, eps) and cost < best[0]:
            best[:] = [cost, t]
    return best[1]


def refine_shared(conf_c, loss_c, fr, eps, k):
    # the shared threshold of the grid leaves unused the part of the budget between LAMBDAS[k] and the next lambda;
    # fit_exit_path does not (its last segment is searched finely). For a fair pair, the same search here: the
    # cheapest lambda that is feasible on these rows among N_LINE points between LAMBDAS[k] and LAMBDAS[k + 1]
    lam_hi = float(LAMBDAS[k])
    lam_lo = float(LAMBDAS[k + 1]) if k + 1 < len(LAMBDAS) else 0.0
    rows, n_exits = np.arange(len(conf_c)), conf_c.shape[1]
    exits = policy_exits(conf_c, np.full(n_exits, lam_hi))
    best = [fr[exits].mean(), lam_hi]
    for alpha in np.linspace(0, 1, N_LINE + 1)[1:]:
        lam = (1 - alpha) * lam_hi + alpha * lam_lo
        exits = policy_exits(conf_c, np.full(n_exits, lam))
        cost, risk = fr[exits].mean(), loss_c[rows, exits].mean()
        if is_feasible(risk, eps) and cost < best[0]:
            best[:] = [cost, lam]
    return best[1]


# ---------------- the input oracles ----------------

def first_agreeing_exit(loss9):
    # first exit (0-based) that agrees with exit 9; exit 9 always agrees
    return loss9.argmin(axis=1)


def oracle_input_eps_exits(first_agree, fr, eps):
    # first agreeing exit of every row, with floor(eps * n) rows moved to exit 1: the rows whose first agreeing exit
    # is not exit 1, in decreasing order of fr[first agreeing exit] - fr[0] (stable sort). Returns exits, n moved
    n_move = int(np.floor(eps * len(first_agree) + 1e-9))
    candidates = np.flatnonzero(first_agree > 0)
    candidates = candidates[np.argsort(-(fr[first_agree[candidates]] - fr[0]), kind="stable")]
    moved = candidates[:n_move]
    exits = first_agree.copy()
    exits[moved] = 0
    return exits, len(moved)


# ---------------- one domain ----------------

def case_arrays(logits, labels):
    # what every level needs from the logits (N, 9, C) of one domain
    conf = exit_scores(logits, "max_softmax")
    preds = logits.argmax(axis=2)
    loss9 = preds != preds[:, -1:]
    losses, exits = consistency_losses(logits, conf)
    return {"conf": conf, "loss9": loss9, "correct": preds == labels[:, None], "losses": losses,
            "exits": exits - 1, "first_agree": first_agreeing_exit(loss9)}


def measure(c, rows, exits, fr):
    # savings, risk and accuracies of the rows when rows[i] leaves at exits[i] (0-based)
    return {"savings": 1.0 - fr[exits].mean(), "risk": c["loss9"][rows, exits].mean(),
            "acc_system": c["correct"][rows, exits].mean(), "acc_exit9": c["correct"][rows, -1].mean()}


def best_index(savings, feasible):
    # largest savings among the feasible lambdas, ties to the smallest index
    return int(np.argmax(np.where(feasible, savings, -np.inf)))


def shared_choices(c, fit, cal, fr):
    # index k of every shared threshold level, and the rows where it was chosen
    losses, exits, correct = c["losses"], c["exits"], c["correct"]
    choices = {f"ucb_{n}": (int(ucb_index_fast(losses[:, cal[:n]], EPS, DELTA)), cal[:n]) for n in UCB_NS}
    choices.update({f"crc_{n}": (int(crc_lam(losses[:, cal[:n]], EPS)), cal[:n]) for n in CRC_NS})
    risk_fit = losses[:, fit].mean(axis=1, dtype=np.float64)
    savings_fit = 1.0 - fr[exits[:, fit]].mean(axis=1)
    acc_fit = correct[fit[None, :], exits[:, fit]].mean(axis=1)
    acc9 = correct[fit, -1].mean()
    choices["oracle_shared"] = (best_index(savings_fit, is_feasible(risk_fit, EPS)), fit)
    choices["acc_abs"] = (best_index(savings_fit, acc9 - acc_fit <= ACC_DROP + TOL), fit)
    choices["acc_rel"] = (best_index(savings_fit, acc9 - acc_fit <= ACC_DROP * acc9 + TOL), fit)
    # the shared threshold chosen on cal only: the fair term of comparison of exit_path_cal
    risk_cal = losses[:, cal].mean(axis=1, dtype=np.float64)
    savings_cal = 1.0 - fr[exits[:, cal]].mean(axis=1)
    choices["shared_cal"] = (best_index(savings_cal, is_feasible(risk_cal, EPS)), cal)
    return choices


def split_rows(c, fr, split, key):
    # one row per level for one split of one domain
    adapt, cal, ev = split
    fit = np.concatenate([adapt, cal])
    empty = {"lam": np.nan, "k": None, "savings_fit": np.nan, "risk_fit": np.nan, "thresholds": "",
             "n_moved": None, "seconds": np.nan}
    rows = []
    for level, (k, chosen) in shared_choices(c, fit, cal, fr).items():
        on_fit = measure(c, chosen, c["exits"][k, chosen], fr)
        rows.append({**key, "level": level, **measure(c, ev, c["exits"][k, ev], fr), **empty, "lam": LAMBDAS[k],
                     "k": k, "savings_fit": on_fit["savings"], "risk_fit": on_fit["risk"]})
    shared = rows[[r["level"] for r in rows].index("oracle_shared")]

    # shared_cal searched as finely as the last segment of fit_exit_path: the fair term of comparison of exit_path_cal
    k_cal = rows[[r["level"] for r in rows].index("shared_cal")]["k"]
    lam_fine = refine_shared(c["conf"][cal, :-1], c["loss9"][cal], fr, EPS, k_cal)
    t_fine = np.full(c["conf"].shape[1] - 1, lam_fine)
    on_cal = measure(c, cal, policy_exits(c["conf"][cal, :-1], t_fine), fr)
    assert is_feasible(on_cal["risk"], EPS)
    rows.append({**key, "level": "shared_cal_fine", **measure(c, ev, policy_exits(c["conf"][ev, :-1], t_fine), fr),
                 **empty, "lam": lam_fine, "k": k_cal, "savings_fit": on_cal["savings"], "risk_fit": on_cal["risk"]})

    start = time.time()
    t = fit_exit_thresholds(c["conf"][fit, :-1], c["loss9"][fit], fr, EPS, lam_shared=shared["lam"])
    seconds = time.time() - start
    on_fit = measure(c, fit, policy_exits(c["conf"][fit, :-1], t), fr)
    assert is_feasible(on_fit["risk"], EPS) and on_fit["savings"] >= shared["savings_fit"] - TOL
    rows.append({**key, "level": "oracle_exit_thresholds", **measure(c, ev, policy_exits(c["conf"][ev, :-1], t), fr),
                 **empty, "savings_fit": on_fit["savings"], "risk_fit": on_fit["risk"],
                 "thresholds": " ".join(format(x, ".17g") for x in t), "seconds": seconds})

    # the fair version: path on adapt, stopping point on cal (savings_fit and risk_fit are on cal, where it is chosen)
    start = time.time()
    tp = fit_exit_path(c["conf"][adapt, :-1], c["loss9"][adapt], c["conf"][cal, :-1], c["loss9"][cal], fr, EPS)
    seconds = time.time() - start
    on_cal = measure(c, cal, policy_exits(c["conf"][cal, :-1], tp), fr)
    assert is_feasible(on_cal["risk"], EPS)
    rows.append({**key, "level": "exit_path_cal", **measure(c, ev, policy_exits(c["conf"][ev, :-1], tp), fr),
                 **empty, "savings_fit": on_cal["savings"], "risk_fit": on_cal["risk"],
                 "thresholds": " ".join(format(x, ".17g") for x in tp), "seconds": seconds})

    first = c["first_agree"][ev]
    rows.append({**key, "level": "oracle_input", **measure(c, ev, first, fr), **empty})
    exits, n_moved = oracle_input_eps_exits(first, fr, EPS)
    rows.append({**key, "level": "oracle_input_eps", **measure(c, ev, exits, fr), **empty, "n_moved": n_moved})
    return rows


def frontier_rows(c, fr, key):
    # cost, risk and accuracy of every lambda of the grid on all the rows of the domain
    exits = c["exits"]
    rows = np.arange(exits.shape[1])[None, :]
    return pd.DataFrame({**key, "k": np.arange(len(LAMBDAS)), "lam": LAMBDAS, "cost": fr[exits].mean(axis=1),
                         "risk": c["loss9"][rows, exits].mean(axis=1),
                         "acc_system": c["correct"][rows, exits].mean(axis=1)})


def exit_rows(c, key):
    # accuracy of every exit, agreement with exit 9, share of inputs whose first agreeing exit is this one
    n_exits = c["loss9"].shape[1]
    return pd.DataFrame({**key, "exit": np.arange(1, n_exits + 1), "acc": c["correct"].mean(axis=0),
                         "agree": 1.0 - c["loss9"].mean(axis=0),
                         "oracle_pi": np.bincount(c["first_agree"], minlength=n_exits) / len(c["first_agree"])})


# ---------------- one network ----------------

def run_path(out_dir, dataset, regime, seed):
    return os.path.join(out_dir, "runs", f"{dataset}_{regime}_seed{seed}.pkl")


def one_network(job):
    (dataset, regime, seed), domains, out_dir, force = job
    path = run_path(out_dir, dataset, regime, seed)
    if os.path.exists(path) and not force:
        print(dataset, regime, seed, "already done, skipped", flush=True)
        return path
    start = time.time()
    labels = load_labels(dataset, regime, seed)
    labels_c = np.load(os.path.join(run_dir(dataset, regime, seed), "preds", "cifar_c", "labels.npy"))
    assert np.array_equal(labels, labels_c), "the CIFAR-C labels must be the test labels in the same order"
    fr = fractions_of(dataset)
    splits = make_splits(5)
    levels, frontier, exits = [], [], []
    for corruption, severity in domains:
        logits = load_logits(dataset, regime, seed, None if severity == 0 else corruption, severity)
        c = case_arrays(logits, labels)
        key = {"dataset": dataset, "regime": regime, "seed": seed, "corruption": corruption, "severity": severity}
        frontier.append(frontier_rows(c, fr, key))
        exits.append(exit_rows(c, key))
        for t, split in enumerate(splits):
            levels += split_rows(c, fr, split, {**key, "split": t})
    levels = pd.DataFrame(levels)
    levels["k"] = levels["k"].astype("Int64")
    levels["n_moved"] = levels["n_moved"].astype("Int64")
    tmp = path + ".tmp"
    pd.to_pickle({"levels": levels, "frontier": pd.concat(frontier, ignore_index=True),
                  "exits": pd.concat(exits, ignore_index=True)}, tmp, compression=None)
    os.replace(tmp, path)
    print(dataset, regime, seed, "done in %.1f s" % (time.time() - start), flush=True)
    return path


def parse_runs(specs):
    # "cifar10/joint/seed0" -> ("cifar10", "joint", 0); no option: all 24 networks
    if not specs:
        return all_runs()
    runs = []
    for spec in specs:
        dataset, regime, seed = spec.split("/")
        assert (dataset, regime, int(seed.removeprefix("seed"))) in all_runs(), spec
        runs.append((dataset, regime, int(seed.removeprefix("seed"))))
    return runs


def domains_of(corruptions, severities):
    assert set(corruptions) <= set(CORRUPTIONS) and set(severities) <= {0, 1, 2, 3, 4, 5}
    clean = [("none", 0)] if 0 in severities else []
    return clean + [(c, s) for c in corruptions for s in severities if s > 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--severities", nargs="*", type=int, default=[0, 1, 3, 5])
    parser.add_argument("--corruptions", nargs="*", default=list(CORRUPTIONS))
    parser.add_argument("--out-dir", default=HERE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out_dir, "runs"), exist_ok=True)
    domains = domains_of(args.corruptions, args.severities)
    jobs = [(run, domains, args.out_dir, args.force) for run in parse_runs(args.runs)]
    start = time.time()
    with Pool(args.workers) as pool:
        paths = pool.map(one_network, jobs, chunksize=1)
    tables = [pd.read_pickle(p, compression=None) for p in paths]
    for name in ("levels", "frontier", "exits"):
        df = pd.concat([t[name] for t in tables], ignore_index=True)
        df.to_csv(os.path.join(args.out_dir, f"b1_{name}.csv"), index=False)
        print("saved", f"b1_{name}.csv", len(df), "rows")
    print("total %.1f s" % (time.time() - start))


if __name__ == "__main__":
    main()

