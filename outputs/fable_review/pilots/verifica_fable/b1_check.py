# Fable's independent check of pilot B1 (14 November 2026). It does NOT import the pilot's code: on a random sample
# of cases it computes again, from the saved logits and with separate code, the levels written in
# outputs/fable_review/pilots/B1_bilancio/b1_levels.csv, and compares.
#   ucb_3000                with the slow reference analysis.ucb_index (the pilot uses ucb_index_fast)
#   oracle_shared, shared_cal, acc_abs, acc_rel   the choice is made on integer counts
#   shared_cal_fine         the same search between two lambdas of the grid, written again
#   oracle_input, oracle_input_eps
#   oracle_exit_thresholds, exit_path_cal   the thresholds are read from the CSV and applied again
# Then a separate and simpler optimizer of the 8 thresholds (a greedy path on a quantile grid):
#   on the fit rows, against oracle_exit_thresholds (is the pilot's optimizer weak?)
#   fitted on adapt and stopped on cal, against exit_path_cal (does the fair gain depend on the optimizer?)
# It only prints. Usage (from the repository root): python outputs/fable_review/pilots/verifica_fable/b1_check.py [n]

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/6_block3_review")
from eesavings.analysis import LAMBDAS, make_splits, ucb_index  # noqa: E402
from eesavings.fys import softmax  # noqa: E402
from review_common import all_runs, fractions_of, load_labels, load_logits  # noqa: E402

EPS, DELTA = 0.05, 0.1
B1 = "outputs/fable_review/pilots/B1_bilancio"
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 16


def policy(conf8, t):
    # 0-based index of the exit used (8 = exit 9): the first exit whose confidence is strictly above its threshold
    passed = conf8 > t
    return np.where(passed.any(axis=1), passed.argmax(axis=1), 8)


def value(ex, rows, fr, dis9, correct):
    # savings, risk and system accuracy of the exits ex (given for the rows, in the same order)
    return 1 - fr[ex].mean(), dis9[rows, ex].mean(), correct[rows, ex].mean()


def greedy_path(conf8, dis9, fr, budget_frac=EPS, n_grid=200):
    # a separate optimizer: start with no early exit, at every step lower ONE threshold by one quantile step, the
    # step with the largest compute saved per unit of risk added, while the risk stays within budget_frac.
    # Returns the list of the threshold vectors met along the way (the last one is the final rule)
    n = len(conf8)
    rows = np.arange(n)
    budget = int(np.floor(budget_frac * n + 1e-9))
    grids = [np.concatenate([[1.0], np.quantile(conf8[:, l], np.linspace(1, 0, n_grid + 1))[1:] - 1e-12])
             for l in range(8)]
    idx = np.zeros(8, dtype=int)
    t = np.ones(8)
    ex = policy(conf8, t)
    cost, wrong = fr[ex].sum(), dis9[rows, ex].sum()
    trajectory = [t.copy()]
    while True:
        best = None
        for l in range(8):
            if idx[l] + 1 >= len(grids[l]):
                continue
            t2 = t.copy()
            t2[l] = grids[l][idx[l] + 1]
            ex2 = policy(conf8, t2)
            cost2, wrong2 = fr[ex2].sum(), dis9[rows, ex2].sum()
            if wrong2 > budget:
                continue
            if cost2 < cost - 1e-12:
                ratio = np.inf if wrong2 <= wrong else (cost - cost2) / (wrong2 - wrong)
            else:
                ratio = 0.0  # no compute saved: taken only if nothing better, it can open the next steps
            if best is None or ratio > best[0]:
                best = (ratio, l, t2, cost2, wrong2)
        if best is None:
            return trajectory
        _, l, t, cost, wrong = best
        idx[l] += 1
        trajectory.append(t.copy())


def fine_shared(conf8, dis9, fr, rows, k, n_line=200):
    # between LAMBDAS[k] and the next lambda of the grid: the cheapest lambda with a feasible count on the rows
    budget = int(np.floor(EPS * len(rows) + 1e-9))
    lam_hi = float(LAMBDAS[k])
    lam_lo = float(LAMBDAS[k + 1]) if k + 1 < len(LAMBDAS) else 0.0
    best_cost, best_lam = fr[policy(conf8[rows], np.full(8, lam_hi))].mean(), lam_hi
    for alpha in np.linspace(0, 1, n_line + 1)[1:]:
        lam = (1 - alpha) * lam_hi + alpha * lam_lo
        ex = policy(conf8[rows], np.full(8, lam))
        if dis9[rows, ex].sum() <= budget and fr[ex].mean() < best_cost:
            best_cost, best_lam = fr[ex].mean(), lam
    return best_lam


def check_case(df, dataset, regime, seed, corruption, severity, split):
    logits = load_logits(dataset, regime, seed, None if severity == 0 else corruption, severity)
    labels = load_labels(dataset, regime, seed)
    fr = fractions_of(dataset)
    n = len(logits)
    conf8 = np.stack([softmax(logits[:, l], axis=1).max(axis=1) for l in range(8)], axis=1).astype(np.float64)
    preds = logits.argmax(axis=2)
    dis9 = np.concatenate([preds[:, :8] != preds[:, 8:9], np.zeros((n, 1), dtype=bool)], axis=1)
    correct = preds == labels[:, None]
    adapt, cal, ev = make_splits(5)[split]
    fit = np.concatenate([adapt, cal])
    ex_all = np.stack([policy(conf8, np.full(8, lam)) for lam in LAMBDAS])  # (201, n), my own exits
    loss_all = dis9[np.arange(n)[None, :], ex_all]

    mine = {}
    k = ucb_index(loss_all[:, cal].astype(np.float32), EPS, DELTA)
    mine["ucb_3000"] = value(ex_all[k, ev], ev, fr, dis9, correct)

    def counts(rows):
        return (loss_all[:, rows].sum(axis=1), fr[ex_all[:, rows]].sum(axis=1),
                correct[rows[None, :], ex_all[:, rows]].sum(axis=1), correct[rows, 8].sum())

    wrong_fit, cost_fit, right_fit, right9 = counts(fit)
    wrong_cal, cost_cal, _, _ = counts(cal)
    feasible = {"oracle_shared": (wrong_fit <= np.floor(EPS * len(fit) + 1e-9), cost_fit),
                "acc_abs": ((right9 - right_fit) <= 0.05 * len(fit) + 1e-9, cost_fit),
                "acc_rel": ((right9 - right_fit) <= 0.05 * right9 + 1e-9, cost_fit),
                "shared_cal": (wrong_cal <= np.floor(EPS * len(cal) + 1e-9), cost_cal)}
    chosen = {}
    for name, (ok, cost) in feasible.items():
        chosen[name] = int(np.argmin(np.where(ok, cost, np.inf)))
        mine[name] = value(ex_all[chosen[name], ev], ev, fr, dis9, correct)
    lam_fine = fine_shared(conf8, dis9, fr, cal, chosen["shared_cal"])
    mine["shared_cal_fine"] = value(policy(conf8[ev], np.full(8, lam_fine)), ev, fr, dis9, correct)

    first = dis9.argmin(axis=1)  # first exit that agrees (exit 9 always does)
    fa = first[ev]
    mine["oracle_input"] = value(fa, ev, fr, dis9, correct)
    order = np.argsort(-(fr[fa] - fr[0]), kind="stable")
    moved = [i for i in order[:int(np.floor(EPS * len(ev) + 1e-9))] if fa[i] > 0]
    ex = fa.copy()
    ex[moved] = 0
    mine["oracle_input_eps"] = value(ex, ev, fr, dis9, correct)

    rows = df[(df.dataset == dataset) & (df.regime == regime) & (df.seed == seed) & (df.corruption == corruption)
              & (df.severity == severity) & (df.split == split)].set_index("level")
    out = []
    for level, (s, r, a) in mine.items():
        got = rows.loc[level]
        out.append({"level": level, "d_savings": got.savings - s, "d_risk": got.risk - r, "d_acc": got.acc_system - a})
    out.append({"level": "shared_cal_fine (lambda)", "d_savings": rows.loc["shared_cal_fine"].lam - lam_fine,
                "d_risk": 0.0, "d_acc": 0.0})
    # the two rules with 8 thresholds: apply the thresholds of the CSV again
    applied = {}
    for level, chosen_rows in [("oracle_exit_thresholds", fit), ("exit_path_cal", cal)]:
        got = rows.loc[level]
        t = np.array([float(x) for x in str(got.thresholds).split()])
        s_ev, r_ev, a_ev = value(policy(conf8[ev], t), ev, fr, dis9, correct)
        s_ch, r_ch, _ = value(policy(conf8[chosen_rows], t), chosen_rows, fr, dis9, correct)
        out.append({"level": f"{level} (applied again)", "d_savings": got.savings - s_ev, "d_risk": got.risk - r_ev,
                    "d_acc": got.acc_system - a_ev})
        out.append({"level": f"{level} (on the rows where chosen)", "d_savings": got.savings_fit - s_ch,
                    "d_risk": got.risk_fit - r_ch, "d_acc": 0.0})
        applied[level] = (s_ev, r_ev, s_ch, r_ch)

    # my greedy path on the fit rows, against oracle_exit_thresholds
    tg = greedy_path(conf8[fit], dis9[fit], fr)[-1]
    g_fit, gr_fit, _ = value(policy(conf8[fit], tg), fit, fr, dis9, correct)
    g_ev, gr_ev, _ = value(policy(conf8[ev], tg), ev, fr, dis9, correct)
    # my greedy path fitted on adapt (it may go up to twice eps there) and stopped on cal, against exit_path_cal
    budget_cal = int(np.floor(EPS * len(cal) + 1e-9))
    best_cost, tp = np.inf, np.ones(8)
    for t in greedy_path(conf8[adapt], dis9[adapt], fr, budget_frac=2 * EPS):
        ex = policy(conf8[cal], t)
        if dis9[cal, ex].sum() <= budget_cal and fr[ex].mean() < best_cost:
            best_cost, tp = fr[ex].mean(), t
    p_ev, pr_ev, _ = value(policy(conf8[ev], tp), ev, fr, dis9, correct)
    pr_cal = dis9[cal, policy(conf8[cal], tp)].mean()
    shared_fit = 1 - cost_fit[chosen["oracle_shared"]] / len(fit)
    info = {"case": f"{dataset}/{regime}/s{seed} {corruption} sev{severity} split{split}",
            "exec_fit": applied["oracle_exit_thresholds"][2], "greedy_fit": g_fit, "shared_fit": shared_fit,
            "exec_eval": applied["oracle_exit_thresholds"][0], "greedy_eval": g_ev,
            "exec_risk_eval": applied["oracle_exit_thresholds"][1], "greedy_risk_eval": gr_ev,
            "path_eval": applied["exit_path_cal"][0], "path_risk_eval": applied["exit_path_cal"][1],
            "gpath_eval": p_ev, "gpath_risk_eval": pr_ev, "gpath_risk_cal": pr_cal,
            "shared_fine_eval": mine["shared_cal_fine"][0], "shared_fine_risk_eval": mine["shared_cal_fine"][1]}
    return out, info


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    df = pd.read_csv(f"{B1}/b1_levels.csv")
    rng = np.random.default_rng(20261114)
    runs = all_runs()
    domains = df[["corruption", "severity"]].drop_duplicates().values.tolist()
    cases = []
    while len(cases) < N_SAMPLE:
        d, r, s = runs[rng.integers(len(runs))]
        c, sev = domains[rng.integers(len(domains))]
        cases.append((d, r, s, c, int(sev), int(rng.integers(5))))
    diffs, infos = [], []
    for case in cases:
        out, info = check_case(df, *case)
        for o in out:
            diffs.append({"case": info["case"], **o})
        infos.append(info)
        print("done", info["case"], flush=True)
    diffs = pd.DataFrame(diffs)
    worst = diffs.groupby("level", sort=False)[["d_savings", "d_risk", "d_acc"]].agg(lambda x: np.abs(x).max())
    print("\nLargest absolute difference between b1_levels.csv and my recomputation, by level\n", worst.to_string())
    bad = diffs[(diffs[["d_savings", "d_risk", "d_acc"]].abs() > 1e-9).any(axis=1)]
    print("\nRows that differ by more than 1e-9:", len(bad))
    if len(bad):
        print(bad.to_string())
    infos = pd.DataFrame(infos).set_index("case")
    print("\nRule with 8 thresholds fitted on the 7,000 fit rows: the pilot's optimizer against my greedy path\n",
          infos[["shared_fit", "exec_fit", "greedy_fit", "exec_eval", "greedy_eval", "exec_risk_eval",
                 "greedy_risk_eval"]].round(4).to_string())
    print("mean pilot - greedy, savings on fit:", round((infos.exec_fit - infos.greedy_fit).mean(), 4), "; smallest:",
          round((infos.exec_fit - infos.greedy_fit).min(), 4))
    print("\nFair pair (path on adapt, stopped on cal): the pilot's path, my greedy path, the fine shared threshold\n",
          infos[["path_eval", "gpath_eval", "shared_fine_eval", "path_risk_eval", "gpath_risk_eval",
                 "shared_fine_risk_eval", "gpath_risk_cal"]].round(4).to_string())
    print("mean fair gain on eval: pilot's path", round((infos.path_eval - infos.shared_fine_eval).mean(), 4),
          "; my greedy path", round((infos.gpath_eval - infos.shared_fine_eval).mean(), 4),
          "; mean risk on eval: pilot's path", round(infos.path_risk_eval.mean(), 4), ", greedy path",
          round(infos.gpath_risk_eval.mean(), 4), ", fine shared", round(infos.shared_fine_risk_eval.mean(), 4))


if __name__ == "__main__":
    main()
