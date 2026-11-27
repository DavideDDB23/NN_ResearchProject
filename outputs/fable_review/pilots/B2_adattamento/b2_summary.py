# Pilot B2: checks, accounting, tables and figures from the outputs of job A (b2_bn.py) and job B (b2_heads.py)
# (SPECIFICA.md, section 7). Works with only one of the two jobs, or with a subset of the cases.
#   checks      (a) m2 = the M2 rows of results.csv; (b) head_grid_n4000 = the M8b rows (max softmax);
#               (c) c_bn_n4000 (job B, from the cache) = bn_n4000 (job A); (d) the L-BFGS evaluations and training
#               compute of head_grid_n4000 = outputs/block3_review/adaptation_cost.csv; (e) source (job A) = m2 (job B)
#   accounting  gain over m2, the compute of the adaptation (paid rows and training, in full forwards), paybacks and
#               net savings over a stream of 10,000 / 100,000 / 1,000,000 inputs of the domain
#   tables      b2_table_configs.csv (and b2_table_free.csv, b2_table_bn.csv, b2_table_d3_heads.csv,
#               b2_table_m8a.csv), the hypotheses D1, kill, D2, D3 printed with their numbers
#   figures     b2_gain_vs_compute.png, b2_frontier.png
# Everything printed also goes to <dir>/b2_summary.log.
#
# Usage (from the repository root):
#   python outputs/fable_review/pilots/B2_adattamento/b2_summary.py [--dir outputs/fable_review/pilots/B2_adattamento]

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b2_common import (EPS, HORIZONS, N_CAL, PILOT, adaptation_rows_cost, c9_of, cost_kind, family_of,  # noqa: E402
                       net_savings, payback, stream_cost)
from eesavings.analysis import LAMBDAS  # noqa: E402
from eesavings.data.cifar import CORRUPTIONS, NUM_CLASSES  # noqa: E402
from eesavings.models import FEATURE_DIMS  # noqa: E402
from eesavings.savings import RESNET20_CIFAR_CUM_MMACS, RESNET20_FULL_MACS  # noqa: E402
from review_common import RESULTS, start_log  # noqa: E402

KEY = ["dataset", "regime", "seed", "corruption", "severity", "split"]
HG = "head_grid_n4000"
BASE_COLS = ["savings", "risk", "acc_system", "savings_acc02", "savings_acc05", "k_acc05", "dis_exit_right",
             "dis_ref_right", "dis_both_wrong"]
COMPARED = ["lambda", "k", "risk", "savings", "valid_savings", "acc_system", "acc_exit9", "acc_exit9_src",
            "savings_acc02", "savings_acc05", "acc_loss_acc02", "acc_loss_acc05"]
ADAPTATION_COST = "outputs/block3_review/adaptation_cost.csv"
D1_CONFIGS = [f"head_{k}_n4000" for k in ["k5", "k10", "k20"]] + [f"logit_{o}_n4000" for o in
                                                                    ["grid", "k5", "k10", "k20"]]
BN_CONFIGS = ["bn_n500", "bn_n1000", "bn_n2000", "bn_n4000", "bn_traffic"]
FRONTIER = ["m2", HG, "head_k10_n1000", "bn_n4000", "c_head_grid_n4000"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)


# ---------------- reading ----------------

def read_folder(d, sub):
    files = sorted(glob.glob(os.path.join(d, sub, "*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, dtype={"l2": str}) for f in files], ignore_index=True)


def parse_name(name):
    # "<dataset>_<regime>_seed<k>_<corruption>" -> (dataset, regime, seed, corruption)
    corruption = next(c for c in CORRUPTIONS if name.endswith("_" + c))
    rest = name[:-len(corruption) - 1]
    dataset = rest.split("_")[0]
    return dataset, rest[len(dataset) + 1:rest.rindex("_seed")], int(rest.rsplit("_seed", 1)[1]), corruption


def read_curves(d, sub):
    # {(dataset, regime, seed, corruption, severity, split, config): {"cost", "acc", "risk"}}
    out = {}
    for f in sorted(glob.glob(os.path.join(d, sub, "*.npz"))):
        case = parse_name(os.path.basename(f)[:-4])
        z = np.load(f)
        for i, (config, sev, split) in enumerate(zip(z["config"], z["severity"], z["split"])):
            out[case + (int(sev), int(split), str(config))] = {k: z[k][i] for k in ["cost", "acc", "risk"]}
    return out


def read_results():
    cols = ["dataset", "regime", "seed", "score", "eps", "method", "corruption", "severity", "split", "lambda",
            "risk", "savings", "l2"]
    df = pd.read_csv(RESULTS, usecols=cols, dtype={"l2": str})
    return df[(df.seed == 0) & np.isclose(df.eps, 0.05) & df.method.isin(["M2", "M8b", "M8a"])]


# ---------------- checks ----------------

def report(name, n, diffs, tol, extra=""):
    # prints the rows compared and the largest differences; CHECK FAILED when one is above tol
    if n == 0:
        print(f"check ({name}): no rows to compare")
        return
    text = ", ".join(f"{c} {d:.3g}" for c, d in diffs.items())
    print(f"check ({name}): {n} rows compared, largest differences: {text} (tolerance {tol:g}) {extra}")
    failed = {c: d for c, d in diffs.items() if not d <= tol}
    if failed:
        print(f"CHECK FAILED ({name}): {failed}")


def risk_diff(a, b):
    # the risk is a float32 mean; results.csv keeps it with the shortest float32 representation in some rows, so the
    # two are compared at float32 precision
    return np.abs(np.asarray(a, dtype=np.float32).astype(np.float64) - np.asarray(b, dtype=np.float32))


def check_against_results(name, ours, ref, compare_l2):
    on = ["dataset", "regime", "corruption", "severity", "split"]
    m = ours.merge(ref, on=on, suffixes=("", "_ref"))
    diffs = {"lambda": np.abs(m["lambda"] - m.lambda_ref).max() if len(m) else 0,
             "risk": risk_diff(m.risk, m.risk_ref).max() if len(m) else 0,
             "savings": np.abs(m.savings - m.savings_ref).max() if len(m) else 0}
    extra = ""
    if compare_l2 and len(m):
        different = (m.l2.astype(str) != m.l2_ref.astype(str)).sum()
        extra = f"; chosen L2 different in {different} of {len(m)} rows"
        diffs["l2_rows_different"] = float(different)
    if any(diffs[c] > 0 for c in ["lambda", "risk", "savings"]):
        extra += "; NOT exactly zero"
    report(name, len(m), diffs, 1e-9, extra)
    if len(m) < len(ours):
        print(f"   ({len(ours) - len(m)} rows of ours have no counterpart in results.csv)")


def run_checks(bn, hd, results):
    print("\n=== checks ===")
    if len(hd):
        ms = results[results.score == "max_softmax"]
        check_against_results("a: m2 = M2 of results.csv", hd[hd.config == "m2"], ms[ms.method == "M2"], False)
        check_against_results("b: head_grid_n4000 = M8b (max softmax) of results.csv", hd[hd.config == HG],
                              ms[ms.method == "M8b"], True)
        cost = pd.read_csv(ADAPTATION_COST)
        ours = hd[(hd.config == HG) & (hd.regime == "joint") & (hd.split == 0) & (hd.severity == 3)]
        m = ours.merge(cost, on=["dataset", "regime", "corruption", "severity"], suffixes=("", "_ref"))
        diffs = {"evals_total (relative)": (np.abs(m.evals_total - m.evals_total_ref) / m.evals_total_ref).max()
                 if len(m) else 0,
                 "train_fwd (relative)": (np.abs(m.train_fwd - m.train_full_forwards) / m.train_full_forwards).max()
                 if len(m) else 0}
        report("d: head_grid_n4000 evaluations and training compute = adaptation_cost.csv", len(m), diffs, 0.01)
    else:
        print("checks (a), (b), (d): no output of job B")
    if len(bn) and len(hd):
        c = hd[hd.config == "c_bn_n4000"].merge(bn[(bn.config == "bn_n4000") & (bn.split == 0)], on=KEY,
                                                 suffixes=("", "_a"))
        report("c: c_bn_n4000 (job B, cache) = bn_n4000 (job A)", len(c),
               {col: np.abs(c[col] - c[col + "_a"]).max() if len(c) else 0 for col in ["savings", "risk"]}, 1e-6)
        e = hd[hd.config == "m2"].merge(bn[bn.config == "source"], on=KEY, suffixes=("", "_a"))
        report("e: source (job A) = m2 (job B)", len(e),
               {col: np.abs(e[col] - e[col + "_a"]).max() if len(e) else 0 for col in COMPARED}, 0.0)
    else:
        print("checks (c), (e): need the outputs of both jobs")


# ---------------- rows of the accounting ----------------

def extra_compute(rows):
    # Added by Fable (14 November) after the note of the executor on the free configurations. Two columns per row:
    #   extra_fwd     compute, in full forwards, that the stream formula of b2_common adds to the training:
    #                 *_free: the new heads (or logit maps) evaluated on the stored features of the 3,000 cal rows;
    #                 bn_traffic: the stored inputs run again through the blocks they reached (sum over the blocks
    #                 of rows_block_j x the cost of block j), because block j must see inputs normalized with the
    #                 new statistics of the blocks before it
    #   n_recal_rows  audits of a second calibration: 3,000 for bn_traffic (the backbone changed after M2 was
    #                 calibrated), 0 otherwise
    # A logit map adds nothing at inference: z' = A (W f + b) + a is one linear layer with weights A W.
    extra, recal = np.zeros(len(rows)), np.zeros(len(rows))
    for i, r in enumerate(rows.itertuples()):
        full, n_classes = RESNET20_FULL_MACS[r.dataset], NUM_CLASSES[r.dataset]
        if r.config.endswith("_free"):
            per_row = n_classes * (sum(FEATURE_DIMS) if r.config.startswith("head") else len(FEATURE_DIMS) * n_classes)
            extra[i] = N_CAL * per_row / full
        elif r.config == "bn_traffic":
            block_macs = np.diff(np.concatenate([[0.0], RESNET20_CIFAR_CUM_MMACS])) * 1e6
            extra[i] = sum(getattr(r, f"rows_block_{j + 1}") * block_macs[j] for j in range(len(block_macs))) / full
            recal[i] = N_CAL
    return extra, recal


def combine(bn, hd, curves_bn, curves_hd):
    # every configuration row of the two jobs; the baseline m2 comes from job B, or from the source rows of job A
    # for the cases job B has not done
    frames, curves = [], dict(curves_hd)
    if len(hd):
        frames.append(hd)
    if len(bn):
        have_m2 = set(map(tuple, hd[hd.config == "m2"][KEY].values)) if len(hd) else set()
        src = bn[bn.config == "source"]
        missing = src[[tuple(r) not in have_m2 for r in src[KEY].values]].copy()
        missing["config"] = "m2"
        frames += [bn[bn.config != "source"], missing]
        for key, c in curves_bn.items():
            if key[-1] == "source" and key[:-1] not in have_m2:
                curves[key[:-1] + ("m2",)] = c
            elif key[-1] != "source":
                curves[key] = c
    rows = pd.concat(frames, ignore_index=True)
    for col in ["train_fwd", "n_rows_paid"]:
        rows[col] = rows[col].fillna(0.0)
    rows["extra_fwd"], rows["n_recal_rows"] = extra_compute(rows)
    rows["family"] = rows.corruption.map(family_of)
    rows["what"] = rows.config.map(lambda c: "m2" if c == "m2" else c.split("_")[0])
    rows["row_set"] = rows.config.map(lambda c: c.split("_")[-1] if c != "m2" else "")
    base = rows[rows.config == "m2"][KEY + BASE_COLS].rename(columns={c: c + "_m2" for c in BASE_COLS})
    return rows.merge(base, on=KEY, how="inner"), curves


def acc_at_m2_compute(g, curves):
    # accuracy of the mean curve of the configuration at the mean cost of m2 at its certified point, minus the same
    # for the mean curve of m2 (same cases)
    keys = [tuple(r) for r in g[KEY].values]
    cfg = [curves.get(k + (g.config.iloc[0],)) for k in keys]
    m2 = [curves.get(k + ("m2",)) for k in keys]
    if any(c is None for c in cfg + m2):
        return np.nan
    x0 = (1 - g.savings_m2).mean()

    def at(cs):
        cost, acc = np.mean([c["cost"] for c in cs], axis=0), np.mean([c["acc"] for c in cs], axis=0)
        return np.interp(x0, cost[::-1], acc[::-1])  # cost falls with the lambda index
    return at(cfg) - at(m2)


def config_stats(g, curves):
    # one configuration on one group of cases
    config, c9 = g.config.iloc[0], c9_of(g.dataset.iloc[0])
    gain_case = g.savings - g.savings_m2
    s, s_m2, gain = g.savings.mean(), g.savings_m2.mean(), gain_case.mean()
    n, t = g.n_rows_paid.mean(), g.train_fwd.mean()
    # extra_fwd and n_recal_rows (extra_compute above): the audits of a second calibration count as paid rows, the
    # extra compute goes with the training in the paybacks and in the stream
    x, n_recal = g.extra_fwd.mean(), g.n_recal_rows.mean()
    rows_inc, rows_full = adaptation_rows_cost(n + n_recal, c9, s_m2)
    r = {"config": config, "n_cases": len(g), "savings": s, "savings_m2": s_m2, "gain": gain,
         "gain_sd_corruptions": gain_case.groupby(g.corruption).mean().std(), "share_gain_pos": (gain_case > 0).mean(),
         "risk": g.risk.mean(), "share_above_eps": (g.risk > EPS).mean(),
         "share_above_eps_vs_m2": (g.risk > EPS).mean() - (g.risk_m2 > EPS).mean(),
         "acc_system": g.acc_system.mean(), "acc_exit9": g.acc_exit9.mean(), "n_rows_paid": n, "T": t,
         "extra_fwd": x, "n_recal_rows": n_recal, "rows_inc": rows_inc, "rows_full": rows_full,
         "cost_inc": rows_inc + t + x, "payback_inc": payback(rows_inc, t + x, gain),
         "payback_full": payback(rows_full, t + x, gain)}
    for h in HORIZONS:
        net_m2 = net_savings(stream_cost("m2", h, c9, s_m2), h)
        r[f"net_{h}"] = net_savings(stream_cost(cost_kind(config), h, c9, s_m2, s, n, t, x), h)
        r[f"net_{h}_vs_m2"] = r[f"net_{h}"] - net_m2
    # equal audit budget: defined only for the cases with at most 2,000 paid rows. For the match configurations
    # that is a subset of the cases, so m2 and the training are taken on the same subset (fixed by Fable: before,
    # the savings of the subset were compared with the m2 of all the cases)
    eqb = g.dropna(subset=["savings_eqb"]) if "savings_eqb" in g else g.iloc[:0]
    r["n_cases_eqb"] = len(eqb)
    r["savings_eqb"] = eqb.savings_eqb.mean() if len(eqb) else np.nan
    r["gain_eqb"] = (eqb.savings_eqb - eqb.savings_m2).mean() if len(eqb) else np.nan
    for h in HORIZONS:
        if len(eqb):
            s_m2_e, t_e = eqb.savings_m2.mean(), eqb.train_fwd.mean()
            net_m2 = net_savings(stream_cost("m2", h, c9, s_m2_e), h)
            net = net_savings(stream_cost("eqb", h, c9, s_m2_e, r["savings_eqb"], 0, t_e), h)
        else:
            net_m2 = net = np.nan
        r[f"net_eqb_{h}"], r[f"net_eqb_{h}_vs_m2"] = net, net - net_m2
    r["acc_at_m2_compute"] = acc_at_m2_compute(g, curves)
    return r


def split_versions(rows):
    # split 0 for everything; splits 0 and 1 together for the configurations present on both
    both = set(rows[rows.split == 1].config)
    return {"0": rows[rows.split == 0], "0+1": rows[rows.split.isin([0, 1]) & rows.config.isin(both)]}


def config_table(rows, curves):
    tables = []
    for version, d in split_versions(rows).items():
        for grouping, extra in [("dataset_severity", []), ("regime", ["regime"]), ("family", ["family"])]:
            for keys, g in d.groupby(["dataset", "severity"] + extra):
                t = pd.DataFrame([config_stats(gc, curves) for _, gc in g.groupby("config", sort=False)])
                ref = t[t.config == HG]
                t["retained"] = t.gain / ref.gain.iloc[0] if len(ref) else np.nan
                t["compute_ratio"] = t["T"] / ref["T"].iloc[0] if len(ref) else np.nan
                t["cost_ratio"] = (t.cost_inc / ref.cost_inc.iloc[0]) if len(ref) else np.nan
                t.insert(0, "group", keys[2] if extra else "all")
                t.insert(0, "severity", keys[1])
                t.insert(0, "dataset", keys[0])
                t.insert(0, "grouping", grouping)
                t.insert(0, "splits", version)
                tables.append(t)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


# ---------------- hypotheses and the other tables ----------------

def print_main(table):
    cols = ["config", "n_cases", "savings", "gain", "gain_sd_corruptions", "share_gain_pos", "retained", "risk",
            "share_above_eps", "share_above_eps_vs_m2", "acc_system", "n_rows_paid", "T", "compute_ratio",
            "cost_inc", "payback_inc", "net_10000_vs_m2", "net_100000_vs_m2", "net_1000000_vs_m2", "gain_eqb",
            "net_eqb_10000_vs_m2", "net_eqb_100000_vs_m2", "acc_at_m2_compute"]
    for version in ["0", "0+1"]:
        main = table[(table.splits == version) & (table.grouping == "dataset_severity")]
        for (dataset, sev), g in main.groupby(["dataset", "severity"]):
            print(f"\n--- configurations, {dataset} severity {sev}, splits {version} (means over {g.n_cases.max()} "
                  f"cases at most) ---")
            print(g[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def passes_d1(r):
    return r.retained >= 0.70 and r.compute_ratio <= 0.10 and r.payback_inc < 30000


def hypothesis_d1(main):
    sev3 = main[main.severity == 3]
    datasets = sorted(sev3.dataset.unique())
    variants = sorted(c for c in sev3.config.unique() if c.split("_")[0] in ("head", "logit") and c != HG)
    for label, configs in [("D1 as written", D1_CONFIGS), ("D1 with the variants", variants)]:
        print(f"\n{label}: retained >= 0.70, compute_ratio <= 0.10, payback_inc < 30,000 at severity 3 on both "
              f"datasets ({', '.join(datasets)}), split 0")
        winners, missing = [], 0
        for c in configs:
            parts, ok = [], len(datasets) == 2
            for ds in datasets:
                r = sev3[(sev3.dataset == ds) & (sev3.config == c)]
                if not len(r):
                    parts.append(f"{ds} missing")
                    ok = False
                    missing += 1
                    continue
                r = r.iloc[0]
                ok &= bool(passes_d1(r))
                parts.append(f"{ds} retained {r.retained:.2f} compute_ratio {r.compute_ratio:.3f} payback_inc "
                             f"{r.payback_inc:,.0f} ({'pass' if passes_d1(r) else 'fail'})")
            if ok:
                winners.append(c)
            print(f"   {c}: " + "; ".join(parts))
        verdict = "HOLDS with " + ", ".join(winners) if winners else "does not hold"
        if not winners and (missing or len(datasets) < 2):
            verdict = f"not decided ({missing} (configuration, dataset) pairs missing, {len(datasets)} datasets present)"
        print(f"   -> {label}: {verdict}")


def hypothesis_kill(main):
    print("\nKill: a configuration with retained >= 0.5 and rows_inc + T <= 0.2 * (rows_inc + T of head_grid_n4000); "
          "and with T alone (split 0)")
    for (ds, sev), g in main.groupby(["dataset", "severity"]):
        ref = g[g.config == HG]
        if not len(ref):
            print(f"   {ds} severity {sev}: head_grid_n4000 missing")
            continue
        ref = ref.iloc[0]
        cand = g[(g.config != HG) & (g.config != "m2") & (g.retained >= 0.5)]
        total = cand[cand.cost_inc <= 0.2 * ref.cost_inc]
        alone = cand[cand["T"] <= 0.2 * ref["T"]]
        fmt = lambda d: ", ".join(f"{r.config} (retained {r.retained:.2f}, rows_inc + T {r.cost_inc:,.0f}, "  # noqa
                                  f"T {r.T:,.1f})" for r in d.itertuples()) or "none"
        print(f"   {ds} severity {sev} (head_grid_n4000: rows_inc {ref.rows_inc:,.0f} + T {ref['T']:,.0f}):"
              f"\n      rows_inc + T: {fmt(total)}\n      T alone: {fmt(alone)}")


def per_corruption_sd(diff, corruption):
    return diff.groupby(corruption).mean().std()


def hypothesis_d2(rows):
    print("\nD2: gain(c_head_grid_n4000) - gain(head_grid_n4000), split 0 (paired by case); at least 0.02?")
    a = rows[(rows.config == "c_head_grid_n4000") & (rows.split == 0)]
    b = rows[(rows.config == HG) & (rows.split == 0)]
    m = a.merge(b, on=KEY, suffixes=("", "_hg"))
    if not len(m):
        print("   no case with both configurations")
        return
    m = m[KEY].assign(diff=(m.savings - m.savings_hg).values)
    for (ds, sev), g in m.groupby(["dataset", "severity"]):
        d = g["diff"].mean()
        print(f"   {ds} severity {sev}: {len(g)} cases, difference {d:.4f} (sd between corruptions "
              f"{per_corruption_sd(g['diff'], g.corruption):.4f}) -> {'yes' if d >= 0.02 else 'no'}")


def hypothesis_d3(rows, heads, out_dir):
    print("\nD3: head_grid_n4000 against m2, split 0")
    hg = rows[(rows.config == HG) & (rows.split == 0)]
    if not len(hg):
        print("   head_grid_n4000 missing")
        return
    m2 = rows[(rows.config == "m2") & (rows.split == 0)].merge(hg[KEY], on=KEY)
    last = len(LAMBDAS) - 1
    h = heads[(heads.config == HG) & (heads.split == 0)].copy() if len(heads) else pd.DataFrame()
    head_tables = []
    for subset, sel in [("all", lambda d: d), ("noise", lambda d: d[d.corruption.map(family_of) == "noise"])]:
        for (ds, sev), g in sel(hg).groupby(["dataset", "severity"]):
            g2 = sel(m2)
            g2 = g2[(g2.dataset == ds) & (g2.severity == sev)]
            gain = (g.savings - g.savings_m2).mean()
            g05 = (g.savings_acc05 - g.savings_acc05_m2).mean()
            g02 = (g.savings_acc02 - g.savings_acc02_m2).mean()
            print(f"   {ds} severity {sev}, corruptions {subset} ({len(g)} cases): gain in savings {gain:.4f}; "
                  f"in savings_acc05 {g05:.4f} (ratio {g05 / gain:.2f}), in savings_acc02 {g02:.4f} (ratio "
                  f"{g02 / gain:.2f})")
            for name, d in [("m2", g2), (HG, g)]:
                print(f"      {name}: disagreements exit right {d.dis_exit_right.mean():.4f}, exit 9 right "
                      f"{d.dis_ref_right.mean():.4f}, both wrong {d.dis_both_wrong.mean():.4f}; share of cases with "
                      f"k_acc05 the last of the grid {(d.k_acc05 == last).mean():.2f}")
            if not len(h):
                continue
            hh = sel(h)
            hh = hh[(hh.dataset == ds) & (hh.severity == sev)].groupby("head")[
                ["acc_src", "acc", "agree_src", "agree", "new_agree", "new_agree_ref_wrong", "lost_agree"]].mean()
            hh["agree_gain_over_acc_gain"] = (hh.agree - hh.agree_src) / (hh.acc - hh.acc_src)
            hh["new_agree_ref_wrong_share"] = hh.new_agree_ref_wrong / hh.new_agree
            print(hh.to_string(float_format=lambda v: f"{v:.4f}"))
            head_tables.append(hh.reset_index().assign(dataset=ds, severity=sev, corruptions=subset))
    if head_tables:
        pd.concat(head_tables).to_csv(os.path.join(out_dir, "b2_table_d3_heads.csv"), index=False)


def free_rows_table(rows, out_dir):
    print("\nFree rows: gain with free, with match (as many paid rows), their ratio; rows of free; share above eps")
    out = []
    for version, d in split_versions(rows).items():
        for (ds, sev), g in d.groupby(["dataset", "severity"]):
            for param in ["head", "logit"]:
                for opt in ["grid", "k5", "k10", "k20"]:
                    f, mt = g[g.config == f"{param}_{opt}_free"], g[g.config == f"{param}_{opt}_match"]
                    if not len(f) or not len(mt):
                        continue
                    gf, gm = (f.savings - f.savings_m2).mean(), (mt.savings - mt.savings_m2).mean()
                    out.append({"splits": version, "dataset": ds, "severity": sev, "config": f"{param}_{opt}",
                                "n_cases": len(f), "gain_free": gf, "gain_match": gm, "ratio": gf / gm,
                                "rows_free": f.n_rows_set.mean(), "above_eps_free": (f.risk > EPS).mean(),
                                "above_eps_m2": (f.risk_m2 > EPS).mean()})
    t = pd.DataFrame(out)
    if len(t):
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        t.to_csv(os.path.join(out_dir, "b2_table_free.csv"), index=False)
    else:
        print("   no free / match rows")


def bn_table(rows, out_dir):
    print("\nBN: exit 9 accuracy gained, gain in savings, system accuracy (split 0)")
    d = rows[rows.config.isin(BN_CONFIGS) & (rows.split == 0)].copy()
    if not len(d):
        print("   no output of job A")
        return
    d["acc9_gain"] = d.acc_exit9 - d.acc_exit9_src
    d["gain"] = d.savings - d.savings_m2
    out = []
    for grouping, extra in [("dataset_severity", []), ("family", ["family"])]:
        t = d.groupby(["dataset", "severity"] + extra + ["config"])[["acc9_gain", "gain", "acc_system"]].mean()
        t["n_cases"] = d.groupby(["dataset", "severity"] + extra + ["config"]).size()
        t = t.reset_index().assign(grouping=grouping)
        t = t.sort_values(["dataset", "severity"] + extra + ["config"],
                          key=lambda s: s.map(BN_CONFIGS.index) if s.name == "config" else s)
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        out.append(t)
    pd.concat(out).to_csv(os.path.join(out_dir, "b2_table_bn.csv"), index=False)
    tr = d[d.config == "bn_traffic"]
    if len(tr):
        blocks = [f"rows_block_{j}" for j in range(1, 10)]
        print("   bn_traffic, mean rows per block:")
        print(tr.groupby(["dataset", "severity"])[blocks].mean().to_string(float_format=lambda v: f"{v:.0f}"))


def m8a_table(rows, results, out_dir):
    print("\nM8a (results.csv, agreement score, eps 0.05) minus M2 (max softmax), on the same cases; reading only")
    cases = rows[rows.config == "m2"][["dataset", "regime", "corruption", "severity", "split"]]
    on = ["dataset", "regime", "corruption", "severity", "split"]
    a = results[(results.method == "M8a") & (results.score == "agreement")].merge(cases, on=on)
    b = results[(results.method == "M2") & (results.score == "max_softmax")].merge(cases, on=on)
    m = a.merge(b, on=on, suffixes=("", "_m2"))
    if not len(m):
        print("   no case")
        return
    m["diff"] = m.savings - m.savings_m2
    t = m.groupby(["dataset", "severity"])["diff"].agg(["mean", "size"]).reset_index()
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    t.to_csv(os.path.join(out_dir, "b2_table_m8a.csv"), index=False)


# ---------------- figures ----------------

COLOURS = {"head": "tab:blue", "logit": "tab:orange", "bn": "tab:green", "c": "tab:red"}
MARKERS = {"n4000": "o", "n2000": "s", "n1000": "^", "n500": "v", "free": "*", "match": "P", "traffic": "X"}


def figure_gain(main, out_dir):
    sev3 = main[(main.severity == 3) & (main.config != "m2")]
    datasets = sorted(sev3.dataset.unique())
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        g = sev3[sev3.dataset == ds]
        x = g.cost_inc.where(g.cost_inc > 1, 1.0)
        for xi, r in zip(x, g.itertuples()):
            what, row_set = r.config.split("_")[0], r.config.split("_")[-1]
            ax.scatter(xi, r.gain, color=COLOURS.get(what, "gray"), marker=MARKERS.get(row_set, "."), s=40)
        xs = np.logspace(0, np.log10(max(x.max(), 10) * 2), 100)
        for h, style in [(10000, ":"), (30000, "--"), (100000, "-.")]:
            ax.plot(xs, xs / h, color="black", linestyle=style, linewidth=0.8, label=f"payback {h:,}")
        ax.set_xscale("log")
        ax.set_ylim(min(g.gain.min(), 0) - 0.005, max(g.gain.max(), 0.01) * 1.2)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("rows_inc + T (full forwards, zero cost at 1)")
        ax.set_ylabel("gain in savings over m2")
        ax.set_title(f"{ds}, severity 3, split 0")
        handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", label=w) for w, c in COLOURS.items()]
        handles += [plt.Line2D([], [], color="gray", marker=m, linestyle="", label=s) for s, m in MARKERS.items()]
        ax.legend(handles=handles + ax.get_legend_handles_labels()[0], fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "b2_gain_vs_compute.png"), dpi=150)
    plt.close(fig)


def figure_frontier(rows, curves, out_dir):
    d = rows[rows.split == 0]
    panels = sorted({(ds, sev) for ds, sev in zip(d.dataset, d.severity)})
    if not panels:
        return
    datasets, sevs = sorted({p[0] for p in panels}), sorted({p[1] for p in panels})
    fig, axes = plt.subplots(len(datasets), len(sevs), figsize=(6 * len(sevs), 4.5 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        for j, sev in enumerate(sevs):
            ax = axes[i][j]
            for config in FRONTIER:
                g = d[(d.dataset == ds) & (d.severity == sev) & (d.config == config)]
                cs = [curves.get(tuple(k) + (config,)) for k in g[KEY].values]
                cs = [c for c in cs if c is not None]
                if not cs:
                    continue
                cost, acc = np.mean([c["cost"] for c in cs], axis=0), np.mean([c["acc"] for c in cs], axis=0)
                line, = ax.plot(cost, acc, label=f"{config} ({len(cs)})", linewidth=1)
                ax.scatter(1 - g.savings.mean(), g.acc_system.mean(), color=line.get_color(), zorder=3)
            ax.set_xlabel("cost (fraction of the plain network)")
            ax.set_ylabel("system accuracy")
            ax.set_title(f"{ds}, severity {sev}, split 0")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "b2_frontier.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=PILOT)
    args = parser.parse_args()
    start_log("b2_summary", args.dir)
    bn, hd = read_folder(args.dir, "runs_bn"), read_folder(args.dir, "runs_heads")
    print(f"job A: {len(bn)} rows, job B: {len(hd)} rows in {args.dir}")
    if not len(bn) and not len(hd):
        print("nothing to summarize")
        return
    results = read_results()
    run_checks(bn, hd, results)
    rows, curves = combine(bn, hd, read_curves(args.dir, "curves_bn"), read_curves(args.dir, "curves_heads"))
    heads = pd.concat([read_folder(args.dir, "heads_bn"), read_folder(args.dir, "heads_heads")], ignore_index=True)
    table = config_table(rows, curves)
    table.to_csv(os.path.join(args.dir, "b2_table_configs.csv"), index=False)
    print("\n=== accounting (S_m2, gain and paybacks per dataset x severity; b2_table_configs.csv also by regime and "
          "by corruption family) ===")
    print_main(table)
    main_table = table[(table.splits == "0") & (table.grouping == "dataset_severity")]
    print("\n=== hypotheses ===")
    hypothesis_d1(main_table)
    hypothesis_kill(main_table)
    hypothesis_d2(rows)
    hypothesis_d3(rows, heads, args.dir)
    print("\n=== other tables ===")
    free_rows_table(rows, args.dir)
    bn_table(rows, args.dir)
    m8a_table(rows, results, args.dir)
    figure_gain(main_table, args.dir)
    figure_frontier(rows, curves, args.dir)
    print("\nsaved b2_table_*.csv, b2_gain_vs_compute.png, b2_frontier.png in", args.dir)


if __name__ == "__main__":
    main()

