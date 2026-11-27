# Phase A2 of the plan v3 (14 November 2026). The three tables of block 3 in README.md that were rewritten today,
# read again from the README and compared cell by cell with the CSV files written by savings_figures.py from
# outputs/savings_under_shift_v2/results.csv (in outputs/block3_review/figures_v2/):
#   1. "How much the savings fall": clean, M1 / M2 at severity 1, M1 / M2 / M8b at severity 3 and 5;
#   2. M1: mean risk / share of the cases above eps at severity 1, 3, 5;
#   3. the week 4 test on the three seeds.
# It only prints. Usage (from the repository root): python outputs/fable_review/fase_a/check_readme_tables.py

import pandas as pd

F = "outputs/block3_review/figures_v2"
NAMES = {"CIFAR10": "cifar10", "CIFAR100": "cifar100"}
lines = open("README.md", encoding="utf-8").read().split("\n")


def table_rows(header_start, must_contain=""):
    # the data rows of the first markdown table whose header starts with header_start
    for i, line in enumerate(lines):
        if line.startswith(header_start):
            rows = []
            for row in lines[i + 2:]:
                if not row.startswith("|"):
                    break
                rows.append([c.strip() for c in row.strip("|").split("|")])
            if all(must_contain in " ".join(r) for r in rows):
                return rows
    raise SystemExit("table not found: " + header_start)


def compare(name, written, expected):
    bad = [(k, written[k], expected[k]) for k in expected if written.get(k) != expected[k]]
    print(f"{name}: {len(expected)} cells, {len(bad)} different", bad if bad else "")


# 1. savings by method
s = pd.read_csv(f"{F}/savings_by_method.csv").set_index(["dataset", "regime", "key", "severity"]).value
written, expected = {}, {}
for dataset, regime, clean, sev1, sev3, sev5 in table_rows("| dataset | regime | clean | severity 1"):
    d = NAMES[dataset]
    written[dataset, regime] = (clean, sev1, sev3, sev5)
    expected[dataset, regime] = (
        f"{s[d, regime, 'M2', 0]:.2f}",
        " / ".join(f"{s[d, regime, k, 1]:.2f}" for k in ["M1", "M2"]),
        " / ".join(f"{s[d, regime, k, 3]:.2f}" for k in ["M1", "M2", "M8b"]),
        " / ".join(f"{s[d, regime, k, 5]:.2f}" for k in ["M1", "M2", "M8b"]))
compare("1. savings by method", written, expected)

# 2. risk of M1
m = pd.read_csv(f"{F}/m1_savings_and_risk.csv").set_index(["dataset", "regime", "severity"])
written, expected = {}, {}
for dataset, regime, sev1, sev3, sev5 in table_rows("| dataset | regime | severity 1 | severity 3 | severity 5 |", "%"):
    d = NAMES[dataset]
    written[dataset, regime] = (sev1, sev3, sev5)
    expected[dataset, regime] = tuple(
        f"{m.risk[d, regime, sev]:.3f} / {100 * m.above_eps[d, regime, sev]:.0f}%" for sev in [1, 3, 5])
compare("2. risk of M1", written, expected)

# 3. week 4 test
t = pd.read_csv(f"{F}/adaptation_test.csv", dtype={"seed": str})
t = t[t.score == "max_softmax"].set_index(["dataset", "regime", "seed"])
written, expected = {}, {}
for dataset, regime, drop, gain, median, shares, together, passes in table_rows("| dataset | regime | mean drop |"):
    d = NAMES[dataset]
    written[dataset, regime] = (drop, gain, median, shares, together, passes)
    expected[dataset, regime] = (
        f"{t['drop'][d, regime, 'all']:.3f}", f"{t.gain_M8b[d, regime, 'all']:.3f}",
        f"{t.median_recovery_M8b[d, regime, 'all']:.2f}",
        " / ".join(f"{t.share_recovering_a_quarter[d, regime, k]:.2f}" for k in "012"),
        f"{t.share_recovering_a_quarter[d, regime, 'all']:.2f}",
        " / ".join("yes" if t.passes[d, regime, k] else "no" for k in "012"))
compare("3. week 4 test", written, expected)
