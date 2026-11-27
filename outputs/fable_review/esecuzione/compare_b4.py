# Task B4: the CSV files written by scripts/6_block3_review/ against the ones of the pilots in
# outputs/fable_review/parte1/. Rows are matched on their keys; numbers must agree to 1e-9, the other columns exactly.
# For adaptation_cost, the frost domain against the numbers printed by the pilot m8b_cost.py.
# Usage (from the repository root): python outputs/fable_review/esecuzione/compare_b4.py

import numpy as np
import pandas as pd

NEW, OLD = "outputs/block3_review", "outputs/fable_review/parte1"
PAIRS = [
    ("m2_validity.csv", "m2_validity.csv", ["dataset", "regime", "seed", "corruption", "severity", "split"]),
    ("calibration_budget.csv", "sensitivity.csv",
     ["dataset", "regime", "seed", "corruption", "severity", "split", "method", "n_cal", "delta", "eps"]),
    ("decomposition_orders.csv", "decomposition_orders.csv",
     ["dataset", "regime", "seed", "corruption", "severity", "split"]),
    ("reference_accuracy.csv", "reference_accuracy.csv", ["dataset", "regime", "seed", "corruption", "severity"]),
    ("reference_disagreements.csv", "reference_disagreements.csv",
     ["dataset", "regime", "seed", "corruption", "severity", "split"]),
    ("reference_alternatives.csv", "reference_alternatives.csv",
     ["dataset", "regime", "seed", "corruption", "severity", "reference", "split"]),
]


def compare(new_name, old_name, key):
    new, old = pd.read_csv(f"{NEW}/{new_name}"), pd.read_csv(f"{OLD}/{old_name}")
    # fix of 14 November: with pandas 3 astype(str) keeps the missing values (n_cal, delta) as NaN floats, and
    # "|".join then fails; they are written as the text "nan" in both files, so the keys still match
    key_new = new[key].astype(str).fillna("nan").agg("|".join, axis=1)
    key_old = old[key].astype(str).fillna("nan").agg("|".join, axis=1)
    assert key_new.is_unique and key_old.is_unique, new_name
    new, old = new.set_index(key_new), old.set_index(key_old)
    print(f"{new_name} ({len(new)} rows) against parte1/{old_name} ({len(old)} rows): same keys "
          f"{set(new.index) == set(old.index)}; columns only in the new file {sorted(set(new.columns) - set(old.columns))}, "
          f"only in the old one {sorted(set(old.columns) - set(new.columns))}")
    old = old.loc[new.index]
    worst, worst_col, other = 0.0, None, []
    for col in [c for c in old.columns if c in new.columns and c not in key]:
        if pd.api.types.is_numeric_dtype(old[col]) and pd.api.types.is_numeric_dtype(new[col]):
            a, b = new[col].to_numpy(dtype=float), old[col].to_numpy(dtype=float)
            assert np.array_equal(np.isnan(a), np.isnan(b)), col
            d = np.nanmax(np.abs(a - b)) if len(a) else 0.0
            if d > worst:
                worst, worst_col = d, col
        elif not (new[col].astype(str) == old[col].astype(str)).all():
            other.append(col)
    print(f"   largest numeric difference {worst:.3g}{f' ({worst_col})' if worst_col else ''}; "
          f"{'OK' if worst <= 1e-9 and not other else 'DIFFERENT'} at 1e-9; non numeric columns that differ: {other}")


for pair in PAIRS:
    compare(*pair)

cost = pd.read_csv(f"{NEW}/adaptation_cost.csv")
pay = pd.read_csv(f"{NEW}/adaptation_cost_payback.csv")
print("\nadaptation_cost, frost severity 3 (pilot m8b_cost.log: cifar10 mean 84, min 34, max 105, 3356 evaluations, "
      "2,682 full forwards; cifar100 mean 97, min 64, max 107, 3872 evaluations, 31,215 full forwards):")
pilot = {("cifar10", 3): 83077, ("cifar10", 5): 40043, ("cifar100", 3): 1687912, ("cifar100", 5): 765941}
for _, c in cost[cost.corruption == "frost"].iterrows():
    print(f"   {c.dataset}: mean {c.evals_mean:.0f}, min {c.evals_min}, max {c.evals_max}, {c.evals_total} evaluations, "
          f"{c.macs:.3g} MACs, {c.train_full_forwards:,.0f} full forwards")
    for _, p in pay[pay.dataset == c.dataset].iterrows():
        frost_payback = (p.adapt_incremental + c.train_full_forwards) / p.gain
        print(f"      severity {p.severity}: payback with the frost training cost only {frost_payback:,.0f} "
              f"(pilot {pilot[c.dataset, p.severity]:,}); without training {p.payback_without_training:,.0f}")
