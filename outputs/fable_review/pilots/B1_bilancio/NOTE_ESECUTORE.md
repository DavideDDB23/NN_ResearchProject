# Pilot B1: notes of the executor

Written by the executor (Opus) on 14 November 2026, following SPECIFICA.md. Only files inside this folder were
written. No git command was used. The full job was not launched.

## What was done

- `b1_levels.py`: the 12 levels of section 4 for every (network, domain, split). The helper functions come from
  the package: `consistency_losses`, `ucb_index_fast`, `crc_lam`, `exit_scores`, `make_splits`, and `review_common`
  for loading, `fractions_of` and `all_runs`. The rule with 8 thresholds of section 5 is in `policy_exits`,
  `coordinate_step`, `lagrangian_descent`, `fit_exit_thresholds` and `greedy_lowering`. There is one file per
  network, `runs/<dataset>_<regime>_seed<k>.pkl`, a pickle with the three tables. It is written to `.tmp` and
  then moved with `os.replace`. At the end the script writes `b1_levels.csv`, `b1_frontier.csv` and
  `b1_exits.csv`.
- `b1_summary.py`: the sections of 6.2 (checks, levels, margins, net savings of M2, accuracy lost, exit 9, figure),
  log in `<dir>/b1_summary.log`.
- `test_b1.py`: tests (a)-(d) of 6.3, 30 test cases in all.
- The CIFAR-C labels: `preds/cifar_c/labels.npy` has shape (10000,), int64. For all 24 networks it is equal to
  `test_labels.npy`. I checked this once before writing the code, and `b1_levels.py` asserts it for every network.

## Implementation choices (not changes of definition)

1. **Risk in float64.** Every risk is the exact mean of the 0/1 loss, `count / n` in float64, with the same
   expression for every level. The reference CSVs store the risk as a float32 mean of the float32 losses. In
   float64 the gap reaches 6.7e-9, above the 1e-9 tolerance, although the choices are identical. So checks (a)
   and (b) compare the risk after rounding both sides to float32. Lambda and savings are compared in float64.
   The summary prints both risk gaps.
2. **Tolerance.** The comparisons with eps, and with the accuracy bounds of `acc_abs` and `acc_rel`, use a
   tolerance of 1e-12. It only absorbs rounding: the granularity of the risk is 1/7000.
3. **Best shared threshold.** In `fit_exit_thresholds` it is the best threshold of the LAMBDAS grid on the given
   rows, the same as `oracle_shared`. `b1_levels.py` passes the lambda of `oracle_shared`. Without that argument
   the function recomputes it (`best_shared_index`).
4. **Ties in `coordinate_step`.** On a tie it takes the first minimum: the fewest rows leaving.
5. **Bisection.** It starts every descent from the solution at the feasible end. There is no bisection when every
   mu of the path is feasible, or when none is. The case "none feasible" never happened; the shared start is then
   returned.
6. **Greedy pass: added.** For l = 7..0, the threshold is set to the cut with the lowest cost that keeps the mean
   loss <= eps, with the other thresholds fixed. This is an exact constrained coordinate step, and it can only
   lower a threshold. The pass repeats until no threshold changes, at most 10 sweeps. On 12 real (case, split)
   pairs it added 0.001-0.006 savings on fit beyond the Lagrangian path. It also brings the fit risk to exactly
   eps; the Lagrangian path alone ended at 0.040-0.049. A consequence I did not interpret: the eval risk of
   `oracle_exit_thresholds` is above eps in 60-100% of the trial cases (mean 0.054).
7. **Integer columns.** `k` and `n_moved` are integer columns with blanks; `read_csv` reads them as float.
   `thresholds` is written with 17 significant digits. Recomputing the eval savings and risk from that string
   gives the saved values exactly (checked on the 100 trial rows).
8. **Skipping networks.** A network already done is skipped without checking that its domains match the options.
   Another set of corruptions or severities needs `--force` or another `--out-dir`.
9. **Margins.** The shares of `total` are ratios of means. The deviation between seeds is the std (ddof 1) of the
   means of every seed; it is NaN in the trial (one seed per setting). Check (c) also shows `price_best` next to
   `price_fst`.
10. **Net savings.** The rows of `calibration_budget.csv` are restricted to the networks and domains present in
    `b1_levels.csv` (all of them in the full job). The best `n_cal` is printed for each method (ucb, crc) and
    each H.

## Deviation from the specification

- **Accuracy lost, "share of cases where k is the last of the grid (everything leaves at exit 1)".** The ties go
  to the smallest k. So when every fit row leaves at exit 1, k is a few steps before the last lambda: 195-197 in
  the trial, because several of the smallest lambdas send every row to exit 1. Read literally (k == 200), the
  share is 0 in every cell. The summary therefore prints two tables:
  - the share of cases where everything leaves at exit 1 on fit (`savings_fit = 1 - fr[0]`), the meaning given
    in the parenthesis;
  - next to it, the literal share with k == last.

## Tests and trial

- `python -m pytest outputs/fable_review/pilots/B1_bilancio/test_b1.py -q -p no:cacheprovider`: 30 passed, about
  3 s. I added `-p no:cacheprovider` so that nothing is written to `.pytest_cache` at the repository root. In this
  environment `PYTHONDONTWRITEBYTECODE=1`, so no `__pycache__` was written in `scripts/` or `src/`.
- Trial of section 7, into `smoke/`: 13 s in all, 10.2 s and 11.3 s per network (10 domains x 5 splits each).
  `fit_exit_thresholds` takes 0.16 s per (case, split) on average and 0.22 s at most. The whole pipeline takes
  about 0.21 s per (case, split).
- Checks on the trial:
  - (a) 100 rows, all differences 0.
  - (b) 100 rows for each of the 5 levels, all differences 0.
  - (d) 0 failures in 100 cases; 150 rows are always moved.
  - No `CHECK FAILED`.
- Extra cross-checks, one-off in the scratchpad, not in the scripts:
  - `b1_frontier.csv` against `m2_validity.csv` (`savings_all` and `risk_all` at its k): difference 0 on 2,000
    rows.
  - `oracle_exit_thresholds` recomputed from the saved thresholds: difference 0.
- Projection of the full job (5,520 (case, split), 3 workers): about 0.21 s x 5,520 / 3, so about 7 minutes. About
  50 s per network, 8 networks per worker. I would allow up to 10 minutes.

## What I could not check

- The full job, and so the tables over 24 networks: the std between seeds, and the 8 settings of
  `exit_vs_shared`.
- The blur family at severity 3: the trial has only three corruptions.
- How far the heuristic of section 5 is from the best rule of the class. It is only a lower bound. The
  invariants hold on every case seen.
- The "no feasible mu" branch, which never happened.

