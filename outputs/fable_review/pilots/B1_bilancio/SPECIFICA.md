# Pilot B1, the compute budget of the savings under shift: specification for the executor

Written by Fable (design and verification) on 14 November 2026, after Davide's go ahead. You are the executor: you
write the code, test it on a few cases and report. You do NOT launch the full job (Fable launches it) and you do NOT
interpret the results beyond what the checks below need. The predictions are already written in
`outputs/fable_review/quaderno.md` (entry B1.01): do not edit that file.

## 1. Rules

- Repository: `/Users/davide/Desktop/NN_ResearchProject`. Python: `/opt/homebrew/Caskroom/miniforge/base/envs/eesavings/bin/python`
  (below `$PY`). Run everything from the repository root.
- No git command that writes (no add, commit, stash, checkout, reset, branch, push, fetch).
- Write ONLY inside `outputs/fable_review/pilots/B1_bilancio/`. Everything else is read only for you: `src/`,
  `scripts/`, `tests/`, `docs/`, `third_party/`, `outputs/cifar/`, `outputs/savings_under_shift*/`,
  `outputs/block3_review/`, the other files of `outputs/fable_review/`.
- No agents, no web. CPU only, at most 3 heavy processes. Every run you launch must end within 5 minutes.
- Style of the repository: a comment at the top of every file (what it does, how to run it), short functions, plain
  English comments, reuse the functions of the package instead of rewriting them. Look at
  `scripts/6_block3_review/calibration_budget.py` for the style to follow.
- zsh on macOS: `sed` has no `\b`; a `grep -c` that returns 0 breaks an `&&` chain; pandas is version 3 (`astype(str)`
  keeps missing values as NaN).
- Keep notes as you go in `outputs/fable_review/pilots/B1_bilancio/NOTE_ESECUTORE.md`: what you did, every deviation
  from this specification with the reason, what you could not check.

## 2. What exists and must be reused

- `scripts/6_block3_review/review_common.py` (import it with `sys.path.insert(0, "scripts/6_block3_review")`):
  `all_runs()` (the 24 networks), `load_logits(dataset, regime, seed, corruption=None, severity=None)` → (10000, 9, C),
  `load_labels`, `fractions_of(dataset)` (cost of the 9 exits as a fraction of the plain ResNet20; the last one is
  above 1), `FAMILY_OF` (corruption → noise, blur, weather, digital), `start_log(name, out_dir)`.
- `eesavings.analysis`: `LAMBDAS` (201 values, descending, row 0 is lambda = 1), `consistency_losses(logits, conf)` →
  `losses (K, N)` and `exits (K, N)` (exits 1-indexed), `make_splits(5)` → list of `(adapt 4000, cal 3000, eval 3000)`
  index arrays, `ucb_index_fast(cal_losses, eps, delta)`.
- `eesavings.fys.crc_lam(cal_losses, eps)` (used as in `calibration_budget.py`), `eesavings.scores.exit_scores(logits,
  "max_softmax")` → (N, 9) with last column 1, `eesavings.data.cifar.CORRUPTIONS` (the 15 standard corruptions).
- The labels of CIFAR-C are the test labels in the same order: check it once per network against
  `outputs/cifar/<dataset>/<regime>/seed<k>/preds/cifar_c/labels.npy` (look at its shape first) and note the result.

## 3. Protocol

- Networks: the 24 of `all_runs()`. Domains: clean (`corruption "none"`, `severity 0`) and the 15 standard corruptions
  at severity 1, 3, 5: 46 domains per network. Splits: the 5 of `make_splits(5)`. Score: max softmax. `eps = 0.05`,
  `delta = 0.1`.
- For every split: `fit = adapt ∪ cal` (7,000 rows), `ev = eval` (3,000 rows). Whatever is chosen is chosen on `fit`
  (or on `cal`, for the calibration levels); EVERY level is measured on `ev`.
- Per case, with `conf = exit_scores(logits, "max_softmax")`, `preds = logits.argmax(2)`, `loss9 (N, 9)` = 1 where the
  exit disagrees with exit 9 (last column 0), `correct (N, 9)` = prediction equal to the label, `fr = fractions_of(dataset)`.
  For a set of rows and the exit used by each row: `savings = 1 - mean(fr[exit])`, `risk = mean(loss9[row, exit])`,
  `acc_system = mean(correct[row, exit])`, `acc_exit9 = mean(correct[row, 8])`.

## 4. The levels (column `level` of the output)

Shared threshold levels: one index `k` of the lambda grid, exits `exits[k]` from `consistency_losses`.

| level | how `k` is chosen |
|---|---|
| `ucb_3000`, `ucb_1000`, `ucb_250` | `ucb_index_fast(losses[:, cal[:n]], eps, delta)` with n = 3000, 1000, 250 |
| `crc_3000`, `crc_250`, `crc_100` | `crc_lam(losses[:, cal[:n]], eps)` |
| `oracle_shared` | among the `k` with `mean(losses[k, fit]) <= eps`, the one with the largest savings on `fit` (ties: the smallest `k`). `k = 0` is always feasible |
| `acc_abs` | among the `k` with `acc_exit9(fit) - acc_system(k, fit) <= 0.05`, the largest savings on `fit` |
| `acc_rel` | the same with the bound `0.05 * acc_exit9(fit)` |

Other levels:

- `oracle_exit_thresholds`: 8 thresholds `t`, an input leaves at the first exit `l` in 1..8 with `conf_l > t_l`, otherwise
  at exit 9. Thresholds from `fit_exit_thresholds` (section 5) on the `fit` rows, then measured on `ev`.
- `oracle_input`: every input leaves at the first exit that agrees with exit 9 (exit 9 if none before). Nothing to fit:
  computed on `ev`. Its risk is 0.
- `oracle_input_eps`: as `oracle_input`, but `floor(eps * 3000) = 150` rows of `ev` are moved to exit 1: the rows whose
  first agreeing exit is not exit 1, taken in decreasing order of `fr[first agreeing exit] - fr[0]` (stable sort). Each
  moved row disagrees, so `risk = moved / 3000`.

## 5. `fit_exit_thresholds(conf8, loss9, fr, eps)`: the best rule with 8 thresholds found

Goal: minimize the mean cost with mean loss `<= eps` on the given rows. It is a heuristic: the result is a lower
bound of the best rule of the class, and that is what is needed. Two invariants must hold on the fit rows and must be
asserted: mean loss `<= eps`, and savings not below those of the best shared threshold on the same rows.

- `policy_exits(conf8, t)`: 0-based index of the exit used (8 = exit 9), first `l` with `conf8[:, l] > t[l]`.
- Lagrangian `J = mean cost + mu * mean loss`. `coordinate_step(l, ...)`: with the other thresholds fixed, the exact
  best `t[l]`. The rows that reach exit `l` are those with `conf8[:, j] <= t[j]` for all `j < l`. For such a row,
  `g = (fr[l] + mu * loss9[row, l]) - (fr[d] + mu * loss9[row, d])`, where `d` is the exit the row would use from the
  exits after `l` (thresholds of `l+1..7`, else exit 9). Sort the rows by `conf8[:, l]` descending (the order of every
  exit is computed once per case), take the cumulative sum of `g` (0 for the rows that do not reach `l`), and choose
  the prefix with the smallest sum among the cuts between two different confidence values, the empty prefix included.
  Threshold: the midpoint between the last confidence kept and the first one left out; `1.0` for the empty prefix,
  `0.0` if every row leaves.
- `lagrangian_descent(..., t0, mu)`: sweeps `l = 7, 6, ..., 0` until no threshold changes, at most 10 sweeps.
- `fit_exit_thresholds`: start from the best shared threshold on these rows (all `t[l]` equal to that lambda). Go
  through `mu = np.geomspace(50, 0.2, 25)` from the largest to the smallest, each time starting from the previous
  solution. Then about 8 bisection steps in `log(mu)` between the smallest `mu` whose solution is feasible and the
  next one that is not. Return, among all the feasible solutions seen (the shared start included), the one with the
  lowest mean cost. If it is simple, add a last greedy pass that lowers one threshold at a time while the rule stays
  feasible and the cost goes down; if you add it, say so in the notes.
- Speed target: about 0.3 seconds per (case, split) or less. The full job has 24 × 46 × 5 = 5,520 of them.

## 6. Files to write (all in `outputs/fable_review/pilots/B1_bilancio/`)

1. `b1_levels.py`. Options: `--workers` (default 3), `--runs dataset/regime/seedK ...` (default all), `--severities`
   (default `0 1 3 5`), `--corruptions` (default all 15), `--out-dir` (default the pilot folder), `--force`. One file
   per network in `<out-dir>/runs/` written through a temporary file and `os.replace`; networks already done are
   skipped unless `--force`. At the end it concatenates into:
   - `b1_levels.csv`, one row per (network, domain, split, level): `dataset, regime, seed, corruption, severity, split,
     level, savings, risk, acc_system, acc_exit9, lam` (NaN where there is no shared lambda), `k`, `savings_fit, risk_fit`
     (on the rows where the level was chosen: `fit`, or the calibration rows; NaN for the two input oracles),
     `thresholds` (the 8 numbers separated by a space for `oracle_exit_thresholds`, empty otherwise), `n_moved`
     (for `oracle_input_eps`), `seconds` (time of `fit_exit_thresholds`, on its row).
   - `b1_frontier.csv`, one row per (network, domain, `k`), on ALL the 10,000 rows: `dataset, regime, seed, corruption,
     severity, k, lam, cost` (mean of `fr[exit]`), `risk, acc_system`.
   - `b1_exits.csv`, one row per (network, domain, exit 1..9), on all the 10,000 rows: `acc` of the exit, `agree` with
     exit 9, `oracle_pi` (share of inputs whose first agreeing exit is this one).
   Print one line per network when it is done, with the seconds it took.
2. `b1_summary.py` with `--dir` (default the pilot folder). It prints through `start_log("b1_summary", dir)` and
   writes the tables and the figure below. Sections:
   - **Checks.** (a) `ucb_3000` against `outputs/savings_under_shift/results.csv` (score `max_softmax`, eps 0.05 with
     `np.isclose`, method `clean` for severity 0 and `M2` for the corruptions, the 5 splits): `lambda`, `risk`,
     `savings` equal within 1e-9; print the number of rows compared and the largest difference. (b) `ucb_1000`,
     `ucb_250` and the three `crc_*` against `outputs/block3_review/calibration_budget.csv` (ucb: delta 0.1, eps 0.05;
     crc: eps 0.05), same columns and tolerance. (c) `oracle_shared - ucb_3000` by dataset and severity, printed next
     to `price_fst` of `outputs/block3_review/m2_validity_oracles.csv` (a different protocol: no tolerance, only side
     by side). (d) the invariants of section 5 from `savings_fit` and `risk_fit`; `oracle_input` risk equal to 0;
     `oracle_input_eps` risk at most 0.05. With a subset of networks or domains the checks run on the rows present.
     If (a), (b) or (d) fail, print `CHECK FAILED` with the detail and go on.
   - **Levels.** Mean savings by dataset × severity for every level (mean over networks, corruptions, splits), with the
     mean risk on eval, the share of cases with risk above 0.05, the mean `acc_system` and `acc_exit9`. The same
     savings table by dataset × regime × severity. Save `b1_table_levels.csv`.
   - **Margins.** By dataset × severity, and by dataset × regime × severity, and by dataset × corruption family at
     severity 3: `total = oracle_input_eps - ucb_3000`, `input_vs_exit = oracle_input_eps - oracle_exit_thresholds`,
     `exit_vs_shared = oracle_exit_thresholds - oracle_shared`, `price = oracle_shared - ucb_3000`, each also as a
     share of `total`; `oracle_input_eps - oracle_input`; `oracle_input - oracle_shared`. The same by seed, with the
     standard deviation between the 3 seeds. For `exit_vs_shared`: by setting (dataset × regime) at severity 0 and 3,
     the mean and how many of the 8 settings are at 0.02 or above; the same per seed. Save `b1_table_margins.csv`.
   - **Net savings of M2.** With `c9 = fractions_of(dataset)[-1]` and `S` the mean savings of a calibration
     configuration: `net = (n_cal * (1 - c9) + (H - n_cal) * S) / H` for `H` in 10,000, 100,000 and 1,000,000, by
     dataset and severity 0, 3, 5. Use `calibration_budget.csv` for the full grid of `n_cal` (100, 250, 500, 1000,
     2000, 3000; ucb with delta 0.1 and eps 0.05; crc with eps 0.05). Print the best `n_cal` for every `H`. Save
     `b1_net_savings.csv`.
   - **Accuracy lost.** For `acc_abs`, `acc_rel`, `ucb_3000` and `oracle_shared`: mean savings by dataset × severity,
     the share of the clean savings left at severity 3 and 5 (savings at that severity over savings at severity 0), the
     mean of `acc_exit9 - acc_system` and of the risk on eval; for `acc_abs` and `acc_rel` the share of cases where
     `k` is the last of the grid (everything leaves at exit 1), by dataset × severity.
   - **Savings against the accuracy of exit 9.** For every network, over its 45 shifted domains (mean over the splits):
     the Spearman correlation between the savings of `ucb_3000` and `acc_exit9`, and the mean absolute error of a
     linear fit `savings ~ acc_exit9` leaving one corruption out at a time. Print median, minimum and maximum over
     the 24 networks.
   - **Figure** `b1_levels.png` (matplotlib with the Agg backend, dpi 150): two panels, CIFAR10 and CIFAR100; x =
     severity 0, 1, 3, 5; y = savings; one line per level: `oracle_input_eps`, `oracle_input`,
     `oracle_exit_thresholds`, `oracle_shared`, `ucb_3000`, `ucb_250`, and dashed `acc_rel`, `acc_abs`. Legend, axis
     labels, no title needed.
3. `test_b1.py` (run with `$PY -m pytest outputs/fable_review/pilots/B1_bilancio/test_b1.py -q`): (a)
   `policy_exits` with all thresholds equal to a lambda gives the same exits as `eesavings.fys.compute_exits` on
   random confidences; (b) `coordinate_step` is exact: on small random problems its threshold reaches the minimum of
   `J` over every candidate threshold of that exit (brute force); (c) `fit_exit_thresholds` on random problems is
   feasible and not worse than the best shared threshold; (d) `oracle_input_eps` on a hand made example.
4. `NOTE_ESECUTORE.md`.

## 7. Trial run and what to report

- Trial: `$PY outputs/fable_review/pilots/B1_bilancio/b1_levels.py --runs cifar10/joint/seed0 cifar100/joint_sd/seed1
  --corruptions gaussian_noise frost jpeg_compression --workers 2 --out-dir outputs/fable_review/pilots/B1_bilancio/smoke`
  and then `b1_summary.py --dir outputs/fable_review/pilots/B1_bilancio/smoke`. Both must run to the end, the
  checks (a), (b), (d) must pass on the rows present, the tests must pass.
- Leave the default output folder empty: the full job is `caffeinate -i $PY
  outputs/fable_review/pilots/B1_bilancio/b1_levels.py --workers 3` and Fable launches it.
- Final message, at most 300 words, no code: the files written; the outcome of tests and checks on the trial; the
  seconds per (case, split) and the projected time of the full job with 3 workers; every deviation from this
  specification; what you could not check.
