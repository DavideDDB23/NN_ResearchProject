# Pilot B2, an adaptation phase that pays back its compute: specification for the executor

Written by Fable (design and verification) on 14 November 2026, after Davide's go ahead. You are the executor: you
write the code, test it on a few cases and report. You do NOT launch the full jobs (Fable launches them) and you do
NOT interpret the results. The predictions are already written in `outputs/fable_review/quaderno.md` (entry B2.01):
do not edit that file.

## 1. Rules

- Repository: `/Users/davide/Desktop/NN_ResearchProject`. Python: `/opt/homebrew/Caskroom/miniforge/base/envs/eesavings/bin/python`
  (below `$PY`). Run everything from the repository root.
- No git command that writes (no add, commit, stash, checkout, reset, branch, push, fetch).
- Write ONLY inside `outputs/fable_review/pilots/B2_adattamento/`. Everything else is read only for you: `src/`,
  `scripts/`, `tests/`, `docs/`, `third_party/`, `data/`, `outputs/cifar/`, `outputs/savings_under_shift*/`,
  `outputs/block3_review/`, the other files of `outputs/fable_review/`.
- No agents, no web. At most 2 heavy processes while you test, and at most ONE process using MPS. Every run you
  launch must end within 5 minutes. Check `pmset -g batt` before a run and do not start one below 40%.
- Style of the repository: a comment at the top of every file (what it does, how to run it), short functions, plain
  English comments, reuse the functions of the package instead of rewriting them
  (`scripts/5_savings_under_shift/savings_under_shift.py` and `scripts/6_block3_review/adaptation_cost.py` are the
  style and the logic to follow).
- zsh on macOS: `sed` has no `\b`; a `grep -c` that returns 0 breaks an `&&` chain; pandas is version 3 (`astype(str)`
  keeps missing values as NaN). Multiprocessing uses spawn: keep the `if __name__ == "__main__":` guard.
- Keep notes as you go in `outputs/fable_review/pilots/B2_adattamento/NOTE_ESECUTORE.md`: what you did, every
  deviation from this specification with the reason, what you could not check.

## 2. What exists and must be reused

- Networks: `outputs/cifar/<dataset>/<regime>/seed0/` with `model.pt` and `preds/` (`test_labels.npy`,
  `cifar_c/<corruption>_logits.npy` of shape (5, 10000, 9, C), `cifar_c/<corruption>_features.npy` of shape
  (5, 10000, 4352) float16: the features of the pooled heads of exits 1..8, one block after the other, sizes
  `eesavings.models.FEATURE_DIMS`). Index 0 of the first axis is severity 1. Use `mmap_mode="r"`.
- CIFAR-C images: `eesavings.data.cifar.load_cifar_c(dataset, corruption, data_dir)` → (5, 10000, 32, 32, 3) uint8 and
  the labels (`data_dir` is `data`); `to_tensor_batch`, `CORRUPTIONS`, `NUM_CLASSES`.
- `eesavings.adaptation`: `L2_GRID`, `MAX_ITER`, `adapt_heads`, `fit_head`, `head_logits`, `head_weights`,
  `distillation_loss`, `split_by_exit`. This is M8b: read it before anything else.
- `eesavings.analysis`: `LAMBDAS`, `consistency_losses`, `make_splits`, `ucb_index_fast`, `measure`.
  `eesavings.scores.exit_scores`. `eesavings.savings`: `RESNET20_EXIT_MACS`, `RESNET20_FULL_MACS`, `cost_fractions`.
  `eesavings.models.MultiExitResNet20` (read its `forward` and `BasicBlock.forward`), `eesavings.train.load_model`,
  `eesavings.export.predict`.
- `scripts/6_block3_review/review_common.py` (import with `sys.path.insert(0, "scripts/6_block3_review")`):
  `load_logits`, `load_labels`, `fractions_of`, `run_dir`, `FAMILY_OF`, `start_log`.
- `scripts/6_block3_review/adaptation_cost.py`: the convention for the compute of the training (count the closure
  evaluations of L-BFGS; one evaluation of a linear head on N rows costs `2 * N * D * C` MACs; a "full forward" is
  `RESNET20_FULL_MACS[dataset]` MACs; the held-out loss that chooses L2 is not counted in the training but reported
  apart).
- `outputs/fable_review/pilots/00_controlli_preliminari/probe_bn_and_patience.py`: an earlier quick BN adaptation,
  only as a reference.
- `outputs/fable_review/esecuzione/run_m8b_resume.sh`: the pattern for a job with a status file and a battery
  watchdog.
- `outputs/savings_under_shift/results.csv`: rows `M2`, `M8b`, `M8a` to check against (columns `dataset, regime, seed,
  score, eps, method, corruption, severity, split, lambda, risk, savings, valid_savings, pi_1..pi_9, l2`).

## 3. Protocol

- Networks: seed 0 of `cifar10/joint`, `cifar10/joint_sd`, `cifar100/joint`, `cifar100/joint_sd`.
- Domains: the 15 standard corruptions at severity 3 and 5. Splits: 0 and 1 of `make_splits(5)` (`adapt` 4,000 rows,
  `cal` 3,000, `eval` 3,000). A "case" is (network, corruption, severity, split): 240 cases.
- Score max softmax, `eps = 0.05`, `delta = 0.1`. Lambda with `ucb_index_fast` on the `cal` rows, everything measured
  on the `eval` rows. Nothing may ever be fitted on `cal` or `eval` rows. Labels are used only to measure.
- `fr = fractions_of(dataset)`; `c9 = fr[8]` (above 1).

## 4. `b2_common.py`: the measurement shared by the two jobs

`measure_config(new, src, labels, fr, n_rows_paid)`, where `new` and `src` are the logits (6000, 9, C) of the
configuration and of the source network on the rows `np.concatenate([cal, ev])` (the first 3,000 are `cal`), `labels`
the labels of the same rows. Everything below uses the exit 9 of `new` as the reference (for the BN configurations it
is the adapted exit 9). It returns:

1. **Certificate.** `losses, exits = consistency_losses(new, exit_scores(new, "max_softmax"))`,
   `k = ucb_index_fast(losses[:, :3000], eps, delta)`; on the eval rows `lam, k, risk, savings, pi_1..pi_9` (as
   `analysis.measure`).
2. **Equal audit budget.** If `0 < n_rows_paid <= 2000`: `k_eqb = ucb_index_fast(losses[:, :3000 - n_rows_paid], ...)`
   and `savings_eqb, risk_eqb, n_cal_eqb`. Otherwise NaN.
3. **Accuracy.** `acc_system` (eval, at `k`), `acc_exit9` (exit 9 of `new`), `acc_exit9_src` (exit 9 of `src`), and the
   three shares of the eval rows that make the risk at `k`: `dis_exit_right` (the exit used is right, exit 9 wrong),
   `dis_ref_right` (exit 9 right, exit used wrong), `dis_both_wrong`.
4. **Certificate on the accuracy lost** (labels of the `cal` rows used, as a reading). Signed loss per (lambda, row):
   `g = 1[exit used wrong] - 1[exit 9 wrong]`, rescaled `x = (g + 1) / 2`. For `eps_acc` in 0.02 and 0.05:
   `k_acc = ucb_index_fast(x[:, :3000], (eps_acc + 1) / 2, delta)`; on eval `savings_acc02`, `acc_loss_acc02` (=
   `acc_exit9 - acc_system` at that `k`), `k_acc02`, and the same for 05.
5. **Heads** (a list of 8 records, on the eval rows): `acc` and `agree` (with exit 9 of `new`) of the head in `new`;
   `acc_src` and `agree_src` (head and exit 9 of `src`); `new_agree` = share of rows where the head agrees in `new` and
   did not in `src`; `new_agree_ref_wrong` = share of rows with a new agreement where exit 9 of `new` is wrong;
   `lost_agree` = share of rows that agreed in `src` and do not in `new`.
6. **Curves** (float32 arrays of length 201, on the eval rows, one value per lambda): `cost` (mean of `fr[exit]`),
   `acc` (system accuracy), `risk`.

Also here: `row_sets(adapt, exit_src_adapt)` and the accounting helpers of section 7, so that they can be tested.

## 5. Job A, `b2_bn.py`: BatchNorm statistics estimated on the new domain (MPS, one process)

For every network, corruption (load the images once), severity 3 and 5, and split in `--splits`:

- `src` = the saved logits. `k_m2` = the M2 choice on the `cal` rows of the source logits; `exit_src` = the exit (1..9)
  of every `adapt` row at that lambda.
- Configurations:
  - `bn_n500`, `bn_n1000`, `bn_n2000`, `bn_n4000`: the statistics of every BatchNorm layer estimated on the images
    `adapt[:n]`, in ONE batch. `n_rows_paid = n`.
  - `bn_traffic`: the statistics of the stem (`bn1`) from all the 4,000 `adapt` rows; those of the two BatchNorm layers
    of block `j` (1..9) only from the `adapt` rows with `exit_src >= j` (the inputs that reach block `j` under M2).
    `n_rows_paid = 0`. Save the number of rows used for every block. A block with fewer than 64 rows keeps the source
    statistics (log it).
- One function does both: `bn_statistics(model, x, block_masks)`. It follows `MultiExitResNet20.forward` by hand
  without the heads; at every BatchNorm it computes mean and variance over the selected rows (dims 0, 2, 3), keeps the
  unbiased variance for the running statistics, and normalizes ALL the rows of the batch with that mean and the biased
  variance (what PyTorch does in training mode), so the next layer sees inputs normalized with the new statistics. The
  adapted model is a deep copy with `running_mean` and `running_var` replaced.
- With the adapted model in eval mode, `predict` on the rows `concatenate([cal, ev])` → `measure_config`. For
  `bn_n4000` on split 0 run instead all the 10,000 rows with `return_features=True` and save the cache for job B:
  `cache_bn/<dataset>_<regime>_seed0_<corruption>_sev<s>_split0_logits.npy` (float32, (10000, 9, C)) and
  `..._features.npy` (float16, (10000, 4352)), written through a temporary file and `os.replace`.
- Also one row `source` per case: `measure_config(src, src, ...)` from the saved logits, and, once per network, the
  largest absolute difference between the saved logits and those of the source model run now on MPS for the 6,000
  rows of the first case (to know how far MPS is from the export).
- Options: `--runs`, `--corruptions`, `--splits` (default `0 1`), `--out-dir`, `--force`, `--data-dir`. One output per
  (network, corruption): `runs_bn/<dataset>_<regime>_seed0_<corruption>.csv` (one row per case and configuration, the
  scalars of `measure_config` plus `dataset, regime, seed, corruption, severity, split, config, n_rows_paid, seconds`
  and `rows_block_1..9`), `heads_bn/...csv` (one row per case, configuration and head) and `curves_bn/...npz`.
  Atomic writes; a (network, corruption) already done is skipped unless `--force`; with `--splits 1` after a run with
  `--splits 0` the new rows are added to the file, not written over it. Print a line with the seconds for every
  (network, corruption).

## 6. Job B, `b2_heads.py`: heads and logit maps (CPU, `--workers 3`, one thread each)

Jobs are (network, corruption) pairs, `chunksize=1`, interleaving CIFAR10 and CIFAR100. In every worker
`torch.set_num_threads(1)`. For every severity 3 and 5 and split in `--splits`:

- **Row sets** (index arrays into the 10,000 rows, in the order of the split):
  `n4000 = adapt`, `n2000 = adapt[:2000]`, `n1000 = adapt[:1000]`, `n500 = adapt[:500]`;
  `free` = the `adapt` rows with `exit_src == 9`; `match = adapt[:len(free)]`.
  For a row set `R`: `fit = R[:(3 * len(R)) // 4]`, `held = R[(3 * len(R)) // 4:]`. For `n4000` this is exactly the
  3,000 + 1,000 of M8b.
- **What is adapted.** `head`: the weights and bias of the linear layer of each of the 8 heads, on its features (as
  `adaptation.fit_head`). `logit`: an affine map on the logits of each head, `z' = z @ A.T + a`, `A` (C × C) starting
  at the identity and `a` at zero, same distillation loss towards the softmax of exit 9, penalty `l2 * ||A - I||^2`,
  same L-BFGS with `line_search_fn="strong_wolfe"`.
- **How.** `grid`: for every head, every L2 of `L2_GRID` with `max_iter = MAX_ITER`, keep the L2 with the lowest
  distillation loss on `held` (first best, as `adapt_heads`). It uses `fit` and `held`: `n_rows_paid = len(R)`.
  `k5`, `k10`, `k20`: one L2 = 0.01 and `max_iter` = 5, 10, 20, on `fit` only: `n_rows_paid = len(fit)`.
  For the row set `free`, `n_rows_paid = 0` in every case.
- **Names.** `<head|logit>_<grid|k5|k10|k20>_<n4000|n2000|n1000|n500|free|match>`: 48 configurations.
  Split 0: all of them. Split 1: only `head_grid_n4000` and the 36 with `k5`, `k10`, `k20`.
  A row set with fewer than 200 rows is skipped (log it).
- **Counting.** Write `fit_head_counted` and `fit_affine_counted`, which also return the number of closure
  evaluations. `fit_head_counted` must do exactly the operations of `adaptation.fit_head`, so the weights are
  identical. For every configuration: `evals_total`; `train_macs` = sum over heads and L2 values of
  `evals * 2 * len(fit) * D_l * C` (head) or `evals * 2 * len(fit) * C * C` (logit); `train_fwd = train_macs /
  RESNET20_FULL_MACS[dataset]`; `heldout_fwd` apart (one forward on `held` for every head and L2: `len(held) * D_l * C`
  or `len(held) * C * C`); the chosen L2 values as in the column `l2` of `results.csv`; `seconds`.
- **Logits of a configuration.** `head`: `np.concatenate([head_logits(features, new_w, new_b), logits[:, -1:]], axis=1)`.
  `logit`: the mapped logits of exits 1..8 and the source exit 9. Then `measure_config` on `concatenate([cal, ev])`.
- **Baseline.** One row `m2` per case: `measure_config(src, src, ...)`, `n_rows_paid = 0`.
- **BN then heads** (split 0 only, and only if the cache of job A for the case exists; otherwise log and skip):
  features and logits from `cache_bn/`, teacher = softmax of the adapted exit 9, starting weights = the source heads.
  Configurations `c_head_grid_n4000`, `c_head_k10_n4000`, `c_head_k20_n4000`, `c_logit_grid_n4000`,
  `c_logit_k10_n4000`, all with `n_rows_paid = 4000` (the BatchNorm statistics used the 4,000 `adapt` rows), plus a
  row `c_bn_n4000` measured from the cached logits (it must agree with the row `bn_n4000` of job A).
- Outputs per (network, corruption), with atomic writes and skip unless `--force`: `runs_heads/...csv`,
  `heads_heads/...csv`, `curves_heads/...npz`. Options as job A plus `--workers`.

## 7. `b2_summary.py`

Reads `runs_bn/`, `runs_heads/`, the heads and curves folders under `--dir` (default the pilot folder), prints through
`start_log("b2_summary", dir)`, writes the CSV tables and the figures named below. It must also run when only job B
or only a subset is there.

**Checks** (print the number of rows compared and the largest difference; on failure print `CHECK FAILED` with the
detail and go on):
- (a) `m2` against the rows `M2` of `results.csv` (seed 0, `max_softmax`, eps 0.05): `lambda`, `risk`, `savings` within 1e-9.
- (b) `head_grid_n4000` against the rows `M8b` with score `max_softmax`: the same three columns within 1e-9, and the
  chosen L2 values equal. If the difference is small but not zero, report its size.
- (c) `c_bn_n4000` (job B, from the cache) against `bn_n4000` (job A) on split 0: savings and risk within 1e-6.
- (d) `head_grid_n4000`, regime joint, split 0, severity 3, corruptions frost, gaussian_noise and jpeg_compression:
  `evals_total` and `train_fwd` against `outputs/block3_review/adaptation_cost.csv` (within 1%).
- (e) the source rows of job A against the rows `m2` of job B (equal).

**Accounting**, per dataset and severity, means over the cases (state in every table which splits it uses: split 0
for everything; splits 0 and 1 together, as a second version, for the configurations present on both):
- `S_m2` = mean savings of `m2`; `gain` = mean over the cases of (savings of the configuration − savings of `m2`).
- `rows_inc = n_rows_paid * (c9 - (1 - S_m2))`, `rows_full = n_rows_paid * c9`, `T` = mean `train_fwd`.
- `payback_inc = (rows_inc + T) / gain`, `payback_full = (rows_full + T) / gain`; infinite if the gain is not positive.
- Net savings over a stream of `H` inputs of the domain, `H` in 10,000, 100,000, 1,000,000, with `S` the mean savings
  of the configuration:
  - `m2`: `cost = 3000 * c9 + (H - 3000) * (1 - S_m2)`;
  - configurations with paid rows: `cost = (3000 + n_rows_paid) * c9 + T + (H - 3000 - n_rows_paid) * (1 - S)`;
  - `free` and `bn_traffic`: `cost = 3000 * c9 + 4000 * (1 - S_m2) + T + (H - 7000) * (1 - S)` (the first 4,000 inputs
    still run with M2 while the rows are collected);
  - equal audit budget: `cost = 3000 * c9 + T + (H - 3000) * (1 - S_eqb)`;
  `net = 1 - cost / H`, and the difference with the net of `m2`.

**Tables** (by dataset × severity; the main one is severity 3; also by regime and by corruption family):
- `b2_table_configs.csv`: for every configuration the number of cases, mean savings, `gain`, the standard deviation
  of the gain between corruptions, the share of cases with a positive gain, `retained = gain / gain of
  head_grid_n4000`, mean risk, share of cases above eps and its difference with `m2` on the same cases, mean
  `acc_system`, `acc_exit9`, mean `n_rows_paid`, `T`, `compute_ratio = T / T of head_grid_n4000`, `rows_inc`,
  `rows_full`, the two paybacks, the three nets and their difference with `m2`, `savings_eqb` with its gain and nets,
  and `acc_at_m2_compute`: the accuracy of the mean curve of the configuration (mean over the cases of `cost` and
  `acc` at every lambda index) interpolated at the mean cost of `m2` at its certified point, minus the same for `m2`.
- **Hypotheses**, printed with the numbers that decide them:
  - D1 as written: among the configurations with row set `n4000` and (`k5`, `k10`, `k20` or `logit`): `retained >=
    0.70`, `compute_ratio <= 0.10`, `payback_inc < 30000` at severity 3, on BOTH datasets. D1 with the variants: the
    same three conditions over every `head_*` and `logit_*` configuration except `head_grid_n4000`.
  - Kill: is there a configuration with `retained >= 0.5` and `rows_inc + T <= 0.2 * (rows_inc + T of
    head_grid_n4000)`? Also with `T` alone.
  - D2: `gain(c_head_grid_n4000) - gain(head_grid_n4000)` on split 0, by dataset × severity, with the standard
    deviation between corruptions; is it at least 0.02?
  - D3: by dataset × severity and for the noise family, for `head_grid_n4000`: per head `acc_src → acc`, `agree_src →
    agree`, the ratio of the two increases, `new_agree_ref_wrong / new_agree`; the gain of `savings_acc05` and of
    `savings_acc02` over `m2` next to the gain of `savings`, and their ratio; the three shares of the disagreements
    for `m2` and `head_grid_n4000`; the share of cases where `k_acc05` is the last of the grid.
- Free rows: for every `<param>_<opt>`: gain with `free`, with `match`, their ratio, the mean number of rows, the
  share of cases above eps for `free` and for `m2`.
- BN: for `bn_n500..bn_n4000` and `bn_traffic`: `acc_exit9 - acc_exit9_src`, the gain in savings, `acc_system`, by
  dataset × severity and by corruption family; the mean rows per block of `bn_traffic`.
- M8a, from `results.csv` (score `agreement`, eps 0.05, the same cases): its savings minus those of `M2` with
  `max_softmax`. Reading only.

**Figures** (matplotlib, Agg backend, dpi 150):
- `b2_gain_vs_compute.png`: one panel per dataset at severity 3; x = `rows_inc + T` (log scale; put the zero cost
  ones at x = 1), y = `gain`; one marker per configuration (colour by what is adapted: head, logit, bn, c; marker by
  row set); straight reference lines `gain = x / H` for paybacks of 10,000, 30,000 and 100,000 inputs.
- `b2_frontier.png`: 2 × 2 panels (dataset × severity); mean curves accuracy against compute (`cost` on x, `acc` on y)
  for `m2`, `head_grid_n4000`, `head_k10_n1000`, `bn_n4000`, `c_head_grid_n4000`, with the certified point of each
  marked (`1 - mean savings`, mean `acc_system`).

## 8. `test_b2.py` (run with `$PY -m pytest outputs/fable_review/pilots/B2_adattamento/test_b2.py -q`)

(a) `fit_head_counted` returns the same weights as `adaptation.fit_head` on random data and a positive count. (b) the
`grid` route for `head` on a small random problem returns the same weights and L2 values as `adaptation.adapt_heads`.
(c) `fit_affine_counted` lowers the distillation loss and with a huge L2 stays at the identity. (d) `bn_statistics`
with every mask true gives the running statistics of PyTorch (`reset_running_stats`, `momentum=None`, training mode,
one forward of the same batch) within a relative 1e-4, on CPU with a small random model. (e) `measure_config(src, src,
...)` gives the lambda, risk and savings of `analysis.measure` with the M2 choice. (f) `row_sets` and the accounting
helpers on numbers worked out by hand. (g) with a mask that selects half of the rows for one block, `bn_statistics`
gives for that block the statistics of those rows only.

## 9. Launch scripts, trial runs and what to report

- `run_b2_bn.sh [splits...]` and `run_b2_heads.sh [workers]`, on the pattern of `run_m8b_resume.sh`: `caffeinate -i`,
  the log appended to `b2_bn.log` or `b2_heads.log`, a `STATUS.txt` with START and END lines (exit code and battery),
  a watchdog that stops the job below 30% of battery, checked every 120 seconds. Do not run them.
- Trials, each under 5 minutes, outputs in `outputs/fable_review/pilots/B2_adattamento/smoke/`:
  1. `b2_bn.py --runs cifar10/joint/seed0 --corruptions frost --splits 0`
  2. `b2_heads.py --runs cifar10/joint/seed0 cifar100/joint/seed0 --corruptions frost --splits 0 --workers 2`
     (if the CIFAR100 part would pass 5 minutes, give it `--severities 3` or a similar option and say so)
  3. `b2_summary.py --dir .../smoke`: it must run to the end; checks (a), (b), (d) must pass on the rows present; (c)
     and (e) on the CIFAR10 case.
  4. the tests.
- Measure and report: images per second of job A (eval forward with and without features, and the statistics pass);
  seconds per case of job B by dataset, split by family of configurations (`head_grid`, `logit_grid`, the `k`
  configurations, the `c_` ones); from these the projected time of `b2_bn.py --splits 0`, of `--splits 1`, and of
  `b2_heads.py` with 3 workers on splits 0 and 1; the size on disk of the cache per case.
- Leave the default output folders empty. Final message, at most 350 words, no code: the files written; the outcome
  of tests and checks on the trial; the timings and projections above; every deviation from this specification; what
  you could not check.
