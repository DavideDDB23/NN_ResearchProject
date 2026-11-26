# Pilot B2: notes of the executor (Opus), 14 November 2026

## What I did

- Read SPECIFICA.md and the code it names (adaptation.py, analysis.py, models.py, export.py, scores.py, savings.py,
  data/cifar.py, fys.py, savings_under_shift.py, adaptation_cost.py, review_common.py, run_m8b_resume.sh, the BN
  probe). Checked the columns of `outputs/savings_under_shift/results.csv` and `outputs/block3_review/adaptation_cost.csv`.
- Wrote in this folder: `b2_common.py`, `b2_bn.py` (job A), `b2_heads.py` (job B), `b2_summary.py`, `test_b2.py`,
  `run_b2_bn.sh`, `run_b2_heads.sh`, this file. Nothing written elsewhere (all runs with `PYTHONDONTWRITEBYTECODE=1`
  and pytest with `-p no:cacheprovider`, so no `__pycache__` / `.pytest_cache` outside this folder).
- Waited for b1_levels.py to end before every run. Trials only in `smoke/`; the default output folders are empty.

## How the code follows the specification

- `measure_config` (b2_common): the scalars of section 4 (k, lambda, risk, savings, valid_savings, pi_1..9; k_eqb,
  savings_eqb, risk_eqb, n_cal_eqb; acc_system, acc_exit9, acc_exit9_src, dis_exit_right, dis_ref_right,
  dis_both_wrong; k_acc02/05, savings_acc02/05, acc_loss_acc02/05), 8 head records, curves cost / acc / risk.
  `new_agree_ref_wrong` is a share of ALL eval rows (rows with a new agreement AND exit 9 of new wrong), so that
  D3's `new_agree_ref_wrong / new_agree` is the conditional share.
- `bn_statistics(model, x, block_masks)` (b2_bn): hand forward without the heads; `block_masks` is None or 9 boolean
  tensors (blocks 1..9), the stem always uses every row. Returns (adapted copy, rows per block, blocks that kept the
  source statistics). A block with fewer than 64 selected rows normalizes with, and keeps, the source statistics.
- `fit_head_counted` / `fit_affine_counted` (b2_heads): the affine map is exactly `fit_head` on the logits of the
  head with `w0 = I`, `b0 = 0` (so the penalty is `l2 * ||A - I||^2` and the bias is not penalized, as in M8b).
- head_grid_n4000 uses the same data path as M8b (features float16 -> `split_by_exit` -> `torch.as_tensor`, teacher
  `softmax(logits[fit, -1])`, logits of the configuration from all 10,000 rows), so it can reproduce M8b bit for bit.

## Deviations from the specification (and why)

1. Extra options `--severities` (default 3 5) in both jobs and `--cache-dir` (default `<out-dir>/cache_bn`): needed
   for the trials and for job B to find the cache of job A when `--out-dir` is not the default.
2. Checks (a) and (b) compare the risk at float32 precision (both cast to float32). The risk is a float32 mean, and
   the M2 rows of results.csv store it with the shortest float32 representation (e.g. `0.013666667`): read back as
   float64 it is up to half a float32 ulp (up to 1.9e-9 near 0.05) from the exact value, so a 1e-9 test on float64
   could fail with identical numbers. lambda and savings are compared in float64 within 1e-9 as asked.
3. The MPS check (largest |logit now - saved logit|) is done once per network per invocation of b2_bn.py, on the
   first case processed, and stored in the column `mps_max_abs_diff` of that source row (NaN elsewhere).
4. Job B takes the labels from `load_cifar_c` (as job A), not from `preds/test_labels.npy`.
5. Job A rows have `train_fwd = 0`: the statistics pass of bn_n* is a forward of the n paid rows and is counted
   through `n_rows_paid` only. Extra columns: `seconds_stats`, `seconds_predict`, `n_predicted`,
   `blocks_kept_source` (job A); `n_rows_set`, `n_fit`, `n_held`, `seconds_train`, `train_macs` (job B).
6. Job B rows for k5 / k10 / k20 write `l2` as eight times `0.01`.
7. `gain_sd_corruptions` is the sample standard deviation (ddof 1) of the per corruption mean gains; with one
   corruption it is NaN. Kill and D1 are evaluated per dataset x severity on the split 0 table.
8. Summary writes, besides `b2_table_configs.csv`, `b2_table_free.csv`, `b2_table_bn.csv`, `b2_table_d3_heads.csv`,
   `b2_table_m8a.csv` (the same numbers it prints); `b2_table_configs.csv` also has `cost_ratio` =
   (rows_inc + T) / (rows_inc + T of head_grid_n4000), used by the kill criterion.

## Comparisons that are not fair, or quantities that are not valid as defined

- **bn_traffic is not free, and its accounting misses a calibration.** (i) Its statistics are estimated in ONE pass
  in which block j sees inputs normalized with the NEW statistics of the earlier blocks. The M2 traffic that
  collected these rows ran with the SOURCE statistics, so these activations were never computed in deployment: one
  must store the 4,000 images and run them again (block j for the rows with exit_src >= j), about
  sum_j rows_block_j x (cost of block j): on the trial cases 0.70 (cifar10 sev 3) to 0.88 (cifar100 sev 3) of
  4,000 backbone forwards. With source-normalized
  inputs (what deployment really saw) the statistics would be different. (ii) bn_traffic needs M2 first (it uses
  k_m2), and after a BatchNorm change every activation changes, so the 3,000 cal rows must be run AGAIN through the
  adapted network to recalibrate: the "free" cost formula counts one calibration (3,000 c9) but bn_traffic pays two.
  The bn_n* configurations do not have this problem (adapt first, then calibrate once). The head `free`
  configurations are fine if the 3,000 cal features (all heads run there) are kept: recalibrating new heads only
  re-evaluates linear layers.
- **free and bn_traffic depend on the cal rows.** Their row set (or masks) comes from k_m2, chosen on the cal rows;
  the adapted configuration is then calibrated on the same cal rows. The UCB certificate assumes the configuration
  is fixed before the cal rows are seen; here it depends on them through one scalar. The effect is probably small,
  but it is not the "nothing fitted on cal" of section 3.
- The certificate on the accuracy lost uses the labels of the cal rows (section 4, item 4, "as a reading"): it is
  not a label free quantity.
- logit_* configurations add an inference cost that the savings do not count: an input leaving at exit l evaluates
  l affine maps of C x C MACs (at most 8 C^2: 0.2% of a forward for CIFAR100, 0.002% for CIFAR10). Small, but the
  head_* and bn_* configurations have no such extra cost.

## Trials (all in smoke/), each under 5 minutes, with b1_levels.py finished and the battery at 79-80%

1. `b2_bn.py --runs cifar10/joint/seed0 --corruptions frost --splits 0`: 2 cases in 12.6 s. MPS logits of the
   source model equal to the saved ones (largest difference 0).
   Extra (not in section 9, cheap, to time CIFAR100 and to give job B a CIFAR100 cache): `b2_bn.py --runs
   cifar100/joint/seed0 --corruptions frost --splits 0 --severities 3` (6.5 s, difference 0 again).
2. `b2_heads.py --runs cifar10/joint/seed0 cifar100/joint/seed0 --corruptions frost --splits 0 --workers 2`: 235 s
   wall (cifar10 69 s, cifar100 233 s); no `--severities` needed. c_* ran where the cache exists (cifar10 sev 3 and
   5, cifar100 sev 3); cifar100 sev 5 logged "no cache, c_* skipped".
   Extra: split 1 appended to the files (`b2_bn.py ... --splits 1 --severities 3`, then `--splits 0 1` which computed
   only the missing (5, 1) case; `b2_heads.py ... --splits 1 --severities 3 --workers 1`, 7.5 s): the merge by
   (severity, split) and the skip work.
3. `b2_summary.py --dir .../smoke`: runs to the end (3.5 s). All checks pass with differences exactly 0:
   (a) 5 rows, (b) 5 rows (lambda, risk, savings and the chosen L2 identical: head_grid_n4000 reproduces M8b bit for
   bit), (d) 2 rows (frost: 3,356 and 3,872 evaluations, as adaptation_cost.csv), (c) 3 rows, (e) 4 rows (every
   compared column). Also checked on copies in the scratchpad: the summary runs with only job A or only job B.
4. Tests: 7 passed (`-p no:cacheprovider`).

## Timings and projections (Apple M5, 16 GB)

Job A (MPS), images per second: statistics pass ~5,000-6,000 (bn_n*, one batch; bn_traffic ~2,900 because of the
masks); eval forward without features ~14,500-15,000, with features ~14,100-14,300; same for both datasets. About
6.3 s per case on split 0 (6 configurations + source + cache) and 5.3-5.9 s on split 1.
Projected `b2_bn.py --splits 0`: 120 cases x ~6.3 s = about 13 min; `--splits 1`: about 11-12 min.

Job B (CPU, one thread per worker), seconds per case (sum of the configurations, severity 3 / 5):

| family | cifar10 | cifar100 |
|---|---|---|
| head_grid (6 row sets) | 16.2 / 17.9 | 62.3 / 61.7 |
| logit_grid (6) | 5.9 / 7.1 | 34.3 / 34.3 |
| k5, k10, k20 (36) | 4.5 / 4.9 | 9.3 / 8.9 |
| c_* (6) | 6.1 / 5.8 | 21.5 / - |
| m2 | 0.06 | 0.08 |

Split 0 per case about 34 s (cifar10) and 127 s (cifar100); split 1 (head_grid_n4000 + 36 k configurations) about
8 s and 24 s. Projected `b2_heads.py --workers 3` on splits 0 and 1: (60 x 42 + 60 x 151) / 3 = about 65 min of
balanced work, so 70-80 min with the contention of 3 processes (split 0 alone about 55 min, split 1 alone about
11 min). Run job A on split 0 before job B, or the c_* configurations are skipped.

Cache of job A per case: cifar10 90.6 MB (features 87.0 + logits 3.6), cifar100 123.0 MB (87.0 + 36.0); the 120
cases of split 0 about 12.8 GB.

## What I could not check

- Checks (a), (b), (d) only on frost (and (d) only on frost, not on gaussian_noise and jpeg_compression); joint_sd
  networks and the other corruptions are not run. The bit-for-bit agreement of (b) depends on one thread per worker
  and on the same torch / Accelerate build as the original M8b run; with 3 workers it should hold, not verified.
- Job A and job B were not run at the same time, so the slowdown of the CPU part of job A (measure_config) while
  job B uses 3 cores is not measured.
- The launch scripts were only syntax checked (`zsh -n`), not run, as asked.

