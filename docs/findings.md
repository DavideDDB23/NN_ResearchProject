# Findings log

Everything we found while building the project, block by block, newest last. Some code and README sections were
removed when the repository was cleaned up (9 November 2026); their findings stay here, with the commit where the
code can still be found. The plan and the status for the professor are in the project document
"Compute savings degli early exit sotto shift" (https://claude.ai/artifact/MNreFiy1XTNWcYZcUC8axi).

## Block 1: Fast yet Safe (6 November 2026, commit `ba9360d`)

- **The port is exact.** On synthetic logits and on MSDNet ImageNet, our functions give the same losses, exits and
  chosen thresholds as the original code (`tests/test_upstream_equivalence.py`).
- **The 5-split figure matches the figure saved in their notebook.**
  - The plateaus of the consistency risk are the same: 0.37 MSDNet, 0.32 L2W-DEN, 0.18 DViT, 0.26 DynPerc.
  - The figures differ only in how they are drawn.
- **The guarantees hold with 100 splits.**
  - CRC: for the consistency risk, the mean test risk is at most ε + 0.002, which is noise.
  - CRC and the prediction-gt-gap loss: the risk is ε + 0.011 at ε = 0.01, 2.7 to 4.4 standard errors above ε. This
    is not noise. That loss can be negative and is not monotone in λ, so CRC does not strictly apply to it.
  - UCB: the violation rate is at most 0.11 for a single ε and at most 0.047 on average, compatible with δ = 0.1.
  - The WSR bound alone misses the true mean about 8% of the time, on Bernoulli samples with n = 100.
- **CRC saves more than UCB.** On ImageNet with the consistency risk, CRC saves 4 to 15 points more than UCB, which
  pays for its high probability guarantee.
- **Learn then Test is off by one in the original code.** It returns the first threshold that is not certified.
  Nothing used it, so we removed it on 9 November.

### The code of Pomponi et al. 2022 (checked before block 2, outside this repository)

- **It needs two fixes to start.** `utils.py` imports `models/vgg.py`, which is missing from their repository.
  `main.py` imports `adaptive_trainer`, which `base/trainer.py` does not define. With `training.device=mps` it then
  runs end to end on the Mac.
- **Epoch time** (batch 64, including the evaluation at the end of training):
  - joint, backbone and branches together: 43 s;
  - bernulli, branches with learned confidence: 358 s.

  With their 50 epochs this is about 36 minutes and 5 hours. Their final evaluation takes about 14 minutes, because
  it runs the network again for every threshold.
- **Why we did not use their pipeline.**
  - Each branch has a 128-channel convolution. One early head costs about 41% of the backbone, and the network with
    all heads costs 3.3 times a ResNet20, which would make the savings meaningless.
  - Their cost accounting is wrong. It is not cumulative, the stride is counted twice, and the heads are left out.
  - The forward computes the costs at every step. This doubles the time and updates BatchNorm twice.
  - In the bernulli method the "frozen" backbone stays in train mode, so its BatchNorm statistics change.
  - The dropout block cuts the gradient (`no_grad`), so the confidence learns only from its BCE term.
  - The relaxation temperature never goes below 5.
  - The `bernulli.yaml` configuration can collapse: the confidences go to 0 and the early heads stop learning.

  We kept only their backbone, the ResNet20 of He et al. 2016.

## Block 2: multi exit ResNet20 on CIFAR (6 and 7 November 2026, commits `de7fc3e`, `6f90613`)

- **Heads.**
  - The pooled heads of Kubaty et al. cost 0.12% of the network on CIFAR10 and 1.08% on CIFAR100.
  - Their 4x4 features are 4,352 numbers per image, not the 336 of global average pooling.
  - α is unconstrained: 46 of the 192 heads end with α < 0. Where seeds differ in the sign, the negative ones lose 2.6
    points on average.
- **Hardware.**
  - On the Mac (MPS) one epoch takes 18 s, of which 14 s is GPU time. Two trainings at the same time do not go
    faster, and mixed precision gives only 17%.
  - On a Colab A100 with 3 trainings at a time, an epoch takes about 4 s overall.
  - The 30 W charger of the Mac cannot sustain 8 CPU processes: the battery fell to 8% while charging. Use about 4.
- **The learning rate of the heads.** With lr 0.1 the loss of the heads phase seemed to diverge, but only on a
  512-image test. With a real backbone lr 0.1 works, and is better than 0.01: exit 1 reaches 0.56 against 0.50.
- **CIFAR-C.**
  - All 19 corruptions follow the order of the test set, at every severity.
  - Fog and contrast at severity 4 and 5 wash out the images: only 42% to 89% of them match their clean image by
    correlation, against 90% to 100% elsewhere.
  - The labels file is shared by all corruptions, so only the images can reveal a file in a different order.
- **Accuracy** (tables in the README):
  - The backbone reaches 91.3% on CIFAR10 and 65.9% on CIFAR100.
  - Joint and mixed are 12 to 16 points above disjoint at exit 1.
  - Multi exit training costs exit 9 between 0.3 and 3.9 points. In those regimes exit 9 is at most as accurate as
    exit 8.
  - Self distillation lowers exit 9 by about 2 points. Its training loss rises in the low lr stages.
  - On CIFAR-C, exit 9 loses 24 to 28 points. The early exits lose fewer points, and under strong noise exit 1 can
    beat exit 9.
- **Confidence units fitted on the features.** They are calibrated on average, but they separate correct and wrong
  predictions poorly: AUROC 0.58 to 0.74, against 0.74 to 0.91 for the max softmax. Adding the logits linearly cannot
  help, because the logits are a linear function of the same features.
- **The independent audits found these problems, all fixed:**
  - `.gitignore` hid `src/eesavings/data/`, so a fresh clone would not import.
  - A `model.train()` call would put the BatchNorm of a frozen backbone back in train mode.
  - Mixed replayed the batches of phase 1.
  - The CIFAR-C order test had too strict a threshold for fog and contrast.
  - Some tests were too weak to catch the bugs they targeted.

### First look at Fast yet Safe under shift (script `first_look.py`, removed on 9 November, commit `32b1b79`)

UCB, consistency risk, ε = 0.05, δ = 0.1, max softmax. λ is calibrated on 5,000 clean test images and kept fixed,
over 100 random splits. Risk and savings are measured on the other 5,000 images, clean and corrupted (15 standard
corruptions). Savings count the raw exits, so under shift they are partly illusory.

| dataset | regime | clean risk / savings | severity 1 | severity 3 | severity 5 | violations at severity 5 |
|---|---|---|---|---|---|---|
| CIFAR10 | joint | 0.044 / 0.580 | 0.063 / 0.502 | 0.082 / 0.425 | 0.122 / 0.378 | 0.93 |
| CIFAR10 | disjoint | 0.043 / 0.380 | 0.072 / 0.328 | 0.094 / 0.280 | 0.145 / 0.268 | 0.95 |
| CIFAR10 | mixed | 0.044 / 0.565 | 0.067 / 0.491 | 0.096 / 0.415 | 0.151 / 0.364 | 0.89 |
| CIFAR10 | joint_sd | 0.044 / 0.590 | 0.061 / 0.514 | 0.073 / 0.434 | 0.097 / 0.379 | 0.93 |
| CIFAR100 | joint | 0.043 / 0.248 | 0.046 / 0.187 | 0.051 / 0.139 | 0.062 / 0.118 | 0.56 |
| CIFAR100 | disjoint | 0.043 / 0.153 | 0.053 / 0.117 | 0.064 / 0.091 | 0.088 / 0.084 | 0.69 |
| CIFAR100 | mixed | 0.044 / 0.243 | 0.049 / 0.183 | 0.058 / 0.140 | 0.070 / 0.118 | 0.51 |
| CIFAR100 | joint_sd | 0.043 / 0.302 | 0.046 / 0.237 | 0.050 / 0.178 | 0.063 / 0.156 | 0.56 |

- **Clean data.** The risk is about 0.044, and 5% to 10% of the splits are above ε.
- **Under shift.** The risk goes above ε already at severity 1 on CIFAR10. This was the first sign of hypothesis H2,
  measured properly in block 3.

## Block 3: savings under shift at a fixed risk level (7 to 9 November 2026, commit `32b1b79`)

Full results in the README, section "Block 3".

- **Fast yet Safe with λ fixed breaks under shift (H2).**
  - On CIFAR10 the risk goes from 0.065 at severity 1 to 0.127 at severity 5, and 58% to 90% of the cases are above
    ε.
  - Its raw savings stay high (0.35 to 0.46 on CIFAR10), but they are bought with a risk above ε.
- **With λ calibrated again the guarantee holds and the savings still fall.** 8% to 11% of the cases are above ε, and
  at severity 3 only 59% (CIFAR10) and 53% (CIFAR100) of the clean savings are left. The share left grows with ε.
- **Where the savings go (H1).**
  - On CIFAR10 almost all the loss is at exits 1 to 3.
  - On CIFAR10, 61% to 71% of the drop comes from the shift at the clean λ, depending on which step is taken first
    (85% to 88% on CIFAR100). The rest comes from the rise of λ.
  - joint_sd keeps the most savings on CIFAR100 (+0.036 over joint, on all three seeds). On CIFAR10 the difference
    (+0.011) is inside the spread of the seeds. disjoint keeps the least.
- **Exit scores (H3).**
  - The four scores give similar savings, within 0.04.
  - The agreement score helps only for joint_sd.
  - Learned scores on the head output rank inputs like the max softmax, and adding the pooled features did not
    help in a check on one network.
- **Re-fitting the score on the new domain loses savings (M8a).** A calibrated P(agreement) with one shared λ treats
  every exit alike, while an early exit is worth more. This points to cost-aware thresholds per exit.
- **Adapting the heads offline (M8b).**
  - It recovers a median 11% to 25% of the lost savings on seed 0, more at high severity, before counting its compute.
  - The week 4 criterion (1/4 of the drop on half of the cases) is not met: one setting is just above the line
    (CIFAR10 joint_sd, 0.53 with a 95% interval from 0.36 to 0.71).
  - On a CIFAR100 noisy domain, the agreement of exit 1 with exit 9 goes from 0.14 to 0.59.
- **How to adapt the heads.**
  - Without an L2 pull towards the trained weights, the 1024 x 100 heads overfit 3,000 rows.
  - The score must be fitted on rows the heads did not see. Otherwise it is overconfident, and UCB lets no input exit.
  - With 5 L2 values the choice falls inside the grid for 97% of the heads. A first run with 3 values gave the same
    verdict.
- **The controller bound.** The upward jumps of the martingale are bounded by 1, not by 1/π. That is why Freedman's
  inequality gives a term in 1/√(πT) (`docs/controller_bound.md`).
- **The audit of this block found no error in the methods.** It found an unpaired comparison of M8b with M2, a
  narrow L2 grid, weak tests and imprecise sentences. All are fixed.

## Repository cleanup (9 November 2026)

- **Fast yet Safe in one module.** The port is now `src/eesavings/fys.py` instead of five files. Learn then Test was
  removed.
- **`first_look.py`.** Its savings part is superseded by block 3 and its numbers are kept above. Its summary part is
  now `scripts/4_export/summarize_networks.py`.
- **Scripts by phase.** They are in numbered folders, in the order to run them (README, "How the code is
  organized").
- **Removed dependencies.** hydra-core and omegaconf, used only by the code of Pomponi et al., and tqdm, pyyaml and
  ipykernel.
- **Unchanged results.** After the changes, the block 3 rows and the block 1 reproduction are identical to before.

## Preliminary checks before the next block (10 November 2026, not yet results)

Quick checks made while reviewing the plan for the next steps. They use seed 0 only, 2 to 4 networks, the max softmax
score, and a threshold chosen empirically on all rows, without UCB and without splits. They must be repeated with the
full protocol before being used. Scripts and outputs are in `outputs/fable_review/pilots/00_controlli_preliminari/`.

- **The consistency certificate costs no accuracy, and gets looser under shift.**
  - At the λ that keeps the consistency risk at 0.05, the accuracy lost with respect to exit 9 is about zero:
    +0.012 on clean CIFAR10 joint, 0.000 at severity 5, and slightly negative on CIFAR100 joint (−0.006 to −0.010).
  - With the accuracy lost held at 0.05 (this needs labels), the savings do not fall under shift: 0.70, 0.61 and 0.72
    on CIFAR10 joint at severity 0, 3 and 5, and 0.58, 0.70 and 0.81 on CIFAR100 joint. With the consistency risk they
    go from 0.58 to 0.26 and from 0.26 to 0.11.
  - Exit 9 is right in 58% of the disagreements on clean CIFAR10 and in 31% at severity 5. On CIFAR100 it is right
    in 28% and 13%. Most disagreements under shift are cases where both exits are wrong.
  - The 0.05 here is in absolute points, a weak requirement when exit 9 is at 22% to 47%. A relative version is to be
    checked.
- **The heads can still agree, the score cannot tell when.** An oracle that leaves at the first exit agreeing with
  exit 9 saves 0.78 on clean CIFAR10 joint and 0.65 at severity 5 (0.60 and 0.47 on CIFAR100), far above what the
  threshold reaches.
- **BatchNorm adaptation raises the accuracy but hardly the savings.** With the statistics re-estimated on 2,000
  unlabeled images, exit 9 goes from 0.47 to 0.64 on CIFAR10 joint at severity 5 and from 0.22 to 0.36 on CIFAR100.
  The savings at consistency risk 0.05 go from 0.260 to 0.292 and from 0.112 to 0.114. The head adaptation of block 3
  gave +0.08 and +0.04 there.
- **A patience rule does not help.** Asking also for the same prediction as the previous exit changes the savings by
  less than 0.001.
- **Per-exit thresholds proportional to the cost give no consistent gain** with the max softmax score: between −0.06
  and +0.04 against the shared λ, with a sign that depends on the regime.
- **The advantage of joint_sd on CIFAR10 rests on one seed.** The M2 savings at severity 3 of the three seeds are
  0.333, 0.354 and 0.399 for joint_sd, and 0.343, 0.358 and 0.351 for joint. On CIFAR100 the advantage is clear
  (0.157 to 0.173 against 0.128 to 0.132).

## Review of blocks 1 to 3 (11 to 13 November 2026, commit `a3eadcf`)

A check of everything above, made before the next block. The report is `outputs/fable_review/01_revisione.md`. The
scripts are in `scripts/6_block3_review/` and write to `outputs/block3_review/` (first versions in
`outputs/fable_review/parte1/`, same numbers except the cost of the adaptation, now a mean over three domains).

- **The code and the numbers hold.**
  - All 125 tests passed and the Fast yet Safe reproduction is identical. The review added 11 tests: 136 pass.
  - Nine rows of `results.csv`, recomputed with a measurement written from scratch, match to 3e-9.
  - Every number of the README and of this file that was checked matches the outputs. No leakage was found.
- **Exit 9 is a weak reference where the whole network is trained.**
  - In joint, mixed and joint_sd, exit 9 is at most as accurate as exit 8 also under shift: 0.5 to 1.5 points below
    on CIFAR100. On clean CIFAR100 an ensemble of exits 7 to 9 is 1.4 to 2.4 points above exit 9.
  - At the λ certified by M2 on CIFAR100, the early exit system is more accurate than exit 9 (equal for joint_sd on
    clean data). When the exit used and exit 9 disagree on clean data, the early exit is right in 46% to 50% of the
    cases and exit 9 in 28% to 31% in joint, mixed and disjoint (39% and 38% in joint_sd).
  - With the ensemble as the reference, the clean savings go from 0.231 to 0.293 on CIFAR100 and from 0.526 to 0.549
    on CIFAR10. About half of the gain on CIFAR100 is mechanical, because exits 7 and 8 vote in the reference: when
    only exits 1 to 6 may fire, the gain is +0.032 on CIFAR100 and +0.016 on CIFAR10.
- **The "valid savings" of M1 are a fragile metric.**
  - 12% to 43% of the M1 cases have a risk within two standard errors of ε, so their valid savings depend on the eval
    sample.
  - M1 and M2 are two points of the same curve of savings against risk. On CIFAR10, M1 keeps 0.35 to 0.46 of savings
    at a risk of 0.065 to 0.127.
- **M2 is valid, and it is an optimistic reference.**
  - With 100 splits per domain, its λ has a risk above ε in 8.9% and 7.8% of the calibrations on the 3,000 eval rows
    (CIFAR10 and CIFAR100), in 6.7% and 5.7% on the 7,000 rows outside the calibration, and in 3.4% and 2.8% on the
    whole test set.
  - An oracle of the threshold saves only 0.009 to 0.015 more.
  - M2 knows the domain and spends 3,000 full forwards on it. Over 10,000 inputs its net savings are about 30% lower.
  - With fewer calibration rows the savings fall: 0.526 with 3,000 rows and 0.403 with 100 on clean CIFAR10. On
    CIFAR100 the share left at severity 3 falls from 0.53 to 0.45.
  - CRC hardly depends on the number of rows (0.53 with 100), but it controls the risk only on average: 38% to 47% of
    its calibrations are above ε.
- **As it is, the adaptation of the heads does not pay for its compute.**
  - Training the heads costs as much as 2,600 full forwards on CIFAR10 and 30,900 on CIFAR100.
  - At severity 3 it pays back after 81,000 and 1.7 million inputs of the same domain.
  - On CIFAR100 most of the cost is the training, not the audits. On CIFAR10 it depends on how the audits are
    counted: the training is 2.2 times their extra cost over M2, and 0.65 times their 4,000 full forwards.
- **Two claims do not hold.**
  - joint_sd above joint on CIFAR10: +0.011 with a standard error of 0.020, from one seed (differences by seed:
    −0.010, −0.004, +0.048). The seeds also differ in the sign of α at exits 1 to 3, which may explain a part of it
    (correlation −0.54 inside a setting, on 18 points). On CIFAR100 the advantage is clear: +0.036 with a standard
    error of 0.005.
  - The only setting that passes the week 4 criterion (0.53) has a 95% interval from 0.36 to 0.71.
- **The split of the drop depends on the order.** The shift at the clean λ is 61% of the drop on CIFAR10 in the order
  of the README and 71% in the other order (85% and 88% on CIFAR100).
- **Intervals.** At severity 3 the share of the clean savings left is 0.59 (0.56 to 0.62) on CIFAR10 and 0.53 (0.50
  to 0.56) on CIFAR100. On CIFAR10 it goes from 0.41 for glass blur to 0.95 for brightness.
- **The controller bound is correct and loose.**
  - It holds against five adversaries, three of them adaptive, and it fails, as it must, when the adversary sees the
    audit coin.
  - With ε = 0.05 and π = 0.05 it is 0.156 after 10,000 inputs and 0.081 after 100,000.
- **Smaller points.**
  - The pooling of the heads is not counted in the MACs: with it the heads cost 0.52% of the network on CIFAR10, not
    0.12%.
  - A five times finer λ grid gives +0.002 to +0.003 of savings and the same ranking of the scores.
  - No latency has been measured yet.

## M8b on three seeds (14 November 2026, working tree after `a3eadcf`)

In block 3 the adaptation of the heads (M8b) was run on seed 0 only. Now it is run on seeds 1 and 2 too. The tables
are in the README, section "Week 4 test on the adaptation".

- **How.** `savings_under_shift.py --only-adaptation --recompute-features`, on a copy of the per-network files. The
  CIFAR-C features of seeds 1 and 2 are not saved, so they are computed again from `model.pt`. The result is
  `outputs/savings_under_shift_v2/results.csv`: the 230,040 rows of the first file, unchanged, plus 7,200 M8b rows
  (16 networks, 450 each). `outputs/savings_under_shift/` was not touched.
- **The gain carries over to the other seeds.**
  - At severity 3 to 5 with max softmax, M8b gains 0.047, 0.043 and 0.044 of savings over M2 on seeds 0, 1 and 2
    (mean over the 8 settings). The spread between the seeds is 0.003.
  - On CIFAR10 the gain is 0.065, 0.056 and 0.057. On CIFAR100 it is 0.029, 0.031 and 0.031.
  - With the agreement score it is lower: 0.029, 0.026 and 0.026.
- **The week 4 criterion is still not met.** CIFAR10 joint_sd is at the line on every seed: 0.53, 0.49 and 0.60 of
  the cases recover a quarter of the drop, 0.54 on the three seeds together. No other setting passes on any seed, and
  none with the agreement score.
- **M8b keeps the guarantee as M2 does.**
  - The adaptation rows fit the heads, so only the 3,000 eval rows can measure the risk of M8b.
  - On the same cases and the same rows, the risk of M8b is above ε in 10.2% of the cases and the risk of M2 in
    10.7%. The difference is −0.005, with a 95% interval from −0.022 to +0.013 (bootstrap over networks and
    corruptions).
  - Both are above ε by more than two standard errors in about 1% of the cases. The rates above 10% come from the
    noise of the eval rows, as the review measured for M2.
- **The L2 strength** is chosen inside the grid for 97% to 98% of the heads on every seed.
- **The compute is the one of the review above:** about 2,600 and 30,900 full forwards for the training, and a
  payback after about 81,000 and 1.7 million inputs.
- **Checks.**
  - The gains computed again in another way match to 1e-16, and the table of `savings_figures.py` matches an
    independent script.
  - Three predictions were written before the job ended, and all held: the gain of the new networks within 0.010
    of seed 0, no new setting passing the criterion, M8b not above ε more often than M2.
