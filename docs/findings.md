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
  - Its raw savings stay high but are illusory. Once the violating cases are discarded, little is left: 0.02 at
    severity 5 on CIFAR10 joint.
- **With λ calibrated again the guarantee holds and the savings still fall.** 8% to 11% of the cases are above ε, and
  at severity 3 only 59% (CIFAR10) and 53% (CIFAR100) of the clean savings are left. The share left grows with ε.
- **Where the savings go (H1).**
  - On CIFAR10 almost all the loss is at exits 1 to 3.
  - About 60% comes from the shift at the clean λ and 40% from the rise of λ.
  - joint_sd keeps the most savings, disjoint the least.
- **Exit scores (H3).**
  - The four scores give similar savings, within 0.04.
  - The agreement score helps only for joint_sd.
  - Learned scores on the head output rank inputs like the max softmax, and adding the pooled features did not
    help in a check on one network.
- **Re-fitting the score on the new domain loses savings (M8a).** A calibrated P(agreement) with one shared λ treats
  every exit alike, while an early exit is worth more. This points to cost-aware thresholds per exit.
- **Adapting the heads offline (M8b).**
  - It recovers a median 11% to 25% of the lost savings, more at high severity.
  - The week 4 criterion (1/4 of the drop on half of the cases) passes in 1 of 8 settings.
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
