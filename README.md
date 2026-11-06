# Compute savings of early exit networks under distribution shift

Neural Networks project (Sapienza, prof. Scardapane).

We study how the compute savings of an early exit network change under distribution shift
when its label-free risk control (Fast yet Safe, Jazbec et al., NeurIPS 2024) must stay valid,
and how much of those savings an adaptation phase of the exit heads can recover, net of its cost.
The full plan is in the project document "Compute savings degli early exit sotto shift"
(https://claude.ai/artifact/MNreFiy1XTNWcYZcUC8axi).

Block 1 sets up the code and reproduces Fast yet Safe on its released ImageNet logits.
Block 2 trains a multi exit ResNet20 on CIFAR10 and CIFAR100 with four training regimes and saves
its outputs on the clean test set and on CIFAR-10-C / CIFAR-100-C, for the analysis of the savings under shift.

## Setup

```bash
conda env create -f environment.yml
conda activate eesavings
```

The reference repositories go in `third_party/` (not tracked by git):

```bash
# Fast yet Safe (Jazbec et al. 2024)
git clone https://github.com/metodj/RC-EENN.git third_party/RC-EENN
git -C third_party/RC-EENN checkout 3db9a52
# Pomponi et al. 2022, base for our multi exit networks
git clone https://github.com/jaryP/ConfidenceBranchNetwok.git third_party/ConfidenceBranchNetwok
git -C third_party/ConfidenceBranchNetwok checkout 405d273
```

## Code

- `src/eesavings/risk/`: exit rule, losses, bounds and threshold calibration, copied from Fast yet Safe with the same function names,
  plus our `rc_main_fast` (same results as `rc_main`, but faster), which the script uses.
- `src/eesavings/fys.py`: losses and exits for every threshold of the grid, from the logits of all exits.
- `src/eesavings/savings.py`: compute savings and their decomposition over the exits, and the cost of every exit of our ResNet20.
- `src/eesavings/models.py`: multi exit ResNet20 (9 exits), pooled heads with a confidence unit, MAC counting.
- `src/eesavings/data/cifar.py`: CIFAR10/100, CIFAR-C, train/val split.
- `src/eesavings/train.py`: losses, training phases and regimes.
- `src/eesavings/export.py`: outputs of a trained network on a set of images, fitting of the confidence units.
- `scripts/download_fys_logits.sh`: downloads the logits released with Fast yet Safe (ImageNet about 3.4 GB, CIFAR100 about 60 MB).
- `scripts/reproduce_fys.py`: reproduces their image classification experiment.
- `scripts/download_cifar_c.sh`, `train_multiexit.py`, `run_trainings.py`, `export_outputs.py`, `export_all.sh`,
  `first_look.py` and `notebooks/colab_training.ipynb`: block 2 (see below).
- `tests/`: unit tests, plus tests that compare our code with the original code of Fast yet Safe and of Pomponi et al.
  Tests that need data (CIFAR, CIFAR-C, exported outputs) are skipped when the data is missing: `pytest -rs` lists them.

## Reproducing Fast yet Safe

Run everything from the repository root.

```bash
bash scripts/download_fys_logits.sh imagenet
python scripts/reproduce_fys.py              # 5 splits, as in their notebook
python scripts/reproduce_fys.py --trials 100 # 100 splits, as in the paper
pytest
```

Results go to `outputs/fys/trials<N>/`: a CSV with test risk, violation rate, mean exit and
savings for every model, risk, method and eps, a figure, and `models.json` with the accuracy
of every exit and, for the consistency risk, how the inputs spread over the exits.

## Results of the reproduction

All numbers are on the ImageNet logits of Fast yet Safe (50,000 images, 100 for calibration, the rest for test).

**Same numbers as the original.** `tests/test_upstream_equivalence.py` checks that our functions give exactly
the same losses, exits and chosen thresholds as the original code (synthetic logits and MSDNet on ImageNet).
With 5 splits our figure matches the one saved in their notebook (`third_party/RC-EENN/img_cls/img_cls.ipynb`):
same plateaus of the test risk (for the consistency risk 0.37 MSDNet, 0.32 L2W-DEN, 0.18 DViT, 0.26 DynPerc) and same
mean exits. The figures differ only in how they are drawn: they draw UCB dashed and CRC solid (we do the
opposite), they plot the mean exit while we plot mean exit / L, their panels are in a different order
(consistency is the third one), and their eps axis stops at 0.4 with an extra point at eps = 0.

**The guarantees hold with 100 splits.**

- CRC (risk below eps in expectation): for the consistency risk the mean test risk is at most eps + 0.002,
  which is sampling noise. The largest excess, eps + 0.011, is for the prediction-gt-gap risk at eps = 0.01
  (all four models are slightly above eps there, by 2.7 to 4.4 standard errors). This is not noise: that loss
  can be negative and is not monotone in lambda, so the CRC guarantee does not strictly apply to it.
- UCB (risk below eps with probability 1 - delta = 0.9): the fraction of splits with test risk above eps is
  at most 0.11 at any eps and at most 0.047 on average over eps. With 100 splits one value has a noise of
  about 0.03, so 0.11 is compatible with 0.1.
- With only 5 splits (as in the notebook) the same check is too noisy: DViT with the consistency risk gets
  2 violations out of 5 at eps 0.05 to 0.07, which disappears with 100 splits. We also checked that the
  WSR bound alone, on Bernoulli samples with n = 100, misses the true mean about 8% of the time (below 10%).

**Consistency risk and savings** (100 splits). Test risk / savings with cost proportional to the exit
index, 1 - mean exit / L (the real FLOPs per exit are not in their files).

| model   | method | eps = 0.05  | eps = 0.1   | eps = 0.2   |
|---------|--------|-------------|-------------|-------------|
| MSDNet  | UCB    | 0.019 / 0.36 | 0.053 / 0.46 | 0.127 / 0.58 |
| MSDNet  | CRC    | 0.046 / 0.46 | 0.093 / 0.54 | 0.199 / 0.66 |
| DViT    | UCB    | 0.019 / 0.43 | 0.048 / 0.52 | 0.122 / 0.61 |
| DViT    | CRC    | 0.046 / 0.51 | 0.094 / 0.58 | 0.169 / 0.66 |
| L2W-DEN | UCB    | 0.021 / 0.40 | 0.051 / 0.51 | 0.125 / 0.63 |
| L2W-DEN | CRC    | 0.045 / 0.50 | 0.096 / 0.59 | 0.191 / 0.69 |
| DynPerc | UCB    | 0.020 / 0.42 | 0.049 / 0.57 | 0.128 / 0.67 |
| DynPerc | CRC    | 0.048 / 0.57 | 0.092 / 0.63 | 0.187 / 0.71 |

CRC uses the risk budget almost fully (test risk close to eps) and saves 4 to 15 points more than UCB,
which pays for its high probability guarantee. These are the in-distribution savings that the next part
of the project measures again under shift.

## Notes on the Pomponi et al. code (for the next part)

We ran one epoch of ResNet20 with its 10 branches on CIFAR10, on the Mac (Apple M5, MPS), outside this
repository. The original code needs two small fixes to start: `utils.py` imports `models/vgg.py`, which is
not in the repository, and `main.py` imports `adaptive_trainer`, which `base/trainer.py` does not define.
With `training.device=mps` it then runs end to end (training, evaluation, saved checkpoints).

Time of one training epoch (batch 64, including the evaluation at the end of training):

| method   | what it trains                              | time per epoch |
|----------|---------------------------------------------|----------------|
| joint    | backbone and branches together              | 43 s           |
| bernulli | branches with learned confidence (Pomponi)  | 358 s          |

With the 50 epochs of their configuration this is about 36 minutes for joint and about 5 hours for
bernulli on the Mac. The final evaluation of their script is also slow (about 14 minutes), because it
runs the network again for every threshold.

## Block 2: multi exit ResNet20 on CIFAR

### Model

- **Backbone.** The CIFAR ResNet20 of He et al. 2016 with option A shortcuts, copied from Pomponi et al. 2022
  (`models/resnet.py`, commit 405d273). A test checks that both give the same block outputs.
- **Exits.** Nine, one after each residual block.
  - Exits 1 to 8 use the "pooled head" of Kubaty et al. 2025 (SDN pooling): α·maxpool + (1 − α)·avgpool down to 4x4,
    with α learnable, then one linear layer. Each head also has a confidence unit, a linear output with a sigmoid
    on the same features.
  - Exit 9 is the usual global average pooling and linear layer, so the backbone alone is a standard ResNet20.
- **Why not the heads of Pomponi et al.** Their branches have a 128 channel convolution: one early head costs about
  41% of the backbone, and the network with all heads costs 3.3 times the ResNet20.
- **Costs** (`models.exit_costs`, counted with `FlopCounterMode`, convolutions and linear layers only).
  - An input that leaves at exit l pays blocks 1..l and heads 1..l. An input that reaches exit 9 also pays all heads.
  - Savings are measured against the plain ResNet20 (40.55 M MAC plus the final layer), so they can be negative.
  - The heads cost 0.12% of the network on CIFAR10 and 1.09% on CIFAR100.

### Training regimes

The regimes follow Kubaty et al. 2025, built from three phases:

| Phase | What trains | Loss |
|---|---|---|
| 1, backbone | backbone and exit 9 | cross entropy of exit 9 |
| 2, all | whole network | mean of the cross entropies of the 9 exits |
| 3, heads | exits 1 to 8 only; backbone frozen, BatchNorm statistics included | mean of the cross entropies of exits 1 to 8 |

- **joint** = phase 2.
- **disjoint** = phase 1, then phase 3.
- **mixed** = phase 1, then phase 2.
- **joint_sd** = phase 2 with self distillation from exit 9: for exits 1 to 8,
  (1 − α)·CE + α·τ²·KL(teacher ‖ exit), with α = 0.5, τ = 3, and the teacher detached.
- **Budget.** Every regime gets the same total budget, as in Kubaty's `scripts/regimes_resnet.py`: 160 epochs per phase,
  320 for joint and joint_sd. Phase 1 is shared by disjoint and mixed of the same seed.
- **Recipe of He et al. 2016 in every phase:**
  - SGD, lr 0.1, momentum 0.9, weight decay 1e-4, batch 128;
  - lr divided by 10 at 50% and 75% of the phase;
  - random crop with 4 pixels of padding and horizontal flip.

  We keep the model of the last epoch.
- **Seeds 0, 1, 2.** They change the initialization and the batches; the splits are fixed.

### Data

- **Training images.** 45,000 for training and 5,000 for val, fixed split (`data/cifar.py`). The val set is never used
  to train the network: it is used for the confidence units and, later, for temperature scaling.
- **Test set.** All 10,000 images are kept; the calibration / evaluation split is chosen at analysis time.
- **CIFAR-10-C and CIFAR-100-C** (Hendrycks and Dietterich 2019; Zenodo records 2535967 and 3555552): 19 corruptions,
  15 standard and 4 extra, each at 5 severities. Row (s − 1)·10000 + i of a file is test image i at severity s.
  The tests check this order for every corruption by comparing images, because the labels file is shared.

### Confidence units

- One logistic regression per exit on the head features, fitted on the val set with the network frozen.
  The L2 strength is chosen by 5-fold cross validation.
- **Target "label"** (is the exit correct): fitted at export time. It is a post hoc approximation of the learned
  confidence of Pomponi et al. 2022, not a reproduction: they train it together with the network, with a class
  balanced loss.
- **Why not class balanced.** We use the plain loss because Fast yet Safe uses one threshold for all exits, and class
  balancing would shift the score of each exit by a different amount.
- **Target "agreement"** (does the exit agree with exit 9) uses the same function, in the week 4 analysis.

### How to run

```bash
bash scripts/download_cifar_c.sh                 # CIFAR-10-C and CIFAR-100-C, about 5.8 GB, md5 checked
# training: on Colab with notebooks/colab_training.ipynb (networks saved on Drive), or locally:
python scripts/run_trainings.py --seeds 0 --jobs 1
# copy the folder outputs/cifar from Drive into outputs/cifar, then on the Mac:
caffeinate -i bash scripts/export_all.sh         # val, test and CIFAR-C outputs of every network
python scripts/first_look.py                     # accuracy per exit and a first check with Fast yet Safe
pytest -rs
```

- **Training outputs:**
  - each network goes in `outputs/cifar/<dataset>/<regime>/seed<k>/`, with `model.pt`, `args.json` and `log.csv`
    (loss and val accuracy of every exit at each epoch);
  - phase 1 goes in `outputs/cifar/<dataset>/backbone/seed<k>/`.
- **Export outputs** (`preds/`, numpy arrays that can be opened with `mmap_mode="r"`):
  - logits (N, 9, C), confidences (N, 9) and labels for val and test;
  - head features (N, 4352, float16) for val and test;
  - per corruption, logits (5, 10000, 9, C) and confidences;
  - CIFAR-C features only for seed 0, because they take about 8.3 GB per network;
  - also `conf_label.pt` and `metrics.json`, with the accuracy of every exit, clean and per corruption and severity.
- **Timing.** On the Mac (Apple M5, MPS) one epoch takes about 18 s, mostly GPU time: about 14 s for a training step
  without data loading. Two trainings at the same time do not go faster, so the full block (about 6,000 epochs) runs
  on Colab.
- **Reproducibility.** Results on MPS and CUDA are not reproducible bit by bit.
