# Compute savings of early exit networks under distribution shift

Neural Networks project (Sapienza, prof. Scardapane).

We study how the compute savings of an early exit network change under distribution shift
when its label-free risk control (Fast yet Safe, Jazbec et al., NeurIPS 2024) must stay valid,
and how much of those savings an adaptation phase of the exit heads can recover, net of its cost.
The full plan is in the project document "Compute savings degli early exit sotto shift"
(https://claude.ai/artifact/MNreFiy1XTNWcYZcUC8axi).

This first part sets up the code and reproduces Fast yet Safe on its released ImageNet logits.

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
- `src/eesavings/savings.py`: compute savings and their decomposition over the exits.
- `scripts/download_fys_logits.sh`: downloads the logits released with Fast yet Safe (ImageNet about 3.4 GB, CIFAR100 about 60 MB).
- `scripts/reproduce_fys.py`: reproduces their image classification experiment.
- `tests/`: unit tests, plus a test that compares our code with the original on the same data.

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
