# Reproduce the image classification results of Fast yet Safe on their ImageNet logits.
# Same settings as their notebook (RC-EENN/img_cls/img_cls.ipynb): lambda from 1 to 0 with
# step 0.01, delta = 0.1, eps from 0.01 to 0.50, n_cal = 100, UCB (WSR) and CRC.
# Besides the mean exit (their efficiency measure) we also save the savings with a cost
# proportional to the exit index, 1 - mean_exit / L, since the real FLOPs per exit are not
# in their files.
#
# Usage:
#   python scripts/2_fys_reproduction/reproduce_fys.py              (5 splits, like the notebook)
#   python scripts/2_fys_reproduction/reproduce_fys.py --trials 100 (100 splits, like the paper)

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eesavings.data.fys_logits import read_img_cls_data
from eesavings.fys import losses_and_exits
from eesavings.fys import get_acc_per_exit, get_preds_per_exit, rc_main_fast
from eesavings.savings import exit_distribution, per_exit_disagreement

DATA_DIR = "data/fys"
MODELS = ["msdnet", "dvit", "l2wden", "dynperc"]
RISK_TYPES = ["prediction-consistency", "prediction-gt-gap", "confidence-brier", "confidence-brier-top-pred"]
RCP_TYPES = ["ucb-wsr", "crc"]
LAMBDAS = np.arange(0.0, 1.001, 0.01)[::-1]  # descending, like in their code
EPSILONS = np.arange(0.01, 0.51, 0.01)
DELTA = 0.1
N_CAL = 100

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=5)
parser.add_argument("--models", nargs="+", default=MODELS)
args = parser.parse_args()

out_dir = f"outputs/fys/trials{args.trials}"
os.makedirs(out_dir, exist_ok=True)

rows = []
model_info = {}

for model in args.models:
    logits, targets = read_img_cls_data(DATA_DIR, model, "imagenet")
    n_exits = logits.shape[1]
    acc = get_acc_per_exit(get_preds_per_exit(logits), targets)
    model_info[model] = {"n_exits": n_exits, "acc_per_exit": [round(float(a), 4) for a in acc]}
    print(model, "exits:", n_exits, "accuracy per exit:", np.round(acc, 3))

    for risk_type in RISK_TYPES:
        start = time.time()
        losses, exits = losses_and_exits(logits, targets, LAMBDAS, risk_type)
        mean_exit = exits.mean(axis=1)
        loss_bound = 2 if "brier" in risk_type else 1

        test_risk, eff_gains, _, lam_ids = rc_main_fast(
            losses, mean_exit, EPSILONS, RCP_TYPES, loss_bound=loss_bound,
            n_trials=args.trials, n_cal=N_CAL, delta=DELTA)

        for rcp in RCP_TYPES:
            risk = np.array(test_risk[rcp])  # (trials, eps)
            eff = np.array(eff_gains[rcp])
            for e, eps in enumerate(EPSILONS):
                rows.append({
                    "model": model,
                    "risk_type": risk_type,
                    "rcp": rcp,
                    "eps": round(eps, 2),
                    "test_risk_mean": risk[:, e].mean(),
                    "test_risk_std": risk[:, e].std(),
                    "violation_rate": (risk[:, e] > eps).mean(),
                    "mean_exit": eff[:, e].mean(),
                    "savings_uniform_cost": 1 - eff[:, e].mean() / n_exits,
                    "lambda_median": np.median(LAMBDAS[lam_ids[rcp][:, e]]),
                })

        # for the consistency risk also save how inputs spread over the exits
        if risk_type == "prediction-consistency":
            model_info[model]["by_lambda"] = {}
            for lam in [0.9, 0.8, 0.7, 0.6, 0.5]:
                k = np.argmin(np.abs(LAMBDAS - lam))
                q = per_exit_disagreement(exits[k], losses[k], n_exits)
                model_info[model]["by_lambda"][str(lam)] = {
                    "consistency_risk": round(float(losses[k].mean()), 4),
                    "pi": exit_distribution(exits[k], n_exits).round(4).tolist(),
                    "q": [None if np.isnan(v) else round(float(v), 4) for v in q],
                }

        print(f"  {risk_type}: {time.time() - start:.0f}s")

df = pd.DataFrame(rows)
df.to_csv(f"{out_dir}/fys_reproduction.csv", index=False)
with open(f"{out_dir}/models.json", "w") as f:
    json.dump(model_info, f, indent=2)

# figure like the one in their notebook: test risk (top) and mean exit / L (bottom) against eps
fig, ax = plt.subplots(2, len(RISK_TYPES), figsize=(16, 6.5), sharex=True)
colors = {"msdnet": "tab:blue", "dvit": "tab:orange", "l2wden": "tab:green", "dynperc": "tab:red"}
styles = {"ucb-wsr": "-", "crc": "--"}
for j, risk_type in enumerate(RISK_TYPES):
    for model in args.models:
        for rcp in RCP_TYPES:
            d = df[(df.model == model) & (df.risk_type == risk_type) & (df.rcp == rcp)]
            ax[0, j].plot(d.eps, d.test_risk_mean, styles[rcp], color=colors[model], label=f"{model} {rcp}")
            ax[1, j].plot(d.eps, d.mean_exit / model_info[model]["n_exits"], styles[rcp], color=colors[model])
    ax[0, j].plot([0, 0.5], [0, 0.5], color="grey", lw=0.8)
    ax[0, j].set_title(risk_type)
    ax[1, j].set_xlabel("eps")
ax[0, 0].set_ylabel("test risk")
ax[1, 0].set_ylabel("mean exit / L")
ax[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{out_dir}/fys_reproduction.png", dpi=150)

# check the guarantees, for each eps: CRC keeps the expected test risk below eps,
# UCB keeps the probability of a test risk above eps below delta.
# With T splits these are noisy estimates (a violation rate moves in steps of 1/T),
# so the check means something only with --trials 100.
# CRC also needs a loss that does not grow when lambda decreases: prediction-gt-gap can be
# negative and breaks this, so its small excess at low eps is real (see the README).
crc = df[df.rcp == "crc"]
ucb = df[df.rcp == "ucb-wsr"]
print("\nCRC, largest (mean test risk - eps) over eps (about <= 0 up to noise, except prediction-gt-gap):")
print((crc.test_risk_mean - crc.eps).groupby([crc.model, crc.risk_type]).max().round(4).to_string())
print(f"\nUCB, violation rate over eps, mean and largest (each eps should be about <= {DELTA}):")
print(ucb.groupby(["model", "risk_type"]).violation_rate.agg(["mean", "max"]).round(3).to_string())
if args.trials < 100:
    print(f"\nonly {args.trials} splits: run with --trials 100 for a real check of the guarantees")
print("\nsaved in", out_dir)
