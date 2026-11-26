# Fable's reading of the follow-up B2.02 (15 November 2026): the small maps on the logits (temperature, bias, scale
# and bias) against the full affine map and against M8b, on the cases of split 0. Gains over m2 with a bootstrap
# interval over the corruptions, and the share of the gain of the full affine map (logit_grid_n4000) each one keeps.
# It only prints. Usage (from the repository root): python outputs/fable_review/pilots/verifica_fable/b2_mech_read.py

import glob

import numpy as np
import pandas as pd

B2 = "outputs/fable_review/pilots/B2_adattamento"
KEY = ["dataset", "regime", "corruption", "severity", "split"]


def read(sub):
    return pd.concat([pd.read_csv(f, dtype={"l2": str}) for f in sorted(glob.glob(f"{B2}/{sub}/*.csv"))],
                     ignore_index=True)


def boot(values, corruptions, reps=2000, seed=0):
    s = pd.Series(np.asarray(values, dtype=float)).groupby(np.asarray(corruptions)).agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(s), size=(reps, len(s)))
    means = s["sum"].to_numpy()[idx].sum(axis=1) / s["size"].to_numpy()[idx].sum(axis=1)
    return float(np.mean(values)), *np.percentile(means, [2.5, 97.5])


def main():
    pd.set_option("display.width", 220)
    hd, mech = read("runs_heads"), read("runs_mech")
    hd = hd[hd.split == 0]
    base = hd[hd.config == "m2"].set_index(KEY)
    rows = pd.concat([hd[hd.config.isin(["head_grid_n4000", "logit_grid_n4000", "logit_grid_n1000",
                                         "logit_k20_n1000"])], mech], ignore_index=True)
    idx = pd.MultiIndex.from_frame(rows[KEY])
    for col in ["savings", "acc_system", "savings_acc05"]:
        rows[col + "_m2"] = base[col].reindex(idx).to_numpy()
    rows["gain"] = rows.savings - rows.savings_m2
    rows["gain_acc05"] = rows.savings_acc05 - rows.savings_acc05_m2
    print(len(mech), "rows of the follow-up;", mech.groupby(["dataset", "severity"]).size().to_dict())
    order = ["head_grid_n4000", "logit_grid_n4000", "logit_diag_n4000", "logit_bias_n4000", "logit_temp_n4000",
             "logit_grid_n1000", "logit_k20_n1000", "logit_diag_n1000", "logit_bias_n1000", "logit_temp_n1000"]
    for (dataset, severity), g in rows.groupby(["dataset", "severity"]):
        ref = g[g.config == "logit_grid_n4000"].gain.mean()
        out = []
        for config in order:
            c = g[g.config == config]
            if not len(c):
                continue
            m, lo, hi = boot(c.gain, c.corruption)
            out.append({"config": config, "gain": f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]", "of_affine": m / ref,
                        "positive": (c.gain > 0).mean(), "above_eps": (c.risk > 0.05).mean(),
                        "acc_system": (c.acc_system - c.acc_system_m2).mean(), "gain_acc05": c.gain_acc05.mean(),
                        "gain_eqb": (c.savings_eqb - c.savings_m2).mean(), "train_fwd": c.train_fwd.mean()})
        print(f"\n{dataset} severity {severity} (m2 {g.savings_m2.mean():.4f}; of_affine = share of the gain of "
              f"logit_grid_n4000)")
        print(pd.DataFrame(out).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    reg = rows[rows.config.isin(["logit_temp_n4000", "logit_bias_n4000", "logit_diag_n4000", "logit_grid_n4000"])]
    print("\nby regime (gain):\n", reg.groupby(["dataset", "severity", "regime", "config"]).gain.mean().unstack("config")
          .round(4).to_string())


if __name__ == "__main__":
    main()
