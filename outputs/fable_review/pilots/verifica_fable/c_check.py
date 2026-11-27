# Checks of the Phase C simulator (outputs/fable_review/fase_c/sim.py), written apart from it (15 November 2026):
#   1. M1 and M2 in the main job: gross savings and risk of the post-shift segment, simulated, against the same
#      quantities computed directly on the 5,000 rows of the stream pool at their thresholds (the stream draws from
#      them uniformly), within 4 standard errors of the sampling
#   2. the threshold path of the controller, for one repetition, against eesavings.controller.controller_step run in a
#      plain loop on the same inputs and audits
#   3. fit_diag of sim.py against fit_map("diag") of pilot B2.02 on the same rows
#   4. restart_lambda with equal weights against the empirical oracle on the same rows
# It only prints. Usage (from the repository root): python outputs/fable_review/pilots/verifica_fable/c_check.py

import glob
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "outputs/fable_review/fase_c")
sys.path.insert(0, "outputs/fable_review/pilots/B2_adattamento")
sys.path.insert(0, "scripts/6_block3_review")
from b2_mech import apply_map, fit_map  # noqa: E402
from eesavings.analysis import LAMBDAS  # noqa: E402
from eesavings.controller import controller_step  # noqa: E402
from eesavings.fys import softmax  # noqa: E402
from review_common import fractions_of, load_logits  # noqa: E402
from sim import EPS, Stream, fit_diag, pools, restart_lambda, simulate  # noqa: E402

MAIN = "outputs/fable_review/fase_c/main"


def direct(logits, lam, fr):
    # gross savings and risk of a threshold on the given logits (max softmax, strict rule)
    conf = np.stack([softmax(logits[:, l], axis=1).max(axis=1) for l in range(8)], axis=1)
    preds = logits.argmax(axis=2)
    above = conf > lam
    e = np.where(above.any(axis=1), above.argmax(axis=1), 8)
    z = (preds[np.arange(len(e)), e] != preds[:, 8]) & (e < 8)
    return 1 - fr[e], z.astype(float)


def check_1():
    print("1. M1 and M2, simulated against direct (post-shift segment of the main job)")
    s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{MAIN}/summary/*.csv"))], ignore_index=True)
    rng = np.random.default_rng(5)
    keys = s[["dataset", "regime", "seed", "domain"]].drop_duplicates().to_numpy()
    worst = 0.0
    for dataset, regime, seed, domain in keys[rng.choice(len(keys), size=8, replace=False)]:
        refs = json.load(open(f"{MAIN}/refs/{dataset}_{regime}_seed{seed}_{domain.replace(':', '_')}.json"))
        corruption, severity = domain.rsplit(":", 1)
        logits = load_logits(dataset, regime, int(seed), corruption, int(severity))[pools()[1]]
        fr = fractions_of(dataset)
        for method, lam in [("M1", refs["lam1"]), ("M2", refs[f"lam2_{domain}"])]:
            g, z = direct(logits, lam, fr)
            row = s[(s.dataset == dataset) & (s.regime == regime) & (s.seed == seed) & (s.domain == domain)
                    & (s.method == method) & (s.window == "post_all")]
            n = 30000 * len(row)
            for name, sim_value, values in [("gross", row.gross.mean(), g), ("risk", row.risk.mean(), z)]:
                se = values.std() / np.sqrt(n)
                gap = (sim_value - values.mean()) / se
                worst = max(worst, abs(gap))
                print(f"   {dataset} {regime} s{seed} {domain} {method} {name}: simulated {sim_value:.4f}, direct "
                      f"{values.mean():.4f}, gap {gap:+.2f} standard errors")
    print(f"   largest gap {worst:.2f} standard errors ->", "OK" if worst < 4 else "CHECK FAILED")


def check_2():
    print("\n2. Threshold path of the controller against controller_step (cifar10 joint seed0, frost:3, 1 repetition)")
    dataset, fr = "cifar10", fractions_of("cifar10")
    cal, stream_rows = pools()
    clean = load_logits(dataset, "joint", 0)
    shifted = load_logits(dataset, "joint", 0, "frost", 3)
    base = {"logits": {"clean": clean[stream_rows], "frost:3": shifted[stream_rows]},
            "labels": {"clean": np.zeros(5000, int), "frost:3": np.zeros(5000, int)}}
    stream = Stream([("clean", 3000), ("frost:3", 7000)], 1, seed=3)
    out, _ = simulate(stream, base, fr, dataset, {"controller": True}, 0.88, 0.05, 2e-4, seed=4)
    lam, path = 0.88, []
    for t in range(stream.T):
        path.append(lam)
        logits = base["logits"][stream.names[stream.dom[t]]][stream.idx[0, t]][None]
        g, z = direct(logits, lam, fr)
        audited = out["audit"][0, t]
        assert abs(out["z"][0, t] - z[0]) < 1e-9 and abs(out["gross"][0, t] - g[0]) < 1e-6, t
        lam = controller_step(lam, audited, z[0], 0.05, EPS, 2e-4)
        lam = min(1.0, lam)
    gap = np.abs(np.array(path) - out["lam"][0]).max()
    print(f"   {stream.T} steps: exits and disagreements identical; largest gap in lambda {gap:.2e} ->",
          "OK" if gap < 1e-6 else "CHECK FAILED")


def check_3_4():
    print("\n3. fit_diag of sim.py against fit_map('diag') of B2.02 (cifar100 joint seed0, gaussian_noise:3, 3,000 rows)")
    logits = load_logits("cifar100", "joint", 0, "gaussian_noise", 3)[:3000]
    scale, bias, _ = fit_diag(logits)
    teacher = torch.softmax(torch.as_tensor(logits[:, 8]), dim=1)
    gaps = []
    for l in range(8):
        z = torch.as_tensor(np.ascontiguousarray(logits[:, l]))
        params, _ = fit_map("diag", z, teacher)
        mapped_b2 = apply_map("diag", z, params).numpy()
        gaps.append(np.abs(mapped_b2 - (logits[:, l] * scale[l] + bias[l])).max())
    print(f"   largest gap in the mapped logits over the 8 heads: {max(gaps):.2e} ->",
          "OK" if max(gaps) < 1e-3 else "CHECK FAILED")
    print("\n4. restart_lambda with equal weights against the empirical oracle (same rows, no maps)")
    fr = fractions_of("cifar100")
    ones = np.ones(len(logits))
    lam_r = restart_lambda(logits, ones, fr, np.ones((8, 100), np.float32), np.zeros((8, 100), np.float32))
    best = None
    for lam in LAMBDAS:
        g, z = direct(logits, lam, fr)
        if z.mean() <= EPS and (best is None or g.mean() > best[0]):
            best = (g.mean(), lam)
    print(f"   restart {lam_r:.6f}, oracle {best[1]:.6f} ->", "OK" if abs(lam_r - best[1]) < 1e-12 else "CHECK FAILED")


if __name__ == "__main__":
    check_1()
    check_2()
    check_3_4()
