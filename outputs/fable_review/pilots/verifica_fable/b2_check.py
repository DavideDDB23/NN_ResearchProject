# Fable's independent check of pilot B2 (15 November 2026). It does NOT import the pilot's code. On a few random
# cases it builds again, from the saved logits, features, weights and images, some configurations of
# outputs/fable_review/pilots/B2_adattamento/ and compares savings, risk and accuracy with the rows the jobs wrote:
#   m2                 the UCB choice with the slow reference analysis.ucb_index
#   head_k10_n1000     the linear heads re-fitted with my own closure (cross entropy with the soft targets of exit 9,
#                      which differs from the KL of the package by a constant), 10 L-BFGS iterations, L2 0.01
#   logit_k10_free     the affine map on the logits, fitted on the adapt rows that reach exit 9 under M2 (my own
#                      selection of the rows)
#   logit_grid_n500    the affine map with the L2 chosen on the held-out rows
#   bn_n1000           the BatchNorm statistics through PyTorch itself (training mode, one batch), not the hand
#                      written forward of the pilot
# The fits are not bit for bit the pilot's (another formula for the loss), so the comparison has a tolerance.
# It only prints. Usage (from the repository root):
#   python outputs/fable_review/pilots/verifica_fable/b2_check.py cifar10 3

import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts/6_block3_review")
from eesavings.adaptation import L2_GRID  # noqa: E402
from eesavings.analysis import LAMBDAS, make_splits, ucb_index  # noqa: E402
from eesavings.data.cifar import CORRUPTIONS, NUM_CLASSES, load_cifar_c, to_tensor_batch  # noqa: E402
from eesavings.export import predict  # noqa: E402
from eesavings.fys import softmax  # noqa: E402
from eesavings.models import FEATURE_DIMS, MultiExitResNet20  # noqa: E402
from eesavings.train import load_model  # noqa: E402
from review_common import fractions_of  # noqa: E402

B2 = "outputs/fable_review/pilots/B2_adattamento"
EPS, DELTA = 0.05, 0.1
TOL = 3e-3


def exits_of(conf8, lam):
    passed = conf8 > lam
    return np.where(passed.any(axis=1), passed.argmax(axis=1), 8)


def measure(new, labels, cal, ev, fr):
    # savings, risk, system accuracy on eval at the lambda chosen with the slow UCB on cal; new: (10000, 9, C)
    rows = np.concatenate([cal, ev])
    z = new[rows]
    conf8 = np.stack([softmax(z[:, l], axis=1).max(axis=1) for l in range(8)], axis=1).astype(np.float64)
    preds = z.argmax(axis=2)
    ex = np.stack([exits_of(conf8, lam) for lam in LAMBDAS])
    used = preds[np.arange(len(rows))[None, :], ex]
    loss = (used != preds[:, 8][None, :]).astype(np.float32)
    k = ucb_index(loss[:, :len(cal)], EPS, DELTA)
    e = slice(len(cal), None)
    return {"k": k, "savings": 1 - fr[ex[k, e]].mean(), "risk": loss[k, e].mean(),
            "acc_system": np.mean(used[k, e] == labels[rows][e])}


def fit_linear(x, teacher, w0, b0, l2, max_iter):
    # cross entropy with soft targets plus l2 * ||w - w0||^2, L-BFGS with the strong Wolfe line search
    w, b = torch.tensor(w0, requires_grad=True), torch.tensor(b0, requires_grad=True)
    start = torch.tensor(w0)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -(teacher * F.log_softmax(x @ w.T + b, dim=1)).sum(dim=1).mean() + l2 * ((w - start) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def soft_ce(x, teacher, w, b):
    return float(-(teacher * F.log_softmax(x @ w.T + b, dim=1)).sum(dim=1).mean())


def blocks(features):
    ends = np.cumsum(FEATURE_DIMS)
    return [torch.as_tensor(np.asarray(features[:, s:e], dtype=np.float32)) for s, e in zip(np.r_[0, ends[:-1]], ends)]


def check_case(dataset, regime, corruption, severity, split, device):
    torch.set_num_threads(1)
    run = f"outputs/cifar/{dataset}/{regime}/seed0"
    fr = fractions_of(dataset)
    c = NUM_CLASSES[dataset]
    logits = np.asarray(np.load(f"{run}/preds/cifar_c/{corruption}_logits.npy", mmap_mode="r")[severity - 1])
    features = np.asarray(np.load(f"{run}/preds/cifar_c/{corruption}_features.npy", mmap_mode="r")[severity - 1])
    images, labels = load_cifar_c(dataset, corruption, "data")
    state = torch.load(f"{run}/model.pt", map_location="cpu", weights_only=True)
    adapt, cal, ev = make_splits(5)[split]
    mine = {"m2": measure(logits, labels, cal, ev, fr)}

    def teacher(rows):
        return F.softmax(torch.as_tensor(logits[rows, 8], dtype=torch.float32), dim=1)

    # head_k10_n1000: the first 750 adapt rows
    fit = adapt[:750]
    x = blocks(features[fit])
    new = logits.copy()
    all_blocks = blocks(features)
    for l in range(8):
        w0 = state[f"heads.{l}.classifier.weight"].numpy()
        b0 = state[f"heads.{l}.classifier.bias"].numpy()
        w, b = fit_linear(x[l], teacher(fit), w0, b0, 0.01, 10)
        new[:, l] = (all_blocks[l] @ w.T + b).numpy()
    mine["head_k10_n1000"] = {**measure(new, labels, cal, ev, fr), "n_fit": len(fit)}

    # logit_k10_free: the adapt rows that reach exit 9 at the M2 lambda (the first three quarters of them)
    conf8 = np.stack([softmax(logits[adapt][:, l], axis=1).max(axis=1) for l in range(8)], axis=1).astype(np.float64)
    free = adapt[exits_of(conf8, LAMBDAS[mine["m2"]["k"]]) == 8]
    fit = free[:(3 * len(free)) // 4]
    new = logits.copy()
    for l in range(8):
        z = torch.as_tensor(np.ascontiguousarray(logits[fit, l]))
        w, b = fit_linear(z, teacher(fit), np.eye(c, dtype=np.float32), np.zeros(c, dtype=np.float32), 0.01, 10)
        new[:, l] = (torch.as_tensor(np.ascontiguousarray(logits[:, l])) @ w.T + b).numpy()
    mine["logit_k10_free"] = {**measure(new, labels, cal, ev, fr), "n_fit": len(fit), "n_rows_set": len(free)}

    # logit_grid_n500: 375 rows fit, 125 choose L2
    fit, held = adapt[:375], adapt[375:500]
    new, chosen = logits.copy(), []
    for l in range(8):
        z, zh = (torch.as_tensor(np.ascontiguousarray(logits[r, l])) for r in (fit, held))
        best = None
        for l2 in L2_GRID:
            w, b = fit_linear(z, teacher(fit), np.eye(c, dtype=np.float32), np.zeros(c, dtype=np.float32), l2, 100)
            val = soft_ce(zh, teacher(held), w, b)
            if best is None or val < best[0]:
                best = (val, w, b, l2)
        chosen.append(best[3])
        new[:, l] = (torch.as_tensor(np.ascontiguousarray(logits[:, l])) @ best[1].T + best[2]).numpy()
    mine["logit_grid_n500"] = {**measure(new, labels, cal, ev, fr), "l2": " ".join(f"{v:g}" for v in chosen)}

    # bn_n1000 through PyTorch: running statistics reset, cumulative average, training mode, one batch
    model = load_model(MultiExitResNet20(c), f"{run}/model.pt").to(device).eval()
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None
            mod.train()
    x_all = np.asarray(images[severity - 1])
    with torch.no_grad():
        m(to_tensor_batch(x_all[adapt[:1000]], dataset).to(device))
    m.eval()
    rows = np.concatenate([cal, ev])
    new = np.full(logits.shape, np.nan, dtype=np.float32)
    new[rows] = predict(m, x_all[rows], dataset, device)[0]
    mine["bn_n1000"] = measure(new, labels, cal, ev, fr)

    name = f"{dataset}_{regime}_seed0_{corruption}"
    theirs = pd.concat([pd.read_csv(f"{B2}/runs_{job}/{name}.csv", dtype={"l2": str}) for job in ("heads", "bn")
                        if os.path.exists(f"{B2}/runs_{job}/{name}.csv")])
    theirs = theirs[(theirs.severity == severity) & (theirs.split == split)].set_index("config")
    out = []
    for config, v in mine.items():
        if config not in theirs.index:
            out.append({"config": config, "note": "not in the outputs"})
            continue
        t = theirs.loc[config]
        row = {"config": config, "savings": v["savings"], "d_savings": v["savings"] - t.savings,
               "d_risk": v["risk"] - t.risk, "d_acc": v["acc_system"] - t.acc_system, "d_k": v["k"] - t.k}
        for extra in ("n_fit", "n_rows_set"):
            if extra in v:
                row["d_" + extra] = v[extra] - t[extra]
        if "l2" in v:
            row["l2_equal"] = v["l2"] == str(t.l2)
        out.append(row)
    return pd.DataFrame(out).assign(case=f"{dataset}/{regime} {corruption} sev{severity} split{split}")


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(20261115 + len(dataset))
    tables = []
    for _ in range(n):
        case = (dataset, ["joint", "joint_sd"][rng.integers(2)], CORRUPTIONS[rng.integers(len(CORRUPTIONS))],
                [3, 5][rng.integers(2)], 0)
        tables.append(check_case(*case, device))
        print("done", case, flush=True)
    t = pd.concat(tables, ignore_index=True)
    pd.set_option("display.width", 220)
    print(t.round(5).to_string(index=False))
    cols = [c for c in ["d_savings", "d_risk", "d_acc"] if c in t]
    worst = t[cols].abs().max()
    print("\nlargest absolute differences:", worst.round(5).to_dict(), "; tolerance", TOL,
          "-> OK" if (worst <= TOL).all() else "-> ABOVE THE TOLERANCE")


if __name__ == "__main__":
    main()
