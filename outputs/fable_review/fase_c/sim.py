# Phase C of the plan v3 (the online part of the project document): a simulator of streams of inputs over the saved
# logits of a multi exit ResNet20. Written by Opus on 15 November 2026. Design and predictions:
# outputs/fable_review/quaderno.md, entry C.01. Nothing here trains a network: everything is a lookup in the logits
# of outputs/cifar, plus the small maps on the logits of the heads chosen in pilot B2 (scale and bias per class).
#
# Rows. For every domain (clean, or a corruption at a severity) the 10,000 test images are split once with a fixed
# permutation: the first 5,000 are the calibration pool (used only by the offline references M1, M2, M8), the other
# 5,000 the stream pool. A stream draws, at every step and independently for every repetition, one row of the stream
# pool of the domain of that step (with replacement).
#
# One step for one input (project document, section "Il sistema online"):
#   exit         the first exit l < 9 with score s_l > lambda_t (max softmax of the heads in use), otherwise exit 9
#   audit        if the input left early, with probability pi_t the rest of the network runs (B_t = 1)
#   z_t          1 if the exit used disagrees with exit 9; it is observed only for an audit (it is 0 at exit 9)
#   controller   lambda_{t+1} = min(lam_max, max(0, lambda_t + gamma (B_t z_t / pi_t - eps)))  (controller.py)
#   buffer       during an adaptation, the rows that reach exit 9 (audits and/or the inputs that got there alone)
#   adaptation   when the buffer is full: one scale and one bias per class on the logits of every head, fitted by
#                distillation towards exit 9 (pilot B2.02, the "diag" map of b2_mech.py); the new heads are used from
#                the next input on (condition 3 of docs/controller_bound.md); backbone and exit 9 never change
# Compute of an input, in full forwards of the plain ResNet20 (fractions_of): the cost of its exit, plus the rest of
# the network if it is an audit; the training of the maps is added at the step where it happens.
#
# The simulation is vectorized over the repetitions: every array has a first axis of size R.

import numpy as np
import torch
import torch.nn.functional as F

from eesavings.adaptation import MAX_ITER
from eesavings.analysis import LAMBDAS, consistency_losses, ucb_index_fast
from eesavings.savings import RESNET20_FULL_MACS

EPS, DELTA = 0.05, 0.1
N_POOL = 5000


def pools(n=10000, seed=0):
    # calibration pool and stream pool: the same for every domain, so a row is the same image in every domain
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[:N_POOL]), np.sort(perm[N_POOL:])


def softmax_max(logits):
    # logits (..., C) -> max softmax and argmax
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return p.max(axis=-1), logits.argmax(axis=-1)


def apply_diag(logits, scale, bias):
    # logits (N, 9, C); scale, bias (8, C) or None -> logits of the heads mapped, exit 9 unchanged
    if scale is None:
        return logits
    out = logits.copy()
    out[:, :8] = logits[:, :8] * scale[None] + bias[None]
    return out


def tables(logits, scale=None, bias=None):
    # scores (N, 8) and predictions (N, 9) of a domain with the given maps
    s, p = softmax_max(apply_diag(logits, scale, bias))
    return s[:, :8].astype(np.float32), p.astype(np.int16)


def fit_diag(logits, weights=None):
    # one scale and one bias per class for every head, by distillation towards the softmax of exit 9 on these rows
    # (the "diag" map of b2_mech.py: same loss, full batch L-BFGS with the strong Wolfe line search, no penalty).
    # weights (entry C.09): one per row, the inverse of its probability of being observed, so that the loss is an
    # unbiased estimate of the loss on the whole domain; None is the plain mean over the rows.
    # Returns scale (8, C), bias (8, C) and the number of closure evaluations.
    torch.set_num_threads(1)
    teacher = F.softmax(torch.as_tensor(logits[:, 8], dtype=torch.float32), dim=1)
    w = None if weights is None else torch.as_tensor(np.asarray(weights, np.float64) / np.sum(weights),
                                                      dtype=torch.float32)
    c = logits.shape[2]
    scales, biases, evals = np.ones((8, c), np.float32), np.zeros((8, c), np.float32), 0
    for l in range(8):
        z = torch.as_tensor(np.ascontiguousarray(logits[:, l]), dtype=torch.float32)
        d, a = torch.ones(c, requires_grad=True), torch.zeros(c, requires_grad=True)
        opt = torch.optim.LBFGS([d, a], max_iter=MAX_ITER, line_search_fn="strong_wolfe")
        count = [0]

        def closure():
            count[0] += 1
            opt.zero_grad()
            log_p = F.log_softmax(z * d + a, dim=1)
            if w is None:
                loss = F.kl_div(log_p, teacher, reduction="batchmean")
            else:
                loss = (w * F.kl_div(log_p, teacher, reduction="none").sum(dim=1)).sum()
            loss.backward()
            return loss

        opt.step(closure)
        scales[l], biases[l], evals = d.detach().numpy(), a.detach().numpy(), evals + count[0]
    return scales, biases, evals


def exit1_probs(logits):
    # softmax of exit 1 with the source head, (N, C) float32: the label-free signal of the guard (entry C.21)
    z = logits[:, 0].astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def fit_cost(evals, n_rows, n_classes, dataset):
    # training compute in full forwards: every evaluation is a forward and a backward of N x C products
    return evals * 2 * n_rows * n_classes / RESNET20_FULL_MACS[dataset]


def risk_at(logits, weights, scale, bias, lam):
    # self-normalized weighted estimate of the risk of the new heads at one lambda, and its standard error
    s, p = tables(logits, scale, bias)
    above = s > lam
    e = np.where(above.any(axis=1), above.argmax(axis=1), 8)
    z = (p[np.arange(len(e)), e] != p[:, 8]).astype(float)
    w = weights / weights.sum()
    r = float((w * z).sum())
    return r, float(np.sqrt(((z - r) ** 2 * w ** 2).sum()))


def restart_lambda(logits, weights, fr, scale, bias, margin=0.0):
    # Threshold for new heads, from rows that reached exit 9 during the adaptation (logits (n, 9, C)), each weighted
    # by one over its probability of being observed (1 if it got there alone, 1 / pi_t if it was an audit):
    # self-normalized estimates of risk and savings along the grid, and the lambda with the largest savings among
    # those with risk + margin * (standard error of the risk estimate) <= eps. Used to restart the controller after
    # the heads change; the controller corrects it from there. The margin (entry C.04) offsets the winner's curse of
    # taking the largest savings under a noisy constraint, which matters when the restart is repeated (M7c).
    s, p = tables(logits, scale, bias)
    above = s[None, :, :] > LAMBDAS[:, None, None]
    exits = np.where(above.any(axis=2), above.argmax(axis=2), 8)                    # (K, n)
    z = (np.take_along_axis(p[None, :, :], exits[..., None], axis=2)[..., 0] != p[None, :, 8]).astype(float)
    w = weights / weights.sum()
    risk = (z * w[None]).sum(axis=1)
    se = np.sqrt(((z - risk[:, None]) ** 2 * (w ** 2)[None]).sum(axis=1))
    savings = 1.0 - (fr[exits] * w[None]).sum(axis=1)
    k = int(np.argmax(np.where(risk + margin * se <= EPS, savings, -np.inf)))
    return float(LAMBDAS[k])


def ucb_lambda(logits, scale=None, bias=None):
    # Fast yet Safe: lambda with UCB (WSR, delta 0.1) on these rows, max softmax of the heads in use
    mapped = apply_diag(logits, scale, bias)
    s, _ = tables(logits, scale, bias)
    conf = np.concatenate([s, np.ones((len(s), 1), np.float32)], axis=1)
    losses, _ = consistency_losses(mapped, conf)
    return float(LAMBDAS[ucb_index_fast(losses, EPS, DELTA)])


class Stream:
    # domain of every step (the same for every repetition) and the row of the stream pool, per repetition
    def __init__(self, segments, n_rep, seed):
        # segments: list of (domain name, length)
        self.segments = segments
        self.names = []
        for name, _ in segments:
            if name not in self.names:
                self.names.append(name)
        self.dom = np.concatenate([np.full(n, self.names.index(name)) for name, n in segments])
        self.T = len(self.dom)
        self.idx = np.random.default_rng(seed).integers(N_POOL, size=(n_rep, self.T))
        self.starts = list(np.cumsum([0] + [n for _, n in segments])[:-1])


def simulate(stream, base, fr, dataset, method, lam_init, pi, gamma, seed, lam_max=1.0):
    # Runs one method on one stream for every repetition.
    # base: {"logits": {domain name: logits of the stream pool (N_POOL, 9, C)}, "labels": {domain name: labels}}
    # method: a dict with
    #   "controller": bool                     the threshold moves (M4 to M7)
    #   "lam_segments": list or None           a fixed threshold per segment (M1, M2, M8)
    #   "maps_segments": list or None          fixed maps (scale, bias) per segment, or None for the source heads (M8)
    #   "monitor": None or {"betas", "threshold"}   fallback to the full network when an e-detector fires (M3)
    #   "phase": None or {"trigger": "oracle", "t_start"} or {"trigger": "lambda", "kappa", "window", "t_min"},
    #            plus "pi_a", "n_target" (training rows that end the phase), "rows" ("audit", "final", "both": the
    #            rows the maps are trained on) and "restart" (after the fit, restart the controller from
    #            restart_lambda on every row observed during the phase, whatever the training rows)
    #   "continuous": None or {"k_refit", "n_target", "n_min", "restart"}: every k_refit inputs the maps are fitted
    #            again on the last n_target rows that reached exit 9 (audits and the others)
    #   "weighted" inside "phase" or "continuous" (entry C.09): the maps are fitted with every row weighted by the
    #            inverse of its probability of being observed (M7w, M7cw); absent or False, the plain mean
    #   "guard": None or {"kappa", "window"} (entry C.21): after every fit, a label-free detector on the exit-1
    #            probabilities removes the maps and puts lambda back when the domain changes (M7wg)
    #   "swap": None or {"t", "maps", "lam"}: at step t every repetition gets the given maps and threshold (M8c:
    #            the oracle maps of the new domain with the same controller as M7)
    # Methods with a fixed threshold and no monitor (M1, M2, M8) make no audits: they are offline references.
    # Returns the per step records (R, T) and the list of the adaptations.
    rng = np.random.default_rng(seed)
    R, T = stream.idx.shape
    rep = np.arange(R)
    names = stream.names
    labels = [base["labels"][n] for n in names]
    S, P = [], []
    for n in names:
        s, p = tables(base["logits"][n])
        S.append(np.broadcast_to(s, (R,) + s.shape).copy())
        P.append(np.broadcast_to(p, (R,) + p.shape).copy())
    lam = np.full(R, lam_init, dtype=np.float64)
    out = {k: np.zeros((R, T), dtype=np.float32) for k in ["cost", "gross", "z", "audit", "correct", "lam"]}
    out["exit"] = np.zeros((R, T), dtype=np.int8)
    events = []
    ph, cont, mon = method.get("phase"), method.get("continuous"), method.get("monitor")
    in_phase = np.zeros(R, dtype=bool)
    done_phase = np.zeros(R, dtype=bool)
    ema = np.full(R, lam_init)
    buffers = [[] for _ in range(R)]
    fallback = np.zeros(R, dtype=bool)
    betas = np.asarray(mon["betas"]) if mon else None
    wealth = np.ones((R, len(betas))) if mon else None
    fallback_t = np.full(R, -1)
    no_audit = method.get("lam_segments") is not None and not mon
    kind = ph["rows"] if ph is not None else "both"
    hold = (ph or cont).get("restart_holdout", 0.0) if (ph is not None or cont is not None) else 0.0
    k_hold = int(round(1 / hold)) if hold else 0
    n_train = np.zeros(R, dtype=int)  # training rows of the allowed kind in the buffer of a phase (kept as a count)

    def split_rows(rows):
        # with restart_holdout = 1/k, every k-th observed row (in order of arrival, so a random subset) is kept out
        # of the training of the maps and used only to restart lambda (added on 15 November, entry C.04: with the
        # same rows for both the restart was optimistic and lambda restarted too low)
        if not hold:
            return rows, rows
        return ([row for i, row in enumerate(rows) if i % k_hold != k_hold - 1],
                [row for i, row in enumerate(rows) if i % k_hold == k_hold - 1])

    def training_rows(rows):
        rows = split_rows(rows)[0]
        if kind == "both":
            return rows
        return [row for row in rows if row[2] == (kind == "audit")]

    swap = method.get("swap")
    # entry C.21: after a fit, a label-free detector compares the moving average of the exit-1 probabilities (source
    # head) with its value at the fit; above guard["kappa"] in total variation it removes the maps and puts lambda back
    # to its value just before the restart of that fit (from there the method is M4)
    guard = method.get("guard")
    if guard is not None:
        P1 = [exit1_probs(base["logits"][n]) for n in names]
        src_tables = [tables(base["logits"][n]) for n in names]
        ema_p = np.tile(exit1_probs(base["cal"]["clean"]).mean(axis=0), (R, 1)).astype(np.float64)
        ref_p = np.zeros_like(ema_p)
        armed = np.zeros(R, dtype=bool)
        lam_pre = np.zeros(R)
    for t in range(T):
        d = stream.dom[t]
        if swap is not None and t == swap["t"]:
            for k, n in enumerate(names):
                s, p = tables(base["logits"][n], *swap["maps"])
                S[k][:], P[k][:] = s[None], p[None]
            lam[:] = swap["lam"]
        if method.get("lam_segments") is not None:
            seg = int(np.searchsorted(stream.starts, t, side="right") - 1)
            lam[:] = method["lam_segments"][seg]
            if t == stream.starts[seg] and method.get("maps_segments") is not None:
                maps = method["maps_segments"][seg]
                for k, n in enumerate(names):
                    s, p = tables(base["logits"][n], *(maps if maps is not None else (None, None)))
                    S[k][:], P[k][:] = s[None], p[None]
        if ph is not None:
            if ph["trigger"] == "oracle":
                start = np.full(R, t == ph["t_start"])
            else:
                ema += (lam - ema) / ph["window"]
                start = (ema > lam_init + ph["kappa"]) & (t >= ph["t_min"])
            new = start & ~in_phase & ~done_phase
            in_phase |= new
            for r in np.flatnonzero(new):
                events.append({"rep": int(r), "t": t, "kind": "phase_start"})
        idx = stream.idx[:, t]
        if guard is not None:
            ema_p += (P1[d][idx] - ema_p) / guard["window"]
            if armed.any():
                tv = 0.5 * np.abs(ema_p - ref_p).sum(axis=1)
                for r in np.flatnonzero(armed & (tv > guard["kappa"])):
                    for k in range(len(names)):
                        S[k][r], P[k][r] = src_tables[k]
                    lam[r] = lam_pre[r]
                    armed[r] = False
                    if cont is not None:
                        buffers[r] = []
                    events.append({"rep": int(r), "t": t, "kind": "guard_fire", "tv": float(tv[r])})
        s = S[d][rep, idx]
        p = P[d][rep, idx]
        lam_used = np.where(fallback, 1.0, lam)
        above = s > lam_used[:, None]
        e = np.where(above.any(axis=1), above.argmax(axis=1), 8)
        pe = p[rep, e]
        z = (pe != p[:, 8]) & (e < 8)
        pi_t = np.where(in_phase, ph["pi_a"], pi) if ph is not None else np.full(R, pi)
        audit = (rng.random(R) < pi_t) & (e < 8) & (not no_audit)
        out["cost"][:, t] = fr[e] + audit * (fr[8] - fr[e])
        out["gross"][:, t] = 1.0 - fr[e]
        out["z"][:, t] = z
        out["audit"][:, t] = audit
        out["correct"][:, t] = pe == labels[d][idx]
        out["lam"][:, t] = lam_used
        out["exit"][:, t] = e
        x = audit * z / pi_t
        if method.get("controller"):
            lam = np.minimum(lam_max, np.maximum(0.0, lam + gamma * (x - EPS)))
        if mon:
            wealth = np.maximum(1.0, wealth) * (1.0 + betas[None] * (x[:, None] - EPS))
            fire = (wealth >= mon["threshold"]).any(axis=1) & ~fallback
            fallback_t[fire] = t
            fallback |= fire
        # every row that reached exit 9 during an adaptation is observed: (domain, row, audit?, probability of
        # being observed); the maps are trained on the rows of the allowed kind
        if ph is not None or cont is not None:
            active = in_phase if ph is not None else np.ones(R, dtype=bool)
            keep = None if cont is None else int(round(cont["n_target"] / (1 - cont.get("restart_holdout", 0.0))))
            for r in np.flatnonzero((audit | (e == 8)) & active):
                pos = len(buffers[r])
                buffers[r].append((d, int(idx[r]), bool(audit[r]), float(pi_t[r]) if audit[r] else 1.0))
                if ph is not None and (not hold or pos % k_hold != k_hold - 1) and \
                        (kind == "both" or bool(audit[r]) == (kind == "audit")):
                    n_train[r] += 1
                if cont is not None and len(buffers[r]) > keep:
                    buffers[r].pop(0)
        if ph is not None:
            to_fit = [r for r in np.flatnonzero(in_phase) if n_train[r] >= ph["n_target"]]
        elif cont is not None and (t + 1) % cont["k_refit"] == 0:
            to_fit = [r for r in range(R) if len(buffers[r]) >= cont["n_min"]]
        else:
            to_fit = []
        for r in to_fit:
            rows = training_rows(buffers[r])
            logits = np.stack([base["logits"][names[dd]][ii] for dd, ii, _, _ in rows])
            # entry C.09: with "weighted" every row weighs 1/q, q its probability of being observed
            weights = np.array([1.0 / q for _, _, _, q in rows]) if (ph or cont).get("weighted") else None
            scale, bias, evals = fit_diag(logits, weights)
            train = fit_cost(evals, len(rows), logits.shape[2], dataset)
            out["cost"][r, t] += train
            for k, n in enumerate(names):
                S[k][r], P[k][r] = tables(base["logits"][n], scale, bias)
            lam_before = float(lam[r])
            if (ph or cont).get("restart"):
                obs = split_rows(buffers[r])[1]
                logits_obs = np.stack([base["logits"][names[dd]][ii] for dd, ii, _, _ in obs])
                weights = np.array([1.0 / q for _, _, _, q in obs])
                # test before the restart (entry C.04): restart only if, with the new heads, the risk estimated at the
                # current lambda is more than restart_test standard errors away from eps (0: always restart)
                test = (ph or cont).get("restart_test", 0.0)
                r_cur, se_cur = risk_at(logits_obs, weights, scale, bias, lam[r]) if test else (np.inf, 0.0)
                if abs(r_cur - EPS) > test * se_cur:
                    lam[r] = restart_lambda(logits_obs, weights, fr, scale, bias,
                                            (ph or cont).get("restart_margin", 0.0))
            if guard is not None:  # entry C.21: the reference of the detector is the domain seen at the fit
                ref_p[r], armed[r], lam_pre[r] = ema_p[r], True, lam_before
            events.append({"rep": int(r), "t": t, "kind": "fit", "n_rows": len(rows),
                           "n_audit": int(sum(row[2] for row in rows)), "n_observed": len(buffers[r]),
                           "n_restart": len(split_rows(buffers[r])[1]) if hold else 0,
                           "train_fwd": float(train), "lam_before": lam_before, "lam_after": float(lam[r])})
            if ph is not None:
                in_phase[r], done_phase[r] = False, True
                buffers[r], n_train[r] = [], 0
    if mon:
        for r in range(R):
            events.append({"rep": r, "t": int(fallback_t[r]), "kind": "fallback"})
    return out, events
