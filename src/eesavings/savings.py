# Compute savings of an early exit network.
# An input that exits at exit l costs c_l, the full network costs c_L.
# Savings S = 1 - mean(c_exit) / c_L = sum over l < L of pi_l * (1 - c_l / c_L),
# where pi_l is the fraction of inputs that exit at l. Exits are 1-indexed like in Fast yet Safe.

import numpy as np

# Cumulative MACs (millions) of a CIFAR ResNet20 after each of its 9 blocks, heads excluded.
# Computed by hand from the architecture, confirmed by the FLOP counter in tests/test_models.py.
RESNET20_CIFAR_CUM_MMACS = np.array([5.16, 9.88, 14.60, 18.14, 22.86, 27.57, 31.11, 35.83, 40.55])

# Cost (MACs) of each exit of our multi exit ResNet20 (eesavings/models.py): blocks 1..l and pooled heads 1..l,
# exit 9 also pays the final layer. RESNET20_FULL_MACS is the plain ResNet20 without intermediate heads,
# the reference of the savings. Counted with models.exit_costs (checked in tests/test_models.py).
RESNET20_EXIT_MACS = {
    "cifar10": np.array([5163776, 9885184, 14606592, 18151168, 22875392, 27599616, 31149824, 35879680, 40598912]),
    "cifar100": np.array([5186816, 9931264, 14675712, 18266368, 23036672, 27806976, 31449344, 36271360, 40996352]),
}
RESNET20_FULL_MACS = {"cifar10": 40551040, "cifar100": 40556800}


def cost_fractions(cum_costs, full_cost=None):
    # cost of each exit as a fraction of the full network (by default the cost of the last exit)
    cum_costs = np.asarray(cum_costs, dtype=float)
    if full_cost is None:
        full_cost = cum_costs[-1]
    return cum_costs / full_cost


def uniform_cost_fractions(n_exits):
    # cost proportional to the exit index (l / L), used when the real costs are unknown
    return np.arange(1, n_exits + 1) / n_exits


def savings_from_exits(exits, fractions):
    # exits: (N,) or (K, N) array of exits, returns one value per row
    exits = np.asarray(exits)
    fractions = np.asarray(fractions, dtype=float)
    return 1.0 - fractions[exits - 1].mean(axis=-1)


def exit_distribution(exits, n_exits):
    # pi_l: fraction of inputs that exit at each exit
    exits = np.asarray(exits)
    counts = np.bincount(exits - 1, minlength=n_exits)
    return counts / exits.size


def per_exit_disagreement(exits, losses, n_exits):
    # q_l: mean loss of the inputs that exit at l (nan if nobody exits there)
    exits = np.asarray(exits)
    q = np.full(n_exits, np.nan)
    for l in range(n_exits):
        mask = exits == l + 1
        if mask.any():
            q[l] = np.mean(losses[mask])
    return q


def savings_change_by_exit(pi_before, pi_after, fractions):
    # contribution of each exit to S_before - S_after; the entries sum to the total change
    return (np.asarray(pi_before) - np.asarray(pi_after)) * (1.0 - np.asarray(fractions))
