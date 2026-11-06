# Compute savings of an early exit network.
# An input that exits at exit l costs c_l, the full network costs c_L.
# Savings S = 1 - mean(c_exit) / c_L = sum over l < L of pi_l * (1 - c_l / c_L),
# where pi_l is the fraction of inputs that exit at l. Exits are 1-indexed like in Fast yet Safe.

import numpy as np

# Cumulative MACs (millions) of a CIFAR ResNet20 after each of its 9 blocks, heads excluded.
# Computed by hand from the architecture; to check with a FLOP counter on the real model.
RESNET20_CIFAR_CUM_MMACS = np.array([5.16, 9.88, 14.60, 18.14, 22.86, 27.57, 31.11, 35.83, 40.55])


def cost_fractions(cum_costs):
    # cost of each exit as a fraction of the full network
    cum_costs = np.asarray(cum_costs, dtype=float)
    return cum_costs / cum_costs[-1]


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
