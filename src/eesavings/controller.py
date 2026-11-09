# Threshold controller with random audits (project document, section "Garanzie"; proof in docs/controller_bound.md).
# After every input the shared threshold moves up when an audit finds a disagreement with exit 9 and slightly
# down otherwise:
#   lambda_{t+1} = max(0, lambda_t + gamma * (B_t * z_t / pi_t - eps))
# B_t = 1 if the input was audited (probability pi_t, known and >= pi_min), z_t = 1 if the exit used disagrees
# with exit 9. It is an adaptation of Kiyani et al. 2026 (Theorem 5.1) and Gibbs et al. 2026 (section 5.1).

import numpy as np


def controller_step(lam, audited, z, pi, eps, gamma):
    # one update of the threshold; z only matters when the input was audited
    return max(0.0, lam + gamma * (audited * z / pi - eps))


def long_run_bound(T, eps, gamma, pi_min, delta):
    # with probability at least 1 - delta, for any stream: mean of z_t over T inputs <= this value
    # (deterministic part + Freedman's inequality, see docs/controller_bound.md)
    deterministic = (1 + gamma / pi_min) / (gamma * T)
    log_term = np.log(1 / delta)
    martingale = np.sqrt(2 * (1 - pi_min) * log_term / (pi_min * T)) + 2 * log_term / (3 * T)
    return eps + deterministic + martingale
