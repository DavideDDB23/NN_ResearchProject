# Long-run bound for the threshold controller with random audits

This note proves the guarantee of the controller in the project document (section "Garanzie"). It is an
adaptation of Theorem 5.1 of Kiyani et al. 2026 and of section 5.1 of Gibbs et al. 2026 to an early exit
network with one shared threshold, not a new result. The code is `src/eesavings/controller.py`. The bound is
checked numerically in `tests/test_controller.py`.

## Setting

- **Inputs and exits.**
  - Inputs $x_1, x_2, \dots$ arrive one at a time.
  - Exit $l = 1, \dots, L$ gives a prediction $\hat y_l(x)$ and a score $s_l(x) \in [0, 1]$.
  - With threshold $\lambda$ the input leaves at the first exit with $s_l(x) > \lambda$ (strictly), otherwise at
    the final exit $L$.
- **Disagreement.** $z_t = \mathbb 1[\hat y_{\text{exit used}}(x_t) \neq \hat y_L(x_t)]$ marks a disagreement with
  the final exit; at the final exit $z_t = 0$.
- **Audits.** With probability $\pi_t$ the input is audited: the rest of the network runs and $z_t$ is observed.
  $B_t \in \{0, 1\}$ says whether it was.
- **Update rule.**

```latex
\lambda_{t+1} = \max\Big(0,\ \lambda_t + \gamma\Big(\frac{B_t z_t}{\pi_t} - \varepsilon\Big)\Big), \qquad \lambda_1 \in [0, 1].
```

- **Filtration.** $\mathcal F_{t-1}$ is everything up to input $t-1$: inputs, audits, thresholds and heads.

## The four conditions

1. **Known audit probability, independent draw.** $\pi_t \ge \pi_{\min} > 0$ is fixed before the final exit runs
   on $x_t$, and $B_t$ is drawn independently: $P(B_t = 1 \mid \mathcal F_{t-1}, x_t) = \pi_t$. In particular
   $\pi_t$ may depend on $x_t$, for example on its scores, but not on $\hat y_L(x_t)$.
2. **No exit above 1.** Scores lie in $[0, 1]$ and the exit rule is strict, so $\lambda_t \ge 1$ implies that the
   input reaches the final exit and $z_t = 0$.
3. **Predictable heads and threshold.** $\lambda_t$ and the heads used at step $t$ are functions of
   $\mathcal F_{t-1}$. Re-training the heads on a buffer is allowed if the new heads are used only after the
   training.
4. **Fixed reference.** Backbone and final exit never change, so $z_t$ always measures the disagreement with the
   same full network.

The stream may be chosen by an adversary that sees the past (including past audits) but not $B_t$ before $x_t$
is chosen. No i.i.d. assumption is needed.

## Theorem

Under conditions 1 to 4, for every $T \ge 1$ and $\delta \in (0, 1)$, with probability at least $1 - \delta$

```latex
\frac{1}{T}\sum_{t=1}^{T} z_t \;\le\; \varepsilon \;+\; \frac{1 + \gamma/\pi_{\min}}{\gamma T}
\;+\; \sqrt{\frac{2(1-\pi_{\min})\log(1/\delta)}{\pi_{\min} T}} \;+\; \frac{2\log(1/\delta)}{3T}.
```

## Proof

**Step 1, the threshold stays bounded.** We show $\lambda_t \le 1 + \gamma/\pi_{\min}$ for every $t$, by
induction. It holds for $\lambda_1 \le 1$.

- If $\lambda_t \ge 1$: by condition 2, $z_t = 0$. The update only subtracts $\gamma\varepsilon$, so
  $\lambda_{t+1} \le \lambda_t$.
- If $\lambda_t < 1$: the largest possible step is $\gamma/\pi_t \le \gamma/\pi_{\min}$, so
  $\lambda_{t+1} < 1 + \gamma/\pi_{\min}$.

**Step 2, a deterministic bound on the weighted disagreements.**

- Removing the $\max(0, \cdot)$ can only lower the threshold, so for every $t$:

```latex
\lambda_{t+1} \ge \lambda_t + \gamma\Big(\frac{B_t z_t}{\pi_t} - \varepsilon\Big).
```

- Summing over $t = 1, \dots, T$ and using Step 1 and $\lambda_1 \ge 0$:

```latex
\sum_{t=1}^{T}\Big(\frac{B_t z_t}{\pi_t} - \varepsilon\Big) \le \frac{\lambda_{T+1} - \lambda_1}{\gamma}
\le \frac{1 + \gamma/\pi_{\min}}{\gamma}
\quad\Longrightarrow\quad
\frac{1}{T}\sum_{t=1}^{T}\frac{B_t z_t}{\pi_t} \le \varepsilon + \frac{1 + \gamma/\pi_{\min}}{\gamma T}.
```

This holds on every sample path, whatever the stream.

**Step 3, from the weighted disagreements to the real ones.**

- **The increments.** Let $X_t = z_t - B_t z_t / \pi_t$.
- **Martingale.** By conditions 1 and 3, $z_t$ is a function of $x_t$ and of $\mathcal F_{t-1}$. Hence
  $E[X_t \mid \mathcal F_{t-1}, x_t] = z_t - z_t \pi_t/\pi_t = 0$, and $\sum_t X_t$ is a martingale.
- **Bound on each increment.** If $B_t = 0$ then $X_t = z_t \le 1$. If $B_t = 1$ then $X_t = z_t(1 - 1/\pi_t) \le 0$.
  So $X_t \le 1$: the upward jumps are bounded by 1, not by $1/\pi$.
- **Conditional variance.** It is $z_t^2 (1-\pi_t)/\pi_t \le (1-\pi_{\min})/\pi_{\min}$, so the sum of the
  conditional variances is at most $v = T(1-\pi_{\min})/\pi_{\min}$.
- **Freedman's inequality** (Freedman 1975) then gives $P(\sum_t X_t \ge a) \le \exp(-a^2 / (2v + 2a/3))$.
- **Choosing $a$.** With $a = \sqrt{2 v \log(1/\delta)} + \tfrac{2}{3}\log(1/\delta)$ we have
  $a^2 \ge (2v + 2a/3)\log(1/\delta)$, so the probability is at most $\delta$.
- **Dividing by $T$**, with probability at least $1 - \delta$:

```latex
\frac{1}{T}\sum_{t=1}^{T} z_t \le \frac{1}{T}\sum_{t=1}^{T}\frac{B_t z_t}{\pi_t}
+ \sqrt{\frac{2(1-\pi_{\min})\log(1/\delta)}{\pi_{\min} T}} + \frac{2\log(1/\delta)}{3T}.
```

**Step 4.** Steps 2 and 3 together give the theorem. $\blacksquare$

## Remarks

- **Size of the terms.** With $\pi = 0.05$ and $\delta = 0.1$ the square root term is about 0.030 after 100,000
  inputs and 0.009 after a million, the values written in the project document. Freedman is needed: Azuma's
  inequality would give a term of order $1/(\pi\sqrt T)$ instead of $1/\sqrt{\pi T}$.
- **What the bound does not say.**
  - It bounds the long-run mean of the disagreements, not the risk at a given time.
  - After a shift the risk can stay above $\varepsilon$ until $\lambda$ has climbed, for about
    $(\lambda_Q - \lambda_P)/(\gamma(\bar R_Q - \varepsilon))$ inputs.
  - It says nothing about accuracy, nor about how much the adaptation saves.
- **Adaptation is allowed.** Condition 3 is the only requirement on the heads, so the bound holds for any
  re-training of the heads on audited data, as long as the new heads are used only from the next input on.
- **Non-uniform audits are allowed.** Condition 1 allows audit probabilities that depend on the scores or on the
  cost of the remaining blocks (Neyman allocation with costs), as long as they stay above $\pi_{\min}$.
