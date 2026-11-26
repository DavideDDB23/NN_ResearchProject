# Piloti B1 e B2: testi proposti per il tracciamento

Regola 8 del piano: ogni scoperta diventa una proposta di testo per `docs/findings.md` (in inglese, nello stile del
file) e una riga per il log del documento di progetto (in italiano). **Niente di questo è stato scritto nei file della
repository né nel documento online.** Aspetto il tuo ok, anche parziale.

Contenuto:
1. sezione nuova per `docs/findings.md`;
2. righe per il log del documento di progetto (file a parte, controllato con lo script dei caratteri vietati).

Nota: gli script dei piloti stanno in `outputs/fable_review/pilots/`, fuori dalla repository. Se la sezione entra in
`docs/findings.md`, conviene prima decidere se portare gli script in `scripts/` con un test, come per il blocco 3.

---

## 1. Sezione proposta per `docs/findings.md`

Da aggiungere in fondo, dopo "M8b on three seeds".

```markdown
## Pilots B1 and B2 (14 to 15 November 2026, working tree after `a3eadcf`)

Two offline pilots on the networks and logits of block 2, before the online part. Scripts, logs and tables are in
`outputs/fable_review/pilots/` (not in the repository yet); the notebook entries are B1.01, B2.01, B2.02 and B2.03
of `outputs/fable_review/quaderno.md`. Exploration only: the 15 standard corruptions, eps 0.05, delta 0.1, max
softmax.

- **B1: where the savings go under shift** (24 networks, clean and severity 1, 3, 5, the 5 splits).
  - At severity 3 an oracle that sends every input to the first exit that agrees with exit 9 saves 0.677 on CIFAR10
    and 0.466 on CIFAR100. M2 certifies 0.311 and 0.123.
  - The statistical price of the certificate (best shared threshold minus M2) is 0.015 and 0.009: 2% to 4% of the
    gap between M2 and the oracle with a budget of eps.
  - Eight thresholds, one per exit, compared with a shared threshold at the same risk on fresh rows, add 0.019 (95%
    interval 0.012 to 0.027) and 0.017 (0.014 to 0.020): 5% of the gap. They add at least 0.02 in 3 of the 8
    settings.
  - Compared without matching the risk, the eight thresholds seemed to add 0.027: fitted on the same rows, they hold
    a risk of 0.055 on fresh rows against 0.049 of the shared threshold.
  - From clean to severity 3 the oracle keeps 89% and 81% of its savings, M2 keeps 59% and 53%.
  - With a shared threshold chosen on the accuracy lost (with labels, 5% of the accuracy of exit 9) the savings at
    severity 3 are 0.85 and 1.05 of the clean ones.
- **B2: which adaptation pays back its compute** (seed 0, joint and joint_sd, severity 3 and 5, split 0, 30 cases per
  dataset and severity).
  - M8b gains 0.053 and 0.022 over M2 at severity 3. Its training costs 2,569 and 30,262 full forwards, and it pays
    back after 74,300 inputs on CIFAR10 and 1.39 million on CIFAR100.
  - An affine map on the logits of each head, fitted by the same distillation, keeps 91% and 114% of that gain with
    38 and 5,679 forwards.
  - One scale and one bias per class on those logits keep 86% of the gain on CIFAR10 with 0.1 forwards and 750 rows
    (payback about 5,600 inputs), and 65% on CIFAR100 with 11 forwards and 3,000 rows (payback about 33,700).
  - With the same 3,000 audits, 750 for the adaptation and 2,250 for the calibration give 0.039 to 0.043 more
    savings than M2 on CIFAR10 at severity 3, and 0.006 on CIFAR100.
  - The rows that reach exit 9 anyway cost nothing, but are worth less than as many random rows: 55% to 95% of
    their gain when L2 is chosen on held-out rows, from 10% to about as much with a fixed L2.
  - Re-estimating the BatchNorm statistics raises the accuracy of exit 9 by 7 to 16 points and leaves the certified
    savings where they were (−0.005 and +0.003 at severity 3). Under noise on CIFAR10 they fall by 0.07.
  - The two measures rank the methods in opposite ways. At the compute of M2, M8b changes the accuracy of the system
    by +0.8 and −0.2 points, the BatchNorm statistics by +7.7 and +8.0.
  - On CIFAR100 the gain of M8b is agreement with a wrong reference. The accuracy of the heads does not move, their
    agreement with exit 9 rises by 4.9 points, and exit 9 is wrong on 78% of the new agreements (92% under noise).
    Certified on the accuracy lost, the gain is +0.014 (−0.006 to +0.036) at severity 3 and −0.020 at severity 5.
  - Hypothesis D1 of the plan (a cheap variant with 70% of the gain, 10% of the compute and a payback below 30,000
    inputs on both datasets) does not hold on these four networks: no configuration meets it on CIFAR100.
- **B2.03: the maps on the logits on all 24 networks** (splits 0 to 2, 540 cases per dataset and severity; M2 and
  M8b from `outputs/savings_under_shift_v2/results.csv`).
  - One scale and one bias per class, fitted on the 3,000 rows M8b fits its heads on, keep 77% (95% interval 64% to
    88%) of the gain of M8b on CIFAR10 and 76% (53% to 111%) on CIFAR100 at severity 3, and 59% and 51% at severity
    5. Per seed, at severity 3: 74% to 81% and 72% to 78%.
  - Their training costs 0.4 and 11 full forwards, against 2,604 and 30,860 for M8b. They pay back after about
    27,300 and 28,900 inputs, against 86,000 and 1.7 million: on these networks they meet D1, at the edge. They were
    chosen after B2, so this is a replication on 20 more networks, not a confirmation.
  - With 750 rows they keep 70% on CIFAR10 and 29% on CIFAR100. A temperature per head alone keeps 9% and 14%, and
    helps only in joint_sd.
  - The share of cases above eps stays within the noise of M2 (11.5% against 13.3% on CIFAR10 and 6.7% against
    11.1% on CIFAR100 at severity 3).
- **Limits.** The variants on the heads and the BatchNorm statistics have one seed and two regimes; the maps on the
  logits have the 24 networks. Everything is offline, with the domain known, on the 15 standard corruptions.
```

## 2. Righe per il log del documento di progetto

In `outputs/fable_review/03_righe_log_documento.txt`: quattro righe, nello stile dei testi per il professore.
