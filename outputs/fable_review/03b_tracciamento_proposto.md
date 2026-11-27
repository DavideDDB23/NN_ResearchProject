# Fase C: testi proposti per il tracciamento

Regola 8 del piano: ogni scoperta diventa una proposta di testo per `docs/findings.md` (in inglese, nello stile del
file) e una riga per il log del documento di progetto (in italiano, in `03b_righe_log_documento.txt`, controllata con
lo script dei caratteri vietati). **Niente di questo è stato scritto nei file della repository né nel documento
online.** Aspetto il tuo ok, anche parziale.

Nota: il simulatore e le letture stanno in `outputs/fable_review/`, fuori dalla repository. Se la sezione entra in
`docs/findings.md`, conviene prima decidere se portare `fase_c/sim.py` e `fase_c/run.py` in `scripts/` con un test.

---

## Sezione proposta per `docs/findings.md`

Da aggiungere in fondo, dopo la sezione dei piloti B1 e B2.

```markdown
## Online part (Phase C, 15 November 2026, working tree after `a3eadcf`)

A stream simulator on the cached logits of block 2 (`outputs/fable_review/fase_c/`, not in the repository yet;
notebook entries C.01 to C.17 of `outputs/fable_review/quaderno.md`). Every input leaves at the first exit whose max
softmax is above lambda_t; with probability pi = 0.05 an input that left early is audited (run to exit 9); the
controller of `src/eesavings/controller.py` updates lambda_t with gamma = 2e-4, chosen on clean streams only by a rule
fixed beforehand; net savings count the exit, the audits and the training. eps 0.05, delta 0.1. Streams draw from
the 5,000 test images of each domain not used for calibration. Main job: 24 networks, the 15 corruptions at severity
3 and 5, 10 repetitions, 10,000 clean inputs then 30,000 shifted. Variants, return to clean, sensitivity, steps of
severity and mixed corruptions: the 8 networks of seed 0 at severity 3.

**RQ2, the controller against the best threshold.** Over the last 10,000 inputs the controller is within 0.005 of the
oracle threshold of the new domain (M2): CIFAR10 0.312 against 0.317 at severity 3, 0.222 against 0.223 at severity
5; on CIFAR100 it is above it (0.126 against 0.122, 0.096 against 0.092), because it spends the whole eps budget
(risk 0.050 against 0.042 to 0.044) and pays 0.007 to 0.017 in audits. The price is the transient: on CIFAR10 the
window of 1,000 inputs is above eps + 0.01 for about 6,500 inputs after the shift (risk 0.087 in the first 1,000 at
severity 3), on CIFAR100 for about 1,000. A fixed clean threshold stays at risk 0.087 (CIFAR10, severity 3).

**RQ3, the adaptation phase.** One scale and one bias per class on the logits of every head, fitted by distillation
towards exit 9 on 3,000 rows that reached exit 9 during a phase with audits at 25%; a quarter of the observed rows is
kept out to restart lambda, after a test that the current lambda is off by more than two standard errors. Over the
last 10,000 inputs it recovers 15.5% and 14.3% of the drop on CIFAR10 (+0.032 and +0.042) and 12.5% and 15.7% on
CIFAR100 (+0.012 and +0.020), close to the same maps fitted with access to the new domain. The cost is in the extra
audits, not in the training (0.1% and 6 to 7% of the cost: H4 holds). Over 30,000 inputs after the shift the phase
breaks even at severity 3 (net -0.003 and +0.003) and gains at severity 5 (+0.014 and +0.010); it pays back within
the stream in 39% to 73% of the cases. Accuracy changes by less than one point.

**The rows the maps are fitted on.** At regime the full fallback (every input completed during the phase) recovers
119% and 159% of the phase with audits, and the phase with only the inputs that reached exit 9 recovers 45% and 17%
(H5 holds, with no higher risk). The more the rows are a random sample of the domain, the better the maps. Weighting
every row by the inverse of its probability of being observed (M7w) gives 118% and 129% of the phase with audits at
the same cost and risk, and it pays back within the stream in 52% and 70% of the cases. The continuous version with
the weighted fit is the best on CIFAR10 (+0.042 at regime, +0.028 net over the stream) but on CIFAR100 its maps are
too noisy (risk 0.058 after the shift).

**When the domain changes again.** After an adaptation the maps and the restarted threshold are safe only on the
domain they were fitted on. On a stream that goes back to clean, the controller alone stays below eps (risk 0.033 and
0.042 in the first 1,000 clean inputs), while the adapted methods reach 0.069 to 0.077 on CIFAR10 and 0.115 to 0.144
on CIFAR100, and stay above eps + 0.01 for 6,000 to 10,500 inputs on CIFAR100. With three corruptions in a row the
adapted methods have a mean risk of 0.066 to 0.069 over 30,000 inputs against 0.058 for the controller alone on
CIFAR10, and 0.061 to 0.075 against 0.053 on CIFAR100 (worst windows of 1,000 inputs up to 0.245 against 0.107). With
steps of severity from 1 to 5 the continuous adaptation follows the drift with less risk than the controller alone on
CIFAR10 (0.056 against 0.060); on CIFAR100 no adaptive method beats the controller alone. Adaptation needs a
protection for streams where the shift repeats.

**A guard after the adaptation.** A label-free detector compares the moving average of the exit-1 probabilities
(source head, available for every input) with its value at the fit; when the total variation goes above a threshold
chosen on clean streams only (0.06 on CIFAR10, 0.10 on CIFAR100, no false alarm in 40 clean runs), it removes the maps
and puts lambda back to its value before the restart. After the return to clean it fires in 99 to 100% of the runs, a
median of 235 and 382 inputs after the change, and the long run risk over the stream goes back to that of the
controller alone (0.050 and 0.048, against 0.057 and 0.062 without the guard). With mixed corruptions its risk equals
the controller alone (0.058 and 0.053). When the domain stays it fires in 1.7% and 0% of the runs and the gain of the
phase stays. With repeated shifts a single phase is paid and its gain is lost at the next change (net savings 0.019
and 0.005 below the controller alone).

**Sensitivity.** The audit rate moves net savings by about the cost of the audits (+0.011 at pi = 0.02, -0.016 at
pi = 0.1 on CIFAR10) without changing the gain of the phase. The transient scales roughly as 1/gamma (9,700, 6,250,
2,900 and 2,150 inputs for gamma 1e-4, 2e-4, 5e-4, 1e-3).

**Guarantee.** The long run risk over 40,000 inputs is below the Freedman bound with the restart term in every
repetition (bound 0.22 to 0.25); it is above eps + 0.01 in 12% to 20% of the repetitions on CIFAR10, because of the
transient after the shift.

Limits: offline on cached logits, synthetic corruptions only, oracle trigger for the phase, one phase per stream;
the same stream and audit draws for all corruptions of a network (common random numbers), so the intervals of the
continuous methods are too narrow.
```
