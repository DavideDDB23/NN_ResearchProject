# Parte 1: testi proposti per il tracciamento

Regola 8 del piano: ogni scoperta diventa una proposta di testo per `docs/findings.md` (in inglese, nello stile del
file) e una riga per il log del documento di progetto (in italiano). **Niente di questo è stato scritto nei file della
repository.** Aspetto il tuo ok, anche parziale.

Contenuto:
1. sezione nuova per `docs/findings.md`;
2. righe per il log del documento di progetto;
3. frasi di README e findings da riscrivere (rilievo R12);
4. due test da aggiungere (rilievi R15 e R09).

---

## 1. Sezione proposta per `docs/findings.md`

Da aggiungere in fondo, dopo "Preliminary checks before the next block".

```markdown
## Review of blocks 1 to 3 (11 to 13 November 2026, commit `a3eadcf`)

A check of everything above, made before the next block. Scripts, logs and tables are in
`outputs/fable_review/parte1/`, the report in `outputs/fable_review/01_revisione.md`.

- **The code and the numbers hold.**
  - All 125 tests pass and the Fast yet Safe reproduction is identical.
  - Nine rows of `results.csv`, recomputed with a measurement written from scratch, match to 3e-9.
  - Every number of the README and of this file that was checked matches the outputs. No leakage was found.
- **Exit 9 is a weak reference where the whole network is trained.**
  - In joint, mixed and joint_sd, exit 9 is at most as accurate as exit 8 also under shift: 0.8 to 1.5 points below
    on CIFAR100. An ensemble of exits 7 to 9 is 1.4 to 2.4 points above exit 9 on CIFAR100.
  - At the λ certified by M2 on CIFAR100, the early exit system is more accurate than exit 9. When the exit used and
    exit 9 disagree on clean data, the early exit is right in 47% to 50% of the cases and exit 9 in 28% to 31%.
  - With the ensemble as the reference, the clean savings go from 0.231 to 0.293 on CIFAR100 and from 0.526 to 0.549
    on CIFAR10. About half of the gain on CIFAR100 is mechanical, because exits 7 and 8 vote in the reference: when
    only exits 1 to 6 may fire, the gain is +0.032 on CIFAR100 and +0.016 on CIFAR10.
- **The "valid savings" of M1 are a fragile metric.**
  - 12% to 43% of the M1 cases have a risk within two standard errors of ε, so their valid savings depend on the eval
    sample.
  - M1 and M2 are two points of the same curve of savings against risk. On CIFAR10, M1 keeps 0.35 to 0.46 of savings
    at a risk of 0.065 to 0.127.
- **M2 is valid, and it is an optimistic reference.**
  - With 100 splits per domain, its λ has a risk above ε in 8.9% and 7.8% of the calibrations on the 3,000 eval rows
    (CIFAR10 and CIFAR100), in 6.7% and 5.7% on the 7,000 rows outside the calibration, and in 3.4% and 2.8% on the
    whole test set.
  - An oracle of the threshold saves only 0.009 to 0.015 more.
  - M2 knows the domain and spends 3,000 full forwards on it. Over 10,000 inputs its net savings are 30% lower.
  - With fewer calibration rows the savings fall: 0.526 with 3,000 rows and 0.403 with 100 on clean CIFAR10. On
    CIFAR100 the share left at severity 3 falls from 0.53 to 0.45.
  - CRC hardly depends on the number of rows (0.53 with 100), but it controls the risk only on average: 38% to 47% of
    its calibrations are above ε.
- **As it is, the adaptation of the heads does not pay for its compute.**
  - Training the heads costs as much as 2,700 full forwards on CIFAR10 and 31,000 on CIFAR100.
  - At severity 3 it pays back after 83,000 and 1.7 million inputs of the same domain.
  - On CIFAR100 most of the cost is the training, not the audits. On CIFAR10 it depends on how the audits are
    counted: the training is 2.2 times their extra cost over M2, and 0.67 times their 4,000 full forwards.
- **Two claims do not hold.**
  - joint_sd above joint on CIFAR10: +0.011 with a standard error of 0.020, from one seed (differences by seed:
    −0.010, −0.004, +0.048). The seeds also differ in the sign of α at exits 1 to 3, which may explain a part of it
    (correlation −0.54 inside a setting, on 18 points). On CIFAR100 the advantage is clear: +0.036 with a standard
    error of 0.005.
  - The only setting that passes the week 4 criterion (0.53) has a 95% interval from 0.36 to 0.71.
- **The split of the drop depends on the order.** The shift at the clean λ is 61% of the drop on CIFAR10 in the order
  of the README and 71% in the other order (85% and 88% on CIFAR100).
- **Intervals.** At severity 3 the share of the clean savings left is 0.59 (0.56 to 0.62) on CIFAR10 and 0.53 (0.50
  to 0.56) on CIFAR100. It goes from 0.41 for glass blur to 0.95 for brightness.
- **The controller bound is correct and loose.**
  - It holds against five adaptive adversaries, and it fails, as it must, when the adversary sees the audit coin.
  - With ε = 0.05 and π = 0.05 it is 0.156 after 10,000 inputs and 0.081 after 100,000.
- **Smaller points.**
  - The pooling of the heads is not counted in the MACs: with it the heads cost 0.52% of the network on CIFAR10, not
    0.12%.
  - A five times finer λ grid gives +0.002 to +0.003 of savings and the same ranking of the scores.
  - No latency has been measured yet.
```

---

## 2. Righe proposte per il log del documento di progetto

Formato della tabella "Cosa è fatto" (data, cosa, esito).

| Data | Cosa | Esito |
|---|---|---|
| 13 novembre 2026 | Revisione critica dei blocchi 1–3 | Codice e numeri confermati: 125 test, riproduzione identica, righe di results.csv rigenerate da zero. Nessun bug e nessun leakage. 24 rilievi in outputs/fable_review/01_revisione.md |
| 13 novembre 2026 | Controllo del riferimento (exit 9) | Nei regimi che allenano tutta la rete l'exit 9 è al massimo accurato quanto l'exit 8 anche sotto shift, e su CIFAR100 pulito ha ragione solo nel 28% dei disaccordi. Con un ensemble degli exit 7–9 come riferimento i savings puliti su CIFAR100 passano da 0,231 a 0,293, per circa metà per un effetto meccanico. Da decidere quale rischio certificare prima della parte online |
| 13 novembre 2026 | Metrica di M1 | I valid savings dipendono dal rumore sulle righe di eval. M1 va riportato come coppia rischio e savings |
| 13 novembre 2026 | Validità e costo di M2 | Violazioni al 6–7% sulle righe fuori dalla calibrazione, sotto δ. M2 resta un limite ottimistico: conosce il dominio e spende 3000 forward complete. Con poche righe di calibrazione i savings calano, e CRC va riportato accanto a UCB |
| 13 novembre 2026 | Compute dell'adattamento delle teste (M8b) | Il training vale 2.700 forward complete su CIFAR10 e 31.000 su CIFAR100. A severity 3 si ripaga dopo 83.000 e 1,7 milioni di input. H4 va precisata: su CIFAR100 domina il training, su CIFAR10 dipende da come si contano gli audit |
| 13 novembre 2026 | Regimi e criterio della settimana 4 | Il vantaggio di joint_sd regge su CIFAR100 ma non su CIFAR10, dove viene da un solo seed. Il criterio della settimana 4 resta non superato |
| 13 novembre 2026 | Scomposizione del calo e intervalli | La quota dello shift a λ ferma è 61–71% su CIFAR10 e 85–88% su CIFAR100 secondo l'ordine. A severity 3 resta 0,59 [0,56, 0,62] e 0,53 [0,50, 0,56] dei savings puliti |
| 13 novembre 2026 | Bound del controller | Dimostrazione ricontrollata. Regge contro avversari adattivi ma è largo sugli orizzonti reali: 0,156 dopo 10.000 input con ε = 0,05 |

---

## 3. Frasi da riscrivere (rilievo R12)

| Dove | Ora | Proposta |
|---|---|---|
| README 330 | "λ calibrated again on the corrupted images: the best savings without adaptation." | "λ calibrated again on 3,000 images of the corrupted domain, which is assumed to be known. It is an optimistic reference without adaptation, and its calibration cost is not subtracted." |
| `analysis.py:8` | "the best savings without adaptation" | "an optimistic reference without adaptation" |
| README 332 | "It needs no labels." | "It needs no labels, but it needs 7,000 full forwards on the new domain (4,000 to adapt and 3,000 to calibrate)." |
| README 372–373 | "but they are illusory: once the cases that break the risk control are discarded, almost nothing is left." | "but they are bought with a risk above ε: M1 breaks the risk control in 58% to 90% of the cases on CIFAR10." |
| README 379 | "joint_sd keeps the most savings under shift, then joint and mixed." | "On CIFAR100 joint_sd keeps the most savings under shift (+0.036 over joint, on all three seeds). On CIFAR10 the difference (+0.011) is inside the spread of the seeds." |
| README 388–391 | "This is about 60% of the drop on CIFAR10 and 85% on CIFAR100." e "the remaining 40% ... and 15%" | "This is 61% to 71% of the drop on CIFAR10 and 85% to 88% on CIFAR100, depending on which of the two steps is taken first." |
| README 410 | "**Reason.** A well calibrated P(agreement) ..." | "**Likely reason.** ... With M8a the traffic moves from exits 1 to 4 to exits 5 to 8." |
| README 433–434 | "it recovers a median of 11% to 25% of the lost savings." | "it recovers a median of 11% to 25% of the lost savings on seed 0, before counting its compute." |
| README 439 | "The criterion passes in 1 of the 8 settings (CIFAR10 joint_sd)." | "The criterion is not met. One setting is just above the line (CIFAR10 joint_sd, 0.53 with a 95% interval from 0.36 to 0.71)." |
| README 453 | "checks it on simulated adversarial and drifting streams." | "checks it on a worst case stream, on a drifting stream and on adaptive adversaries." (dopo aver aggiunto il test) |
| Findings 114, 121, 131 | Le stesse tre frasi su "illusory", joint_sd e criterio | Stesse correzioni |

---

## 4. Test proposti

**Per `tests/test_savings_under_shift.py` (rilievo R15).** Il test attuale usa solo un rischio che cresce con l'indice.

```python
def test_ucb_index_matches_fast_yet_safe_when_the_risk_is_not_monotone():
    # rows that pass and rows that fail alternate, and row 0 is sometimes unsafe
    rng = np.random.default_rng(0)
    for trial in range(20):
        risk = rng.uniform(0, 0.3, size=30)
        if trial % 4 == 0:
            risk[0] = 0.9
        losses = (rng.random((30, 400)) < risk[:, None]).astype(float)
        for eps in [0.02, 0.05, 0.1, 0.2]:
            assert ucb_index(losses, eps, 0.1) == ucb_lam(losses, eps, 0.1)
```

**Per `tests/test_controller.py` (rilievo R09).** Un avversario che usa la storia, e il caso che deve fallire.

```python
def test_the_bound_holds_against_an_adversary_that_uses_the_history():
    # it disagrees only after an input that was not audited; audits always at the lowest probability
    eps, gamma, pi, T, delta = 0.05, 0.01, 0.05, 5000, 0.1
    bound = long_run_bound(T, eps, gamma, pi, delta)
    violations = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        lam, previous_audited, total = 0.5, True, 0
        for t in range(T):
            z = int(not previous_audited) if lam < 1 else 0
            audited = rng.random() < pi
            total += z
            lam = controller_step(lam, audited, z, pi, eps, gamma)
            previous_audited = audited
        violations += total / T > bound
    assert violations / 200 <= delta


def test_the_bound_fails_if_the_adversary_sees_the_audit_coin():
    # condition 1 broken on purpose: the test must be able to tell a wrong bound
    eps, gamma, pi, T, delta = 0.05, 0.01, 0.05, 5000, 0.1
    rng = np.random.default_rng(0)
    lam, total = 0.5, 0
    for t in range(T):
        audited = rng.random() < pi
        z = int(not audited) if lam < 1 else 0
        total += z
        lam = controller_step(lam, audited, z, pi, eps, gamma)
    assert total / T > long_run_bound(T, eps, gamma, pi, delta)
```

Entrambi i comportamenti sono già verificati in `outputs/fable_review/parte1/check_fast_ucb.py` e
`controller_stress.py`.

