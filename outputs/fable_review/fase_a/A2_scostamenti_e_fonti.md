# A2: modifiche a README e findings, scostamenti e fonti

Esecuzione della specifica del 14 novembre 2026 (aggiornamento di `README.md` e `docs/findings.md` dopo la revisione
dei blocchi 1–3). File in sola lettura: tutti tranne `README.md`, `docs/findings.md` e questo file. Nessun comando git
che modifica, nessun pytest, nessuno script della repository. Solo letture, grep e tre one-liner pandas su CSV
piccoli (`reference_disagreements.csv`, quattro colonne di `m2_validity.csv`).

Stato: completato.

Abbreviazioni: `B3/` = `outputs/block3_review/`, `M1CSV` = `B3/figures/m1_savings_and_risk.csv`,
`REV` = `outputs/fable_review/01_revisione.md`, `TRA` = `outputs/fable_review/01_tracciamento_proposto.md`,
`P1/` = `outputs/fable_review/parte1/`.

## 1. Modifiche applicate

Righe dopo le modifiche.

| ID | File | Righe | Note |
|---|---|---|---|
| R1 | README.md | 181–183 | Budget |
| R2a | README.md | 45–47 | ";" alla riga 45, voce nuova 46–47 |
| R2b | README.md | 66–68 | riga 13 nuova, controller e findings rinumerati 14 e 15 |
| R3 | README.md | 332–333 | M1 |
| R4 | README.md | 334–335 | M2, testo di TRA sez. 3 |
| R5 | README.md | 337–338 | M8b, testo di TRA sez. 3; riga 339 ("stay fixed. ...") e "Only seed 0, ..." non toccate |
| R6 | README.md | 357–359 | scostamento: "often above ε" (sez. 3) |
| R7 | README.md | 363–370 | solo i valori di M1 |
| R8 | README.md | 374–388 | intro, tabella del rischio, figura `risk_by_method.png` |
| R9 | README.md | 395–401 | due sotto-punti; scostamento sull'ultima frase (sez. 3) |
| R10 | README.md | 404–408 | scostamenti di forma e "on CIFAR10" (sez. 3) |
| R11 | README.md | 410–411 | testo di TRA sez. 3 |
| R12 | README.md | 420–422 | |
| R13 | README.md | 441–442 | applicato in parte: ultima frase omessa (sez. 5) |
| R14 | README.md | 464–466 | testo di TRA sez. 3 |
| R15 | README.md | 467–470 | scostamento: "with the extra cost of the 4,000 adaptation inputs" (sez. 3) |
| R16 | README.md | 474–476 | tabella della settimana 4 non toccata |
| R17 | README.md | 502–504 | testo di TRA sez. 3 più la frase sul bound |
| R18 | README.md | 485–496 | sottosezione nuova, 2 frasi e 4 punti |
| F1 | docs/findings.md | 114 | |
| F2 | docs/findings.md | 119–120 | |
| F3 | docs/findings.md | 121–122 | |
| F4 | docs/findings.md | 131–133 | |
| F5 | docs/findings.md | 187–245 | sezione nuova in fondo (riga vuota 187, titolo 188) |

Nomi dei test in `tests/test_controller.py` (condizione di R17): `test_the_deterministic_part_holds_on_every_path`
(stream `worst_case`, z = 1 sempre: il caso peggiore), `test_the_bound_holds_with_probability_one_minus_delta`
(`shifting_stream`: drift), `test_the_bound_holds_against_an_adversary_that_uses_the_history` (avversario adattivo),
`test_the_bound_fails_if_the_adversary_sees_the_audit_coin` (avversario adattivo che rompe la condizione 1, il bound
deve fallire), `test_size_of_the_terms_in_the_project_document`, `test_long_run_bound_has_all_three_terms`. La frase
proposta ("worst case stream, drifting stream, adaptive adversaries") è quindi sostenuta.

## 2. Numeri e fonti

### README

| Numero scritto | Fonte |
|---|---|
| R2: 100 splits | `B3/m2_validity.log` "rows 182400 networks 24 domains per network 76 splits 100" |
| R4: 3,000 | README, Setup, Splits ("3,000 rows to calibrate λ with UCB") |
| R5: 7,000 = 4,000 + 3,000 | README, Splits; `B3/adaptation_cost.log` "the 4,000 adaptation inputs" |
| R7: valori M1 | `M1CSV` colonna `savings`, 2 decimali. CIFAR10 joint 0.5005/0.4248/0.3781; disjoint 0.3301/0.2831/0.2710; mixed 0.4859/0.4096/0.3584; joint_sd 0.5085/0.4283/0.3741. CIFAR100 joint 0.1820/0.1362/0.1157; disjoint 0.1163/0.0903/0.0850; mixed 0.1778/0.1361/0.1145; joint_sd 0.2295/0.1726/0.1504. Stessi valori in `B3/figures/savings_figures.log`, prima tabella, colonna M1 |
| R8: rischio / quota | `M1CSV` colonne `risk` (3 decimali) e `above_eps` (percentuale intera). Es. CIFAR10 joint severity 1: 0.06346 / 0.6000 → 0.063 / 60%; CIFAR100 joint severity 3: 0.04967 / 0.4178 → 0.050 / 42%. Stessi valori in `B3/figures/savings_figures.log`, tabella "M1 (clean lambda kept ...)" |
| R9: 0.46, 0.35 | media dei 4 regimi in `M1CSV` (CIFAR10 severity 1: 1.8250/4 = 0.456; severity 5: 1.3816/4 = 0.345); `B3/intervals.log` sez. E, righe "cifar10 1 ... 0.456" e "5 ... 0.345" |
| R9: 58% to 90% | `B3/intervals.log` sez. D: "M1 cifar10 severity 1: ... cases above eps 0.584", "severity 5: ... 0.904" |
| R9: 42% | `B3/intervals.log` sez. E, ultima tabella: "cifar10 1 0.416" (colonna ok) |
| R9: 0.01 to 0.25 | `M1CSV` colonna `valid_savings_old_metric`: minimo 0.00805 (cifar10 disjoint 5), massimo 0.2508 (cifar10 mixed 1) |
| R9: 12% to 43% | `B3/intervals.log` sez. E, "share of M1 cases with risk within 2 se of eps": cifar10 0.12–0.39, cifar100 0.30–0.43; REV R02 |
| R10: 6.7%, 5.7% | `B3/m2_validity.log` "shifted cifar10 ... noncal: 0.0670", "shifted cifar100 ... noncal: 0.0565"; valore esatto da `B3/m2_validity.csv` (risk_noncal > 0.05, corruption != none): 0.06697 e 0.05654 |
| R10: 7,000 | 10,000 − 3,000 righe di calibrazione; REV R14 |
| R10: 56% to 62%, 50% to 56% | `B3/intervals.log` sez. C: "cifar10 severity 3 eps 0.05: share left 0.591 ... [0.564, 0.617]", "cifar100 ... 0.530 ... [0.501, 0.558]" |
| R10: 41%, 95% | `B3/intervals.log` sez. C, "by corruption", riga cifar10: glass_blur 0.41, brightness 0.95 |
| R11: +0.036, tre seed | `B3/intervals.log` sez. B, severity 3: "cifar100: joint_sd - joint = +0.0363", "paired differences by seed: +0.0405, +0.0294, +0.0389" |
| R11: +0.011 | `B3/intervals.log` sez. B, severity 3: "cifar10: joint_sd - joint = +0.0114, Welch standard error 0.0200" |
| R12: 61%, 71%, 85%, 88% | `B3/decomposition_orders.log` "cifar10 0.607 0.393 0.711 ...", "cifar100 0.847 0.153 0.882 ..." |
| R14: 11% to 25% | invariato; `B3/figures/savings_figures.log`, median_recovery_M8b 0.106–0.252 (max softmax) |
| R15: 2,600; 30,900 | `B3/adaptation_cost.log` "training of the heads mean 2,604 full forwards", "mean 30,860 full forwards" |
| R15: 81,000; 1.7 million | `B3/adaptation_cost.log` "payback after 81,403 inputs of the same domain (incremental)", "payback after 1,669,064" |
| R15: 4,000 | `B3/adaptation_cost.log` "the 4,000 adaptation inputs 1,208 incremental over M2" |
| R16: 0.53, 0.36 to 0.71 | `B3/intervals.log` sez. F: "cifar10 joint_sd ... share >= 1/4: 0.53 [0.36, 0.71] -> passes"; con lo score agreement tutte "fails" |
| R17: 0.156, 0.081 | `P1/controller_stress.log` riga 18 "T = 1e4: 0.156, 1e5: 0.081 ... (eps = 0.05, pi_min = 0.05)"; REV R09 |
| R18: 3,000; about 30% | `B3/adaptation_cost.log` "M2 net of its 3,000 calibration forwards over 10000 inputs ...: 0.210 (gross 0.301)" (−30.2%, CIFAR10) e "0.086 (gross 0.127)" (−32.3%, CIFAR100) |
| R18: 46% to 50%, 28% to 31%, 39%, 38% | `B3/reference_check.log` sez. B, cifar100 severity 0: exit_right_share 0.465 (disjoint), 0.488 (joint), 0.388 (joint_sd), 0.495 (mixed); final_right_share 0.311, 0.277, 0.379, 0.275. Rapporto delle medie da `B3/reference_disagreements.csv` (riproduce i 3 decimali del log): 0.4648, 0.4876, 0.3883, 0.4952 e 0.3108, 0.2770, 0.3792, 0.2753 |
| R18: 0.526, 0.403 | `B3/calibration_budget.log`, UCB eps 0.05: "cifar10 100 0.403 ...", "3000 0.526 ..." |

### docs/findings.md

| Numero scritto | Fonte |
|---|---|
| F1: 0.35 to 0.46 | `B3/intervals.log` sez. E (M1 cifar10: 0.456 ... 0.345); il rischio di M1 su CIFAR10 è sopra ε in ogni regime e severity (`M1CSV`, minimo 0.059) |
| F2 | come R12 |
| F3 | come R11 |
| F4 | come R14 e R16 |
| F5: 125 test | `P1/pytest_rs.log` riga 17 "125 passed"; REV Riassunto punto 1 |
| F5: 11 test, 136 | `outputs/fable_review/handoff/pytest_handoff.log` "136 passed"; 136 − 125 = 11 |
| F5: nove righe, 3e-9 | `P1/regen_rows.log`: 9 righe, "largest absolute difference over all regenerated rows and columns: 3.33e-09" |
| F5: 0.5 to 1.5 points | `B3/reference_check.log` sez. A, colonna 9_minus_8, cifar100: joint −0.011/−0.015/−0.011, joint_sd −0.008/−0.008/−0.005, mixed −0.009/−0.015/−0.012 (severity 0/3/5) |
| F5: 1.4 to 2.4 points (clean) | `B3/reference_check.log` sez. A, colonna ens_minus_9, cifar100 severity 0: joint 0.023, joint_sd 0.014, mixed 0.024 (disjoint 0.020) |
| F5: sistema più accurato | `B3/reference_check.log` sez. B, colonna acc_lost, cifar100: tutti negativi tranne joint_sd severity 0 (−0.000, acc_final = acc_system = 0.620) |
| F5: 46% to 50%, 28% to 31%, 39%, 38% | come R18 |
| F5: 0.231 → 0.293, 0.526 → 0.549 | `B3/reference_check.log`, "share of the clean savings left, by reference": cifar100 exit9 0.231, ens789 0.293; cifar10 0.526, 0.549 |
| F5: +0.032, +0.016 | `B3/reference_check.log` sez. D2, "gain of the ensemble reference", exits 1-6 only, severity 0: cifar100 0.0316, cifar10 0.0159 (all exits 0.0612: la metà circa è meccanica) |
| F5: 12% to 43% | come R9 |
| F5: 0.35 to 0.46, 0.065 to 0.127 | `B3/intervals.log` sez. E, colonne risk M1 (0.065 ... 0.127) e savings M1 (0.456 ... 0.345) |
| F5: 8.9%, 7.8%, 6.7%, 5.7%, 3.4%, 2.8% | `B3/m2_validity.log` righe "shifted cifar10/cifar100": eval 0.0889/0.0776, noncal 0.0670/0.0565, all 0.0336/0.0284; esatti da `B3/m2_validity.csv`: 0.08887/0.07756, 0.06697/0.05654, 0.03363/0.02844 |
| F5: 0.009 to 0.015 | `B3/m2_validity.log` sez. 3, colonna price_fst: da 0.0086 (cifar100 severity 5) a 0.0153 (cifar10 severity 1) |
| F5: 3,000; about 30% | come R18 |
| F5: 0.526, 0.403; 0.53 → 0.45 | `B3/calibration_budget.log`, UCB eps 0.05: cifar10 severity 0 n_cal 3000 0.526, n_cal 100 0.403; cifar100 left_3 0.530 (3000), 0.451 (100) |
| F5: 0.53 with 100; 38% to 47% | `B3/calibration_budget.log`, "CRC against UCB": crc cifar10 n_cal 100 savings_clean 0.530; above_eps di crc da 0.377 (cifar100, 250) a 0.474 (cifar10, 500) |
| F5: 2,600; 30,900; 81,000; 1.7 million | come R15 |
| F5: 2.2; 0.65; 4,000 | `B3/adaptation_cost.log` cifar10 severity 3: "the training is 2.16 times the incremental cost and 0.65 times the full forwards", "4,005 as full forwards"; CIFAR100: 55.82 e 7.63 volte (domina il training) |
| F5: +0.011, 0.020, −0.010, −0.004, +0.048 | `B3/intervals.log` sez. B, severity 3: "+0.0114, Welch standard error 0.0200", "paired differences by seed (seed 0, 1, 2): -0.0097, -0.0043, +0.0483" |
| F5: −0.54, 18 points | `P1/alpha_vs_savings.log` riga 28 "vs M2 sev 3 -0.54"; 18 punti da REV R04 (6 setting per 3 seed) |
| F5: +0.036, 0.005 | `B3/intervals.log` sez. B, severity 3: "+0.0363, Welch standard error 0.0049" |
| F5: 0.53, 0.36 to 0.71 | come R16 |
| F5: 61%, 71%, 85%, 88% | come R12 |
| F5: 0.59 (0.56 to 0.62), 0.53 (0.50 to 0.56) | come R10 |
| F5: 0.41, 0.95 | come R10 (solo CIFAR10) |
| F5: cinque avversari, tre adattivi | `P1/controller_stress.log` righe 2–8 (constant, after_miss, low_pi, bang_bang, burst: runs above the bound 0.000; sees_the_coin: 1.000); `P1/controller_stress.py` righe 5–9 e 30–45 |
| F5: 0.156, 0.081 | come R17 |
| F5: 0.52%, 0.12% | `P1/costs_training_export.log` riga 10 "heads with pooling = 0.52% of the network instead of 0.12%"; REV R18 |
| F5: cinque volte più fine, +0.002 to +0.003 | `P1/grid_h3.py` riga 5 "five times denser" (1001 contro 201 valori, `P1/grid_h3.log` riga 143); guadagno medio per score 0.0021–0.0033 (`P1/grid_h3.log` righe 46–51 e 112–117); miglior score per setting uguale nelle tre griglie (riga 60 e seguenti); REV R16 |

## 3. Scostamenti dal testo approvato (TRA) e dalla specifica

1. **R6 (README 357–359).** Specifica: "they come with a risk above ε". Scritto: "they come with a risk that is often
   above ε". Motivo: in 5 celle di CIFAR100 il rischio medio di M1 è sotto ε (`M1CSV`: joint 0.044 e mixed 0.047 e
   joint_sd 0.044 a severity 1, joint 0.0497 e joint_sd 0.048 a severity 3), e la seconda tabella lo mostra.
2. **R9 (README 400–401).** Specifica: "It is kept only in `m1_savings_and_risk.csv`." Scritto: "... and in the
   `valid_savings` column of `results.csv`." Motivo: `outputs/savings_under_shift/results.csv` ha ancora la colonna
   `valid_savings` (intestazione del file). Nota: il CSV esiste oggi solo in `B3/figures/`; con le opzioni di default
   `savings_figures.py` lo scriverebbe nella cartella di `results.csv` (`outputs/savings_under_shift/`), dove non c'è.
3. **R10 (README 404–408).** (a) Gli intervalli sono nella stessa parentesi di ε: "(ε = 0.05; 95% intervals: 56% to
   62% and 50% to 56%)", per evitare due parentesi di seguito. (b) Aggiunto "on CIFAR10" alla frase sulle corruzioni:
   su CIFAR100 il minimo è 0.23 (impulse noise) e il massimo 0.91 (brightness), `B3/intervals.log` sez. C. Le righe
   405–408 sono state riavvolte; la riga 376 originale superava i 120 caratteri.
4. **R13 (README 441–442).** Omessa l'ultima frase "With M8a the traffic moves from exits 1 to 4 to exits 5 to 8."
   (sez. 5).
5. **R15 (README 467–470).** Specifica: "the gain pays this back after about 81,000 ...". Scritto: "pays this back,
   with the extra cost of the 4,000 adaptation inputs, after about 81,000 ...". Motivo: 81,403 conta il training
   (2,604) più il costo incrementale delle 4,000 righe di adattamento (1,208): (2,604 + 1,208) / 0.047 ≈ 81,000. Il
   solo training si ripagherebbe dopo circa 55,000 input su CIFAR10 (mio calcolo, non nel log). Su CIFAR100 la
   differenza è piccola (30,860 + 553).
6. **R18 (README 485–496).** Testo mio, dalla sezione 1 di TRA. Le quote dei disaccordi sono 46% to 50% (non 47%) e
   portano la qualifica sui regimi; il punto sulla calibrazione riporta 0.526 e 0.403 (verificati). Nessuna frase
   in più sul conteggio in MAC (tolta: i savings del blocco 1 non sono in MAC).
7. **F5, introduzione.** Specifica: "(first versions in `outputs/fable_review/parte1/`, same numbers)". Scritto:
   "same numbers except the cost of the adaptation, now a mean over three domains". Motivo: in `P1/m8b_cost.log` e
   in REV R03 il costo è di un solo dominio (frost: 2,682 e 31,215 forward, payback 83,000 e 1,690,000).
8. **F5, exit 9 contro exit 8.** "0.8 to 1.5 points" → "0.5 to 1.5 points" (sez. 4, punto 1).
9. **F5, ensemble.** "1.4 to 2.4 points above exit 9 on CIFAR100" → "On clean CIFAR100 ... 1.4 to 2.4 points".
   Sotto shift lo scarto nei tre regimi va da 0.7 (joint_sd severity 5) a 2.3 punti.
10. **F5, accuratezza del sistema.** Aggiunto "(equal for joint_sd on clean data)": acc_lost −0.000, entrambe 0.620.
11. **F5, disaccordi.** "47% to 50%" → "46% to 50%", con "in joint, mixed and disjoint (39% and 38% in joint_sd)"
    (sez. 4, punto 2).
12. **F5, costo di M2.** "30% lower" → "about 30% lower": −30% su CIFAR10, −32% su CIFAR100.
13. **F5, controller.** "five adaptive adversaries" → "five adversaries, three of them adaptive": constant (z = 1
    sempre) e burst (z = 1 dopo metà stream) non dipendono dalla storia; after_miss, low_pi e bang_bang sì.
14. **F5, intervalli.** "It goes from 0.41 for glass blur to 0.95 for brightness." → "On CIFAR10 it goes ..." (come 3b).
15. **F5, costi e test.** Applicati come da specifica: 2,600; 30,900; 81,000; 2.2 (2.16 nel log); 0.65; frase dei test
    "All 125 tests passed ... The review added 11 tests: 136 pass."
16. **R12.** La specifica chiede di sostituire la seconda frase con "This is the rest."; fatto. TRA non dava il testo
    del secondo punto.

## 4. Discrepanze trovate

1. **Exit 9 sotto exit 8 su CIFAR100.** TRA: 0.8–1.5 punti; riassunto successivo (REV Riassunto punto 2): 0.5–1.5. Il
   log (`B3/reference_check.log` sez. A) dà 0.5–1.5 su severity 0, 3, 5 nei tre regimi (joint_sd severity 5:
   −0.005). 0.8–1.1 vale solo sul pulito, 0.8–1.5 su pulito e severity 3. Scritto 0.5–1.5.
2. **Quote dei disaccordi.** TRA: 47–50% e 28–31%; REV Riassunto punto 3: 47–50% in joint, mixed e disjoint, 39% in
   joint_sd. Il log conferma la qualifica sui regimi (joint_sd 0.388 e 0.379 → 39% e 38%), ma per disjoint la quota
   dell'exit anticipato è 0.4648 (rapporto delle medie, che riproduce lo 0.465 del log): 46%, non 47%. Il 47% viene da
   un doppio arrotondamento (0.4648 → 0.465 → 47%). Le quote dell'exit 9 sono 0.2753–0.3108 → 28–31%, corrette; il
   31% è di disjoint, che non allena tutta la rete: per i soli joint e mixed le quote sono 49–50% e 28%.
3. **Costi dell'adattamento.** TRA (da `P1/m8b_cost.log`, solo frost): 2,700, 31,000, 83,000, 0.67. `B3/adaptation_cost.log`
   (media di tre domini): 2,604, 30,860, 81,403, 0.65; 2.16 (2.2 invariato). REV R03 riporta ancora 2,682, 31,215,
   1,690,000 e 766,000 (severity 5; ora 757,354). La specifica era già allineata al nuovo log, tranne "same numbers"
   (sez. 3, punto 7).
4. **Traffico con M8a.** La frase compare solo in TRA riga 106 e in REV riga 252 (tabella di R12). REV R23 cita come
   evidenza `block3_from_csv.log` sez. G, che (come `B3/intervals.log` sez. G) contiene solo le differenze appaiate
   M8a − M2, non la distribuzione sugli exit. Nessun log in `B3/` o `P1/` la riporta.
5. **"they come with a risk above ε" (specifica R6)**: falso in 5 celle di CIFAR100 (sez. 3, punto 1).
6. **Corruzioni da 41% a 95%**: vale solo per CIFAR10 (sez. 3, punto 3b). Anche REV R06 la dà senza dataset.
7. **"kept only in `m1_savings_and_risk.csv`"**: `results.csv` ha ancora `valid_savings` (sez. 3, punto 2).
8. **"exit 9 is at most as accurate as exit 8"** (TRA, F5) vale sulle medie dei seed. Per rete, l'exit 8 è almeno
   accurato quanto l'exit 9 in 16 reti su 24 sul pulito e 17 a severity 3 e 5 (`B3/reference_check.log` sez. A), cioè
   non in tutte le 18 reti di joint, mixed e joint_sd (CIFAR10 mixed seed 0 e 2: +0.0012 e +0.0005 sul pulito). Non
   cambiato: è la lettura sulle medie, la stessa del README, blocco 2, punto "Cost on exit 9" (riga 280 dopo le modifiche).
9. **Ensemble 1.4–2.4 punti**: solo sul pulito (sez. 3, punto 9).
10. **"five adaptive adversaries"**: tre adattivi (sez. 3, punto 13).
11. **R15, "pays this back"**: 81,403 include le righe di adattamento (sez. 3, punto 5).
12. **M2 sulle 7,000 righe, CIFAR100**: il log dà 0.0565, ambiguo all'arrotondamento; dal CSV 0.05654 → 5.7%.
13. **F5, "in the order of the README"**: dopo R12 il README non nomina più un ordine; resta implicito (prima lo shift a
    λ ferma, poi il cambio di λ, come nei due punti del README). Non cambiato.
14. **git diff --stat di `docs/findings.md`** include la sezione "Preliminary checks before the next block" (31 righe),
    già presente e non committata prima di questo lavoro (HEAD ha 153 righe, il file ne aveva 184). Non è mia.
15. **README righe 256 e 350** superavano già i 120 caratteri prima delle modifiche; non toccate.

## 5. Non verificati e non applicati

### Non verificati

- **R13, ultima frase** ("With M8a the traffic moves from exits 1 to 4 to exits 5 to 8"): nessuna fonte con numeri
  (sez. 4, punto 4). Omessa.
- **R1, il codice di Kubaty** ("same nominal cap, but there early stopping decides the real length"): non è un numero;
  il codice di Kubaty non è in `third_party/` (REV R24). Scritto come da specifica, senza verifica mia.
- **F5, affermazioni senza numero prese da REV** e non ricontrollate su un log: "the Fast yet Safe reproduction is
  identical", "Every number of the README and of this file that was checked matches the outputs", "No leakage was
  found" (REV Riassunto punto 1 e sezione "Cosa ho verificato"; `P1/check_readme_numbers.log` non aperto).
- **F5, "18 points"**: viene da REV R04 (6 setting per 3 seed); il −0.54 è verificato nel log.

### Non applicati

- Nessun ID saltato del tutto. R13 applicato in parte (senza l'ultima frase).

## 6. Verifica di Fable (14 novembre, dopo l'esecutore)

Le sezioni 1–5 sono dell'esecutore Opus. Questa sezione è del modello principale, che ha scritto la specifica.

**Cosa ho controllato.** Il diff di `README.md` e le parti nuove di `docs/findings.md`, riga per riga. Poi i numeri,
riletti da me sui log (non sulle note dell'esecutore):
- le due tabelle di M1 del README, tutte le 48 celle, contro `B3/figures/m1_savings_and_risk.csv`;
- `B3/m2_validity.log`: 8,9% / 7,8%, 6,7% / 5,7%, 3,4% / 2,8%, prezzo dell'oracolo 0,0086–0,0153;
- `B3/decomposition_orders.log`: 0,607 / 0,711 e 0,847 / 0,882;
- `B3/calibration_budget.log`: 0,526 e 0,403, 0,530 e 0,451, CRC 0,530 con 100 righe e 0,377–0,474 sopra ε;
- `B3/intervals.log`: sezioni B (+0,0114 con errore 0,0200, +0,0363 con errore 0,0049, differenze per seed), C (0,591
  [0,564, 0,617], 0,530 [0,501, 0,558], per corruzione), D ed E (M1: 0,584–0,904, rischio 0,065–0,127, quota entro 2
  errori standard 0,12–0,43, 0,416 sotto ε a severity 1), F (0,53 [0,36, 0,71]);
- `B3/reference_check.log`: sezioni A (exit 9 meno exit 8 da −0,005 a −0,015 su CIFAR100; ensemble 0,014–0,024 sul
  pulito), B (quote 0,465 / 0,488 / 0,388 / 0,495 e 0,311 / 0,277 / 0,379 / 0,275), riferimenti (0,231 → 0,293,
  0,526 → 0,549), D2 (+0,0316 e +0,0159 con i soli exit 1–6);
- `B3/adaptation_cost.log`: 2.604, 30.860, 81.403, 1.669.064, 2,16 e 0,65;
- `P1/controller_stress.log`: cinque avversari sotto il bound, `sees_the_coin` sopra; 0,156 e 0,081;
- `P1/costs_training_export.log`: 0,52% contro 0,12%.

**Esito.** Tutti i numeri controllati coincidono. Gli scostamenti dell'esecutore dalla specifica (sezione 3) sono
corretti e li confermo: erano imprecisioni della specifica o del testo approvato.

**Due modifiche mie dopo l'esecutore.**
1. **R1, budget di Kubaty.** Aggiunti i tre numeri: "their paper reports 523, 1610 and 1166 epochs on average for
   disjoint, joint and mixed (ResNet-50 on TinyImageNet)". VERIFICATO da me sulla fonte primaria il 14 novembre
   (codice `scripts/regimes_resnet.py` e arXiv 2407.14320v2): quaderno, voce X.06. La voce "R1" in "Non verificati"
   è quindi chiusa.
2. **R13, traffico con M8a.** Aggiunta la frase "With M8a fewer inputs leave at exits 1 to 4 and more at exits 5 to 8,
   in all 24 networks (17% against 24% and 60% against 46% on CIFAR10, severity 3 to 5)". VERIFICATO con
   `outputs/fable_review/fase_a/m8a_traffic.py` (log accanto): su CIFAR10 la quota agli exit 1–4 passa da 0,241 a
   0,172 e agli exit 5–8 da 0,463 a 0,596; su CIFAR100 da 0,085 a 0,054 e da 0,294 a 0,361; vale in 24 reti su 24.
   Con M8a esce anche meno traffico all'exit 9 (0,296 → 0,233 su CIFAR10): i savings calano perché si svuotano gli
   exit 1–4, non perché arrivano più input in fondo.

**Resta RIPORTATO** (non ricontrollato da me): le tre affermazioni senza numero di F5 prese dalla revisione
("reproduction identical", "every number checked matches", "no leakage"), verificate nella Parte 1 dalla sessione
precedente (quaderno P1.02, P1.12, rilievo R20).

**Parti ancora da fare, dopo il job M8b:** colonna M8b e tabella della settimana 4 su tre seed, la frase "Only seed
0", il paragrafo "Code", la sezione "M8b on three seeds" in findings, le figure in `docs/figures/`.

## 6b. Parte M8b, dopo la fine del job (Fable, 14 novembre, ore 17)

Testo nuovo, non presente tra quelli approvati il 13 novembre: Davide ha scelto "applico e mostro il diff".
Fonti: `B3/figures_v2/` (scritta da `savings_figures.py --results outputs/savings_under_shift_v2/results.csv`),
`B3/m8b_three_seeds.log`, `outputs/fable_review/fase_a/check_a1.log` e `m8b_violations.log`.

| Dove | Cosa | Fonte |
|---|---|---|
| README, Setup, M8b | "Only seed 0 ..." → le feature dei seed 1 e 2 si ricalcolano da `model.pt` | `savings_under_shift.py --recompute-features` |
| README, Code | due sotto-punti: M8b sui seed 1 e 2 (230.040 + 7.200 righe, cartella originale intatta, 4 worker con MPS dall'80% al 30% in 2,5 ore), comando delle figure | `check_a1.log` punti 1–3; quaderno X.02 |
| README, tabella dei savings | colonna M8b su tre seed: cambiano 4 celle a severity 5 (CIFAR10 joint 0,33 → 0,32, mixed 0,31 → 0,30, joint_sd 0,38 → 0,36; CIFAR100 joint 0,14 → 0,15) | `figures_v2/savings_by_method.csv` |
| README, test della settimana 4 | tabella su tre seed (media sui tre seed, quote per seed, insieme, verdetto per seed) | `figures_v2/adaptation_test.csv` |
| README, "What adapting the heads gives" | 0,047 / 0,043 / 0,044, scarto 0,003; a severity 5 +0,07 (0,25 → 0,32) e +0,09 | `m8b_three_seeds.log`; `figures_v2/savings_figures.log` (0,248 → 0,322 e 0,269 → 0,358) |
| README, L2 | "97%" → "97% to 98% on every seed" | `check_a1.log`: al bordo 3,1% / 2,7% / 2,4% |
| README, criterio | 0,53 / 0,49 / 0,60, insieme 0,54; nessun altro setting su nessun seed | `figures_v2/adaptation_test.csv` |
| README, punto nuovo "M8b keeps the guarantee as M2 does" | 10,2% contro 10,7%, differenza −0,005 [−0,022, +0,013], circa 1% oltre 2 errori standard | `m8b_violations.log`, riga "all seeds" |
| findings, sezione nuova "M8b on three seeds" | gli stessi numeri, più CIFAR10 0,065 / 0,056 / 0,057, CIFAR100 0,029 / 0,031 / 0,031, agreement 0,029 / 0,026 / 0,026 | `check_a1.log` punto 4; `m8b_three_seeds.log` |
| `docs/figures/savings_by_method.png` | sostituita con quella di `figures_v2/` (curva di M8b su tre seed) | le altre tre figure sono identiche byte per byte, non toccate |

**Controllo delle tabelle.** `python outputs/fable_review/fase_a/check_readme_tables.py` rilegge dal README le tre
tabelle riscritte oggi e le confronta cella per cella con i CSV di `figures_v2/`: 0 differenze su 24 righe.

**Test.** `pytest -q`: 136 passati (`outputs/fable_review/fase_a/pytest_fase_a.log`, 14 novembre ore 17).

**Una frase lasciata com'era.** README, Code: "(the 30 W charger allows about 4)". Il piano dice al massimo 3 processi
senza MPS; non c'è una misura che lo smentisca, quindi non l'ho cambiata.

## 7. Frasi del documento di progetto online che non concordano più

Dalla copia locale `outputs/fable_review/documento_progetto.md` (10–11 novembre). Il documento online non è stato
toccato: decide Davide se e come correggerlo.

| Riga della copia | Cosa dice | Cosa risulta ora |
|---|---|---|
| 115 | training delle teste circa 26 miliardi di MAC (circa 650 forward complete) | M8b misurato: circa 2.600 forward complete su CIFAR10 e 30.900 su CIFAR100 |
| 135, 169, 320 | i savings di M1 sono "illusori"; nei confronti contano solo i "savings validi" | M1 riportato con savings lordi e rischio; i valid savings sono una metrica fragile |
| 282, 304 | il criterio della settimana 4 "passa in 1 caso su 8" | non superato: un setting al limite, 0,53 con intervallo da 0,36 a 0,71 |
| 298 | "sono illusori, perché tolti i casi che rompono il risk control non resta quasi nulla" | comprati con un rischio sopra ε: M1 rompe il risk control nel 58–90% dei casi su CIFAR10 |
| 300 | casi di M2 sopra ε "tra il 6% e il 10%" | README: 8–11% sulle righe di eval; con 100 split 6,7% e 5,7% fuori dalla calibrazione |
| 300 | circa 60% e 40% del calo | 61–71% su CIFAR10 e 85–88% su CIFAR100, secondo l'ordine |

