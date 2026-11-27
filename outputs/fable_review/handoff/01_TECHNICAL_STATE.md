# 01. Stato tecnico del progetto

Stato al 14 novembre 2026, ore 12. Percorsi relativi alla radice del repository, `/Users/davide/Desktop/NN_ResearchProject`,
salvo dove indicato.

## 1. Ambiente

- **Macchina.** Mac con chip Apple M5 (MPS), macOS (Darwin 25.6.0), caricatore da 30 W, circa 619 GB liberi su disco.
- **Python.** `/opt/homebrew/Caskroom/miniforge/base/envs/eesavings/bin/python` (conda env `eesavings`, Python 3.11).
  Il package `eesavings` è installato in modalità editable (`pip install -e .`).
- **Versioni installate** (VERIFICATO l'11 novembre): torch 2.15.0, numpy 2.4.6, scipy 1.17.1, pandas 3.1.1,
  scikit-learn 1.9.1. `environment.yml` non fissa le versioni (solo `scikit-learn>=1.8`; `scores.fit_learned` usa
  argomenti recenti di sklearn).
- **Colab.** Colab Pro con A100, circa 200 CU disponibili. Il training delle 24 reti è stato fatto lì con
  `notebooks/colab_training.ipynb` (commit `de7fc3e`).
- **Test.** `$PY -m pytest -q` dalla radice. Il 14 novembre: 136 passati, 0 saltati, 101 s
  (`outputs/fable_review/handoff/pytest_handoff.log`).
- **Codice esterno** in `third_party/` (non tracciato da git, sola lettura):
  - `third_party/RC-EENN`, commit `3db9a52` (Fast yet Safe, Jazbec et al. 2024);
  - `third_party/ConfidenceBranchNetwok`, commit `405d273` (Pomponi et al. 2022).

## 2. Struttura della repository

```
NN_ResearchProject/
├── README.md                 # descrizione, ordine di lettura del codice, risultati dei blocchi 1–3
├── HANDOFF.md                # handoff precedente (5 novembre): storia della ricerca, mappa della letteratura
├── environment.yml, pyproject.toml, .gitignore
├── src/eesavings/            # il package (metodi)
├── scripts/1_data … scripts/6_block3_review/   # programmi, per fase, nell'ordine di esecuzione
├── tests/                    # 12 file di test
├── notebooks/colab_training.ipynb
├── docs/                     # findings.md (registro), controller_bound.md (dimostrazione), figures/
├── documenti/                # documento e mail per il professore, piano_fable_5_2.md (v3)
├── data/                     # 9,3 GB, non tracciato
├── third_party/              # non tracciato
└── outputs/                  # non tracciato (circa 108 GB in outputs/cifar)
```

`.gitignore` esclude `/data/`, `/outputs/`, `/third_party/`, `/checkpoints/`, i file Python compilati, gli array e i
modelli (`*.npy`, `*.pt`, …) e `.DS_Store`. Dal 13 novembre c'è anche una riga `.gstack/` (sezione 6).

## 3. Il package `src/eesavings/`

| File | Responsabilità |
|---|---|
| `fys.py` | Porting esatto di Fast yet Safe (RC-EENN `3db9a52`): regola di uscita `compute_exits` (soglia stretta), loss (`compute_risk`: prediction-consistency, gt-gap, Brier, Hellinger), bound (`ucb_wsr`, `ucb_hb`), calibrazione (`rc_main`, `rc_main_fast`, `ucb_lam`, `crc_lam`, `naive_lam`, `find_first_gap`), `losses_and_exits` per una griglia di λ. Learn then Test rimosso (nell'originale è off by one). |
| `savings.py` | Savings dagli exit, distribuzione sugli exit, scomposizione per exit; costanti `RESNET20_EXIT_MACS` e `RESNET20_FULL_MACS` (verificate contro `models.exit_costs` e i 24 `metrics.json`). |
| `analysis.py` | Split adapt/cal/eval (`make_splits`, seed 0), griglia `LAMBDAS`, `consistency_losses`, `ucb_index` (scelta UCB sequenziale), `ucb_index_fast` e `wsr_log_capital` (**aggiunti il 13 novembre**), `measure` (rischio, savings, `valid_savings`, distribuzione sugli exit). |
| `scores.py` | Score degli exit: `max_softmax`, `temperature` (per exit, fit su val), `label` e `agreement` (regressione logistica su log top prob, margine, entropia; `LogisticRegressionCV`). |
| `adaptation.py` | M8b: rifit dello strato lineare di ogni testa per distillazione (KL verso il softmax dell'exit 9) con L-BFGS full batch, penalità L2 verso i pesi allenati, L2 scelto su righe tenute da parte (`L2_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1]`, `MAX_ITER = 100`). |
| `controller.py` | `controller_step` (λ_{t+1} = max(0, λ_t + γ(B_t z_t/π_t − ε))) e `long_run_bound` (parte deterministica più Freedman). |
| `models.py` | `MultiExitResNet20`, `PooledHead`, `exit_macs`, `exit_costs` (FlopCounterMode: conv e Linear). |
| `train.py` | Loss (`joint_loss`, `self_distillation_loss`), fasi (backbone, all, all_sd, heads), regimi, loop di training. |
| `export.py` | `predict` (logit, confidence, feature), `fit_confidence`, `confidence_targets`, `split_features`. |
| `data/cifar.py` | CIFAR10/100 e CIFAR-C, split train 45.000 / val 5.000 (seed 0), `CORRUPTIONS` (15), `EXTRA_CORRUPTIONS` (4), `cifar_c_row`. |
| `data/fys_logits.py` | Lettura dei logit ImageNet rilasciati con Fast yet Safe. |

**Parametri importanti.**
- Calibrazione: `EPSILONS = [0.02, 0.05, 0.1]`, `EPS_ADAPT = 0.05`, `DELTA = 0.1` (in `savings_under_shift.py`);
  `LAMBDAS = [1] + 1 − logspace(−4, 0, 200)` (201 valori, decrescente); `N_ADAPT = 4000`, `N_CAL = 3000`;
  `N_ADAPT_HEADS = 3000` (le altre 1000 righe di adapt scelgono L2 e fittano lo score di M8b).
- Training: SGD lr 0,1, momentum 0,9, weight decay 1e-4, batch 128, lr / 10 al 50% e 75% della fase; 160 epoche per
  fase (320 per joint e joint_sd); self distillation α = 0,5, τ = 3, teacher staccato, fattore τ²; ultima epoca tenuta.
- Costi (MAC): CIFAR10 exit 1–9 `[5163776, 9885184, 14606592, 18151168, 22875392, 27599616, 31149824, 35879680, 40598912]`,
  full `40551040`; CIFAR100 `[5186816, 9931264, 14675712, 18266368, 23036672, 27806976, 31449344, 36271360, 40996352]`,
  full `40556800`. Feature delle teste: 4.352 numeri per immagine (`FEATURE_DIMS`).

## 4. Script

| Cartella | File | Cosa fa | Comando tipico |
|---|---|---|---|
| `scripts/1_data/` | `download_cifar_c.sh`, `download_fys_logits.sh` | Scaricano CIFAR-C (md5 controllato) e i logit di Fast yet Safe | `bash scripts/1_data/download_cifar_c.sh` |
| `scripts/2_fys_reproduction/` | `reproduce_fys.py` | Riproduce Fast yet Safe su ImageNet; scrive in `outputs/fys/trials<N>/` relativo alla cartella corrente (non ha `--out-dir`) | `python scripts/2_fys_reproduction/reproduce_fys.py --trials 100` |
| `scripts/3_training/` | `train_multiexit.py`, `run_trainings.py` | Training di una rete o di tutte | `python scripts/3_training/run_trainings.py --seeds 0 1 2 --jobs 3` |
| `scripts/4_export/` | `export_outputs.py`, `export_all.sh`, `summarize_networks.py` | Export di logit, confidence e feature su val, test e CIFAR-C; riassunto per rete | `caffeinate -i bash scripts/4_export/export_all.sh` |
| `scripts/5_savings_under_shift/` | `savings_under_shift.py` | Blocco 3: clean, M1, M2, M8a, M8b; una riga per (rete, score, ε, metodo, corruzione, severity, split). **Modificato il 13 novembre** (vedi sezione 6) | `python scripts/5_savings_under_shift/savings_under_shift.py --workers 4` |
| | `savings_figures.py` | Tabelle e figure del blocco 3. **Modificato il 13 novembre** | `python scripts/5_savings_under_shift/savings_figures.py --results <csv> --out-dir <dir>` |
| `scripts/6_block3_review/` (**nuovo**, non tracciato) | `review_common.py`, `intervals.py`, `m2_validity.py`, `calibration_budget.py`, `decomposition_orders.py`, `reference_check.py`, `adaptation_cost.py` | Gli script della revisione della Parte 1, portati dai piloti di `outputs/fable_review/parte1/` con la stessa logica; scrivono in `outputs/block3_review/` | `python scripts/6_block3_review/m2_validity.py --workers 3` |

**Tempi misurati** (Mac, il 14 novembre, in parallelo con altro):
- `m2_validity.py --workers 3`: 18.366 s (5,1 ore); anche il pilota originale aveva impiegato circa 5 ore;
- `calibration_budget.py --workers 3`: 5.102 s; `reference_check.py --workers 3`: 1.571 s;
  `decomposition_orders.py`: 4 s; `adaptation_cost.py`: 346 s;
- `savings_under_shift.py` completo sulle 24 reti: circa 2 ore con 8 processi (troppi per il caricatore);
- M8b con `--recompute-features`: circa 25 minuti per rete CIFAR10 e 80–90 per rete CIFAR100, per worker;
- un'epoca di training: 18 s sul Mac, circa 4 s per training su A100 con 3 job;
- export: 2–4 minuti per rete sul Mac.

## 5. Dati e output

### 5.1 `data/` (9,3 GB)

`cifar10/`, `cifar100/` (torchvision), `CIFAR-10-C/` e `CIFAR-100-C/` (19 corruzioni, ognuna `(50000, 32, 32, 3)` uint8,
`labels.npy` condiviso), `fys/` (logit ImageNet di MSDNet, L2W-DEN, DViT, DynPerc).

### 5.2 `outputs/cifar/<dataset>/<regime>/seed<k>/` (108 GB)

`model.pt`, `args.json` (senza i campi di provenienza, che sono stati aggiunti dopo), `log.csv`, `conf_label.pt`,
`metrics.json` (accuracy per exit su test e su ogni corruzione e severity; `exit_macs`, `full_macs`), `preds/`:
- `{val,test}_logits.npy` (N, 9, C) float32, `_conf.npy` (N, 9), `_features.npy` (N, 4352) float16, `_labels.npy`;
- `cifar_c/<corruzione>_logits.npy` (5, 10000, 9, C), `_conf.npy`; `_features.npy` (5, 10000, 4352) float16 **solo
  per il seed 0** (circa 8,3 GB per rete); `cifar_c/labels.npy`.

`outputs/cifar/<dataset>/backbone/seed<k>/` contiene la fase 1 condivisa da disjoint e mixed.
`outputs/cifar/summary_accuracy.csv`, `summary_auroc.csv` da `summarize_networks.py`.

### 5.3 `outputs/savings_under_shift/` (111 MB; da non modificare)

`results.csv` (230.040 righe, md5 `0de4091121255c7f4a1b89920d763b2a`), colonne: `dataset, regime, seed, score, eps,
method (clean, M1, M2, M8a, M8b), corruption, severity, split, lambda, risk, savings, valid_savings, pi_1..pi_9, l2`.
M8b solo per il seed 0 e gli split 0–2. Più `runs/` (un file per rete), `results_settings.json`, tabelle e figure
originali del blocco 3 e i log dei run.

### 5.4 `outputs/savings_under_shift_v2/` (nuovo, 14 novembre)

Copia dei file di `runs/` più le righe M8b dei seed 1 e 2. **Il job che lo completa è in corso** (sezione 7).
`STATUS.txt`, `run_m8b_seeds12.log` (primo tentativo), `run_m8b_seeds12_resume.log` (ripresa). A fine job:
`results.csv` e `results_settings.json`.

### 5.5 `outputs/block3_review/` (nuovo, 13–14 novembre)

Output di `scripts/6_block3_review/` e delle figure rigenerate:
- `intervals*.csv`, `intervals.log` (rumore di fondo, joint_sd per seed, quota rimasta con bootstrap, violazioni,
  famiglie, M1 contro M2, settimana 4, score e M8a);
- `m2_validity.csv`, `m2_validity_oracles.csv`, `m2_validity.log` (100 split per dominio);
- `calibration_budget.csv`, `calibration_budget_crc_against_ucb.csv`, `.log` (n_cal, δ, ε, CRC);
- `decomposition_orders.csv`, `.log`;
- `reference_accuracy.csv`, `reference_disagreements.csv`, `reference_alternatives.csv`, `reference_mechanical.csv`,
  `reference_control.csv`, `reference_check.log`;
- `adaptation_cost.csv`, `adaptation_cost_payback.csv`, `.log`;
- `figures/` (figure del blocco 3 dal `results.csv` originale, con M1 lordo e la figura del rischio);
- `m8b_three_seeds_week4.csv`, `m8b_three_seeds_cases.csv`, `m8b_three_seeds_partial.log` (analisi parziale a tre
  seed, 19 reti su 24, da `outputs/fable_review/esecuzione/m8b_three_seeds.py`).

### 5.6 Altri output

- `outputs/fys/trials5/`, `trials100/`: riproduzione del blocco 1.
- `outputs/fable_review/`: report, quaderno, piloti e handoff (indice in `07_FILE_INDEX.md`).

## 6. Stato di git

- **Branch** `main`, HEAD `a3eadcf` ("Complete the cleanup: Fast yet Safe in fys.py, imports and paths fixed"),
  uguale a `origin/main` al 10 novembre (non riverificato con un fetch). Nessun commit fatto in questa sessione.
- **Modifiche presenti prima della sessione:** `docs/findings.md` (sezione "Preliminary checks before the next block",
  31 righe) e la cartella `.vscode/`.
- **Modifiche della sessione, non committate:**

| File | Stato | Chi e quando | Cosa |
|---|---|---|---|
| `src/eesavings/analysis.py` | modificato | esecutore Opus, 13 nov | `wsr_log_capital`, `ucb_index_fast`; commento di M2 ("an optimistic reference ...") |
| `scripts/5_savings_under_shift/savings_under_shift.py` | modificato | esecutore Opus, 13 nov | `corruption_features` e `--recompute-features`, `--data-dir`; `--only-adaptation` salta le reti con righe M8b (salvo `--force`); scrittura atomica dei file di `runs/`; commenti |
| `scripts/5_savings_under_shift/savings_figures.py` | modificato | esecutore Opus, 13 nov | M1 con i savings lordi e figura `risk_by_method.png`; M8b e test della settimana 4 su ogni seed con righe M8b, più "all"; `--out-dir`; tabella M1 con i vecchi valid savings |
| `tests/test_savings_under_shift.py` | modificato | esecutore Opus | 3 test: UCB con righe non monotone, `ucb_index_fast` uguale a `ucb_index`, feature ricalcolate uguali alle salvate |
| `tests/test_controller.py` | modificato | esecutore Opus | 2 test: avversario che usa la storia; avversario che vede la moneta (deve fallire) |
| `tests/test_block3_review.py` | nuovo | esecutore Opus | 6 test delle funzioni pure di `scripts/6_block3_review/` |
| `scripts/6_block3_review/` | nuovo | esecutore Opus, 13 nov | 7 file (sezione 4) |
| `docs/figures/savings_by_method.png` | modificato | esecutore Opus | figura nuova con M1 lordo (il README non è ancora aggiornato) |
| `docs/figures/risk_by_method.png` | nuovo | esecutore Opus | rischio di M1 e M2 contro la severity |
| `documenti/piano_fable_5_2.md` | non tracciato | Davide (v1, v2), Opus (v3, 14 nov) | il piano, ora v3 |
| `.gitignore` | modificato | strumento gstack, 13 nov | riga `.gstack/`, non richiesta |

- **Non modificati**: `README.md`, `docs/controller_bound.md`, `HANDOFF.md`, `src/` tranne `analysis.py`, `third_party/`,
  `outputs/cifar/`, `outputs/savings_under_shift/`.

## 7. Job in background

| Job | Stato al 14 novembre, ore 12 | Come controllarlo |
|---|---|---|
| M8b sui seed 1 e 2, ripresa (`outputs/fable_review/esecuzione/run_m8b_resume.sh 2`) | **In corso**: avviato alle 11:38:37, staccato dalla sessione con `perl POSIX::setsid` e `nohup`, sotto `caffeinate -i`; PID python 62798; 2 worker; watchdog: uccide il job sotto il 30% di batteria | `cat outputs/savings_under_shift_v2/STATUS.txt`; `pgrep -fl savings_under_shift.py`; `grep -c "adaptation done" outputs/savings_under_shift_v2/run_m8b_seeds12_resume.log` (15 righe per rete, 75 in tutto) |
| Pauser della batteria (`outputs/fable_review/esecuzione/battery_pauser.sh 62798`) | **In corso**: avviato alle 11:57, staccato; sotto il 33% sospende il job (SIGSTOP), al 45% lo riprende (SIGCONT); finisce da solo quando il processo python termina | Righe `PAUSED`, `RESUMED`, `PAUSER_END` in `STATUS.txt`; `pgrep -fl battery_pauser` |

Nessun altro job è attivo. Tutti gli altri job lanciati nella sessione sono finiti (`04_TASKS_AND_ROADMAP.md`).

## 8. Verifica per elemento

| Elemento | Stato | Evidenza |
|---|---|---|
| Porting di Fast yet Safe | Implementato e verificato | `tests/test_upstream_equivalence.py`; rerun con 5 split identico (`outputs/fable_review/parte1/fys_rerun/`) |
| Bound WSR e fixed sequence testing | Implementato e verificato | identico a RC-EENN su 200 campioni; miss rate 7–8,5% sotto δ = 0,1 (`parte1/check_fast_ucb.py`) |
| Regola di uscita stretta, nessuna uscita con λ = 1 | Implementato e verificato | test e misura riscritta |
| Costi in MAC | Implementato e verificato | costanti uguali a `exit_costs` e ai 24 `metrics.json`; pooling non contato (0,52% invece di 0,12% su CIFAR10) |
| Training ed export | Implementato e verificato | backbone di disjoint identico a quello salvato; nuova forward su 50.000 immagini con 0 predizioni diverse |
| Pipeline del blocco 3 (`results.csv`) | Implementato e verificato | 9 righe rigenerate da zero, scarto 3e-9 (`parte1/regen_rows.log`) |
| `ucb_index_fast` | Implementato e verificato | stessa scelta su 36 calibrazioni reali e 160 matrici non monotone; test |
| `--recompute-features` e ripresa di M8b | Implementato e verificato | `esecuzione/smoke_b2.log` (feature identiche, righe del seed 0 riprodotte); `esecuzione/run_b4.times` (e2e: `outputs/savings_under_shift unchanged: yes`; rerun saltato in 2 s) |
| `savings_figures.py` modificato | Implementato e verificato | `esecuzione/compare_b3.log` (M2, M8b, score, scomposizione identici; seed 0 identico); ramo a più seed identico a `m8b_three_seeds.py` (verifica del 14 novembre) |
| `scripts/6_block3_review/` | Implementato e verificato | CSV identici ai piloti (differenza 0 su `m2_validity`, `calibration_budget`, `decomposition_orders`, `reference_*`); `adaptation_cost` sul dominio frost identico al pilota |
| Test nuovi (11) | Implementato e verificato | 136 passati |
| `compare_b4.py` | Da correggere | TypeError sulle chiavi con NaN; confronto rifatto a mano |
| M8b sui seed 1 e 2 | In corso | 19 reti su 24 con righe M8b al momento della stesura |
| README e `docs/findings.md` | Da completare | non aggiornati; testi pronti in `outputs/fable_review/01_tracciamento_proposto.md` |
| Righe per il log del documento di progetto | Da fare | bozza in `01_tracciamento_proposto.md`, sezione 2, da riscrivere nello stile per il professore |
| Piloti B1–B4, parte online, latenza, CIFAR-10.1 | Pianificato, non implementato | `documenti/piano_fable_5_2.md` v3 |

## 9. Problemi tecnici noti

- **Batteria.** Con 4 worker e MPS il caricatore da 30 W non tiene (dall'80% a sotto il 30% in 2,5 ore). Usare 2 worker
  con MPS, 3 senza.
- **Tempi lunghi.** `m2_validity.py` impiega circa 5 ore.
- **MPS** non è deterministico al bit in generale, ma la forward dell'export è risultata identica.
- **`reproduce_fys.py`** scrive in `outputs/fys/...` relativo alla cartella corrente: per non sovrascrivere va lanciato
  da un'altra cartella con un link a `data/` (come in `outputs/fable_review/parte1/fys_rerun/`).
- **Warning innocui**: `divide by zero encountered in log` in `ucb_wsr` (uguale nell'originale); `leaked semaphore`
  alla fine dei pool; `The given NumPy array is not writable` in `to_tensor_batch`.
- **`args.json`** delle 24 reti non ha i campi di provenienza (commit, torch, GPU).

