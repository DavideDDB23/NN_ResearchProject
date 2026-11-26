# 07. Indice dei file importanti

Percorsi relativi alla radice del repository `/Users/davide/Desktop/NN_ResearchProject`, salvo dove sono assoluti.
Tracciato da git: **T**; non tracciato: **N**; ignorato da git (dentro `outputs/`, `data/`, `third_party/`): **I**.

## 1. Documenti di governo e di contesto

| File | Git | Contenuto |
|---|---|---|
| `documenti/piano_fable_5_2.md` | N | **Il piano che governa il lavoro, v3 del 14 novembre** |
| `outputs/fable_review/handoff/piano_fable_5_2_v2_originale.md` | I | Copia integrale della v2 (11 novembre) |
| `outputs/fable_review/documento_progetto.md` | I | Copia del documento di progetto (Claude Doc https://claude.ai/artifact/MNreFiy1XTNWcYZcUC8axi) del 10–11 novembre |
| `documenti/mail_prof_corretta.txt` | T | La mail con la proposta al professore |
| `documenti/documento_prof_31ottobre.md` | T | Il documento del 31 ottobre per il professore |
| `README.md` | T | Descrizione, ordine di lettura del codice, risultati dei blocchi 1–3 (da aggiornare, Fase A2) |
| `docs/findings.md` | T, modificato | Registro di tutte le scoperte; la sezione "Preliminary checks" del 10 novembre non è committata |
| `docs/controller_bound.md` | T | Dimostrazione del bound del controller |
| `docs/figures/*.png` | T (una N) | Figure del blocco 3; `savings_by_method.png` già nuova, `risk_by_method.png` nuova e non tracciata |
| `HANDOFF.md` | T | Handoff precedente (5 novembre): storia della ricerca, mappa della letteratura (§4), errori (§9), lacune (§10), proposta (§6, §12), estensione in finanza (§8) |

## 2. Deliverable e lavoro di questa sessione (`outputs/fable_review/`, I)

| File | Contenuto |
|---|---|
| `01_revisione.md` | Parte 1: 24 rilievi con evidenza, correzione ed effetto sui numeri; dettaglio dei principali; revisore indipendente |
| `01_tracciamento_proposto.md` | Testi approvati da applicare: sezione nuova per `docs/findings.md`, righe per il log del documento di progetto, frasi del README da riscrivere, test |
| `02_letteratura.md` | Parte 2: sintesi della letteratura, cosa è già fatto, frasi di novità, punti ciechi |
| `02b_piloti.md` | Elenco dei piloti con ipotesi, previsioni, kill e revisore 2 (ordine superato dal piano v3) |
| `quaderno.md` | Quaderno di laboratorio: P1.01–P1.18 (Parte 1), X.01–X.02 (applicazione delle correzioni e M8b a tre seed) |
| `pilots/00_controlli_preliminari/` | Controlli preliminari del 10 novembre: `probe_plan_premises.py`, `probe_bn_and_patience.py` e i loro output |
| `handoff/` | Questo pacchetto |

### 2.1 `outputs/fable_review/parte1/` (piloti della revisione)

| File | Voce del quaderno | Contenuto |
|---|---|---|
| `common.py` | — | Helper comuni: test UCB veloce, caricamento dei logit, famiglie di corruzione |
| `pytest_rs.log` | P1.01 | Test dell'11 novembre |
| `fys_rerun/` | P1.02 | Riproduzione di Fast yet Safe con 5 split in una cartella nuova (con un link a `data/`) |
| `check_fast_ucb.py` | P1.03 | Equivalenze del test UCB e copertura del bound WSR |
| `regen_rows.py`, `.log` | P1.04 | Righe di `results.csv` rigenerate con una misura riscritta da zero |
| `block3_from_csv.py`, `.log` | P1.05 | Rumore di fondo, joint_sd per seed, intervalli, violazioni, M1 contro M2, settimana 4, score |
| `m2_validity.py`, `m2_validity_analysis.py`, `.csv`, `.log` | P1.06 | Validità di M2 con 100 split per dominio; oracoli di soglia |
| `decomposition_orders.py`, `.csv`, `.log` | P1.07 | La scomposizione del calo nei due ordini |
| `reference_exit9.py`, `reference_*.csv`, `.log` | P1.08 | Il riferimento: exit 8, exit 9, ensemble; disaccordi; M2 con altri riferimenti |
| `sensitivity.py`, `.csv`, `.log` | P1.09 | n_cal, δ, ε, CRC |
| `costs_training_export.py`, `.log`, `alpha_vs_savings.log` | P1.10 | Costi, training, export, segno di α |
| `controller_stress.py`, `.log` | P1.11 | Il bound contro avversari adattivi |
| `check_readme_numbers.py`, `.log` | P1.12 | I numeri del README e dei findings ricalcolati |
| `grid_h3.py`, `.csv`, `.log` | P1.13 | Gli score con griglie di λ diverse |
| `m8b_cost.py`, `.log` | P1.14 | Compute e payback di M8b |
| `m2_equal_data.py`, `.csv`, `.log` | P1.15 | M8b contro M2 a pari dati |
| `grid_fine_validity.py`, `.log` | P1.16 | Validità con la griglia fine |
| `reference_mechanical.py`, `.log` | P1.18 | L'effetto meccanico dell'ensemble e il controllo con i soli exit 1–6 |
| `revisore/` | P1.17 | Script e log del revisore indipendente (`rev2_*`) e dei tentativi falliti (`rev_*`) |

### 2.2 `outputs/fable_review/parte2/` (note bibliografiche)

| File | Contenuto |
|---|---|
| `L1_arxiv_recente.md` | arXiv dal 29 ottobre al 13 novembre: vie di ricerca, lavori per direzione, gruppi, query senza risultati |
| `L2_openreview_neurips.md` | UAT per intero, 22325 (bloccata), review di ICLR 2027 (bloccate), NeurIPS 2026, query sulla novità |
| `L3_direzione_A.md` | Fast yet Safe, Einbinder et al., 22466, 22325, Mehra et al., C3PO, Schirmer et al. e altri |
| `L4_fonti_citate.md` | 19 riferimenti verificati; Kiyani et al. e Gibbs et al. contro il bound; codice di Kubaty et al.; TinyTTA, Kaplan, DAdEE, SAFE-KD |

### 2.3 `outputs/fable_review/esecuzione/` (applicazione delle correzioni e job)

| File | Contenuto |
|---|---|
| `run_m8b_resume.sh` | **Lo script del job M8b in corso** (riprendibile, watchdog della batteria, md5 di controllo) |
| `m8b_three_seeds.py` | Analisi indipendente di M8b su tre seed (funziona anche con risultati parziali) |
| `run_all.sh`, `run_all.log` | Job del 14 novembre: script di revisione, confronto, primo tentativo di M8b |
| `run_b4.sh`, `run_b4.times`, `run_*.out` | Esecuzione degli script di `scripts/6_block3_review/` con tempi ed esiti; check end to end |
| `smoke_b2.py`, `.log` | Prove di fumo del ricalcolo delle feature e della ripresa di M8b |
| `compare_b3.py`, `.log` | Confronto delle figure nuove con quelle vecchie |
| `check_b3_seeds.py`, `.log` | Prova del ramo a più seed di `savings_figures.py` |
| `compare_b4.py`, `.log` | Confronto degli output degli script di revisione con i piloti (si ferma con un TypeError, BUG1) |
| `pytest_before.log`, `pytest_after.log` | Test prima e dopo le modifiche dell'esecutore |

## 3. Codice

| Percorso | Git | Contenuto |
|---|---|---|
| `src/eesavings/` | T (`analysis.py` modificato) | Il package: `fys.py`, `savings.py`, `analysis.py`, `scores.py`, `adaptation.py`, `controller.py`, `models.py`, `train.py`, `export.py`, `data/cifar.py`, `data/fys_logits.py` |
| `scripts/1_data/` … `scripts/4_export/` | T | Download, riproduzione, training, export |
| `scripts/5_savings_under_shift/` | T, modificati | Blocco 3 e figure |
| `scripts/6_block3_review/` | N | Script della revisione (7 file) |
| `tests/` | T (2 modificati), `test_block3_review.py` N | 12 file, 136 test |
| `notebooks/colab_training.ipynb` | T | Training su Colab |
| `environment.yml`, `pyproject.toml` | T | Ambiente e package |

## 4. Dati e output (I)

| Percorso | Dimensione | Contenuto |
|---|---|---|
| `data/` | 9,3 GB | CIFAR10, CIFAR100, CIFAR-10-C, CIFAR-100-C, logit di Fast yet Safe |
| `third_party/RC-EENN` (`3db9a52`), `third_party/ConfidenceBranchNetwok` (`405d273`) | — | Codice originale, sola lettura |
| `outputs/cifar/` | 108 GB | 24 reti e backbone: modelli, log, logit, confidence, feature (CIFAR-C solo seed 0), `metrics.json` |
| `outputs/fys/` | — | Riproduzione del blocco 1 |
| `outputs/savings_under_shift/` | 111 MB | Risultati del blocco 3 (`results.csv`, md5 `0de4091121255c7f4a1b89920d763b2a`), da non modificare |
| `outputs/savings_under_shift_v2/` | 57 MB e in crescita | Copia dei run più M8b sui seed 1 e 2 (job in corso); `STATUS.txt` |
| `outputs/block3_review/` | 50 MB | Output degli script di revisione e analisi M8b a tre seed parziale |

## 5. Fuori dal repository

| Percorso | Contenuto |
|---|---|
| `/Users/davide/Desktop/Tesi/research_bundle/reports/` | Report di letteratura: "Verifica stato dell'arte 2 novembre.md", "Ricerca estesa 31 ottobre - idea finale.md", "Risk control per early exit.md", "Ordini limite ed early exit.md" |
| `/Users/davide/Desktop/Tesi/research_bundle/research_notes/` | Note per venue, alberi di citazione, verdetti di novità |
| `/Users/davide/Desktop/Tesi/research_bundle/submission_22466/paper.txt` | Testo completo della submission ICLR 2027 22466 |
| `/Users/davide/Desktop/Tesi/research_bundle/scripts/` | Script delle ricerche precedenti, tra cui `check_chars.py` (controllo dei caratteri vietati nei testi per il professore) |
| `/Users/davide/Desktop/Tesi/research_bundle/HANDOFF.md`, `verifiche/` | Handoff e audit precedenti |
| `/Users/davide/.claude/projects/-Users-davide-Desktop-NN-ResearchProject/memory/` | Memoria locale di Claude Code: `MEMORY.md`, `piano-fable-e-cartella-di-lavoro.md`, `delegare-esecuzione-a-opus.md` |
| `/Users/davide/.claude/projects/-Users-davide-Desktop-NN-ResearchProject/47127c54-4d46-451d-9ed2-c2de7c18acba/` | Trascrizioni di questa sessione e degli agenti (`subagents/workflows/`), script dei workflow (`workflows/scripts/`) |
| `/Users/davide/.claude/CLAUDE.md` | Istruzioni globali di Davide (skill gstack per il web, mai `mcp__claude-in-chrome__*`) |

## 6. Cosa trasferire se la nuova istanza non vede questo Mac

Indispensabili (piccoli, sotto 1 MB in tutto):
- `outputs/fable_review/handoff/` (tutto);
- `documenti/piano_fable_5_2.md`;
- `outputs/fable_review/01_revisione.md`, `01_tracciamento_proposto.md`, `02_letteratura.md`, `02b_piloti.md`,
  `quaderno.md`, `documento_progetto.md`;
- `README.md`, `docs/findings.md`, `docs/controller_bound.md`, `HANDOFF.md`;
- `outputs/fable_review/parte2/L1–L4`.

Utili: `outputs/fable_review/parte1/*.log`, `outputs/block3_review/*.log`, `outputs/fable_review/esecuzione/*.log`,
`/Users/davide/Desktop/Tesi/research_bundle/reports/` e `submission_22466/paper.txt`.

Per eseguire esperimenti servono il repository completo, `data/` (9,3 GB) e `outputs/cifar/` (108 GB): non
trasferibili con il pacchetto.

