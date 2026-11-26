# 04. Attività, agenti, job in background e roadmap

Stato al 14 novembre 2026, ore 12. Il piano che governa il lavoro è `documenti/piano_fable_5_2.md` (v3); le fasi A–E
qui sotto sono le sue.

## 1. Il primo passo

```bash
cd /Users/davide/Desktop/NN_ResearchProject
cat outputs/savings_under_shift_v2/STATUS.txt
pgrep -fl savings_under_shift.py
```

Poi segui la tabella della sezione 4.1 da A1. Prima di lanciare i piloti (Fase B) serve l'ok di Davide (Checkpoint 2b).

## 2. Stato delle attività

### 2.1 Completate

| Attività | Risultato | Dove |
|---|---|---|
| Parte 1 del piano v2: revisione critica | 24 rilievi, approvati il 13 novembre | `outputs/fable_review/01_revisione.md`, `quaderno.md` P1.01–P1.18, `parte1/` |
| Revisore indipendente della Parte 1 | 34 numeri ricalcolati, coincidono; 3 obiezioni integrate | `parte1/revisore/rev2_*` |
| Correzioni al codice e ai test (sotto "Si applica tutto") | `analysis.py`, `savings_under_shift.py`, `savings_figures.py`, `scripts/6_block3_review/`, 11 test nuovi; 136 test passano | `01_TECHNICAL_STATE.md` sezione 6 |
| Esecuzione degli script di revisione e confronto con i piloti | Differenza 0 | `outputs/block3_review/`, `esecuzione/run_b4.times` |
| Verifica end to end del percorso M8b nuovo | `outputs/savings_under_shift/` intatto; ripresa che salta le reti fatte | `esecuzione/run_b4.times`, `smoke_b2.log` |
| Parte 2 del piano v2: letteratura | Sintesi e note | `02_letteratura.md`, `parte2/L1–L4` |
| Elenco dei piloti con ipotesi e kill | Superato nell'ordine dal piano v3 | `02b_piloti.md` |
| Analisi M8b a tre seed, parziale | 19 reti su 24 | `outputs/block3_review/m8b_three_seeds_partial.log` |
| Piano v3 | Parti fatte tolte, fasi successive completate | `documenti/piano_fable_5_2.md` |
| Questo pacchetto | 00–08 più il prompt | `outputs/fable_review/handoff/` |

### 2.2 In esecuzione

| Attività | Stato | Risultato atteso |
|---|---|---|
| M8b sui seed 1 e 2 (Fase A1) | Job di shell staccato, avviato alle 11:38:37 del 14 novembre, 2 worker, 5 reti di CIFAR100 mancanti; con un pauser della batteria (avviato alle 11:57) che lo sospende sotto il 33% e lo riprende al 45% | `outputs/savings_under_shift_v2/results.csv` con le righe M8b delle 24 reti; stima di fine tra le 16:00 e le 17:00, più le pause |

### 2.3 In coda (non iniziate)

- A2 README e `docs/findings.md`; A3 righe per il log del documento di progetto; A4 piccole correzioni.
- Fase B: piloti B1, B2, B3, B4.
- Fasi C, D, E.

### 2.4 Bloccate o in attesa di Davide

| Attività | Cosa serve |
|---|---|
| Lettura delle review di ICLR 2027 (22466, 22325, 22472) e del testo della 22325 | Davide apre le pagine di OpenReview dal browser e incolla punteggi, critiche e PDF |
| Riga `.gstack/` in `.gitignore` | Decisione di Davide |
| Commit delle modifiche | Richiesta esplicita di Davide |
| Scrittura nel documento di progetto online | Ok di Davide (o lo scrive lui) |
| Qualunque job oltre un'ora o su Colab, oltre a M8b | Ok di Davide con stima di tempo e CU |
| Lancio dei piloti | Ok di Davide al Checkpoint 2b |

## 3. Agenti e job usati nella sessione

Nessun agente è in esecuzione. Trascrizioni in
`/Users/davide/.claude/projects/-Users-davide-Desktop-NN-ResearchProject/47127c54-4d46-451d-9ed2-c2de7c18acba/subagents/workflows/<run>/`
(`journal.jsonl` e `agent-*.jsonl`); script dei workflow in
`/Users/davide/.claude/projects/-Users-davide-Desktop-NN-ResearchProject/47127c54-4d46-451d-9ed2-c2de7c18acba/workflows/scripts/`.
Questi file sono dell'account che ha lavorato finora: una sessione su un altro account potrebbe non vederli.

### 3.1 Agenti (workflow)

| Run | Ruolo e obiettivo | Modello | Agenti (tentativi) | Esito | Cosa ha lasciato |
|---|---|---|---|---|---|
| `wf_925bc165-68a` "revisore-indipendente-parte1" | Revisore 2: confutare la bozza della Parte 1, ricalcolare numeri | Fable 5.2 (modello della sessione), effort high | `a527c478c1dfde9fc`, `a2501c672302e744b` | Fallito due volte sul limite d'uso, nessuna conclusione | `parte1/revisore/rev_common.py`, `rev_reference.py`, `rev_reference_cifar10.log` (parziali) |
| `wf_c8cdde9f-e9f` "revisore-indipendente-parte1-v2" | Stesso compito, perimetro stretto (8 controlli, circa 30 chiamate) | Opus 5.5, effort medium | `ac204cac61f059692` | Completato in 3,5 minuti; 34 numeri ricalcolati | `parte1/revisore/rev2_checks.py/.log`, `rev2_mechanical.py/.log`, `rev2_m1lambda.log`; obiezioni integrate in `01_revisione.md` |
| `wf_b6b298a6-7ff` "applica-correzioni-parte1" | Esecutore del codice (B1–B5: UCB veloce, `--recompute-features`, figure, script della revisione, test), poi esecutore dei testi (A1–A5: README, findings, righe per il documento) | Opus 5.5, effort xhigh | codice: `a77e6138fd93846f4`, `ae5cd55ad7581856f`, `afcafe4ad5f58fd15`; testi: `aaa66e5dee7934705` | Entrambi falliti sul limite d'uso (13 novembre: 09:44, poi dopo le riprese delle 14:13 e 19:12). Il codice è stato fatto quasi tutto; i testi per niente; nessun report finale (`B_report.md` e `A_report.md` mai scritti) | Codice e test nel working tree; `outputs/fable_review/esecuzione/` (`smoke_b2.*`, `compare_b3.*`, `check_b3_seeds.*`, `compare_b4.py`, `run_b4.sh`, log di pytest); `outputs/block3_review/intervals*`, `figures/` |
| `wf_9756b68d-746` "letteratura-parte2" | Quattro ricerche in parallelo: arXiv recente (L1), OpenReview e NeurIPS (L2), direzione A (L3), fonti citate e controller (L4) | Opus 5.5, effort xhigh | 11 tentativi tra i quattro compiti | Solo L4 ha restituito un risultato strutturato; L1 ha lasciato le note complete; L2 e L3 nulla di usabile. Circa 1,8 milioni di token | `parte2/L1_arxiv_recente.md`, `parte2/L4_fonti_citate.md` |
| `wf_92fb6ad8-d0a` "letteratura-direzione-A" | Novità della direzione A, un agente solo | Opus 5.5, effort xhigh | 1 | Completato | `parte2/L3_direzione_A.md` |
| `wf_ac58d238-c9c` "letteratura-openreview-neurips" | UAT, 22325, review di ICLR 2027, accettati di NeurIPS 2026, un agente solo | Opus 5.5, effort xhigh | 1 | Completato in 6,4 minuti; review e 22325 bloccate da OpenReview | `parte2/L2_openreview_neurips.md` |

Ultime istruzioni date agli agenti: i prompt completi sono negli script dei workflow (cartella sopra). Le regole
comuni erano: nessun git che modifichi, `third_party/` in sola lettura, mai scrivere in `outputs/cifar/` e
`outputs/savings_under_shift/`, al massimo 3 processi, niente corruzioni extra, nessun agente annidato, note scritte su
file man mano, risultati negativi scritti come "non trovato con queste query", citazioni sotto le 15 parole.

### 3.2 Job di shell

| ID | Cosa | Esito |
|---|---|---|
| `b5txyoko8` | pytest dell'11 novembre | 125 passati |
| `b1f18ysxr` | `reproduce_fys.py --trials 5` in `parte1/fys_rerun/` | Identico all'originale |
| `bm9yvnu7r` | `parte1/m2_validity.py` (100 split) | Finito, circa 5 ore |
| `b3mvtnla1` | `parte1/sensitivity.py` | Finito |
| `bbcmca52b` | `parte1/grid_h3.py` | Finito |
| `bs0b2aivu` | `parte1/grid_fine_validity.py` | Finito |
| `beaxfgt6e` | `esecuzione/run_all.sh`: script di revisione (`run_b4.sh`), `compare_b4.py`, poi M8b sui seed 1 e 2 con 4 worker | Script finiti; `compare_b4.py` fermo su un TypeError; M8b fermato dal watchdog della batteria alle 11:12 del 14 novembre (exit 143), 11 reti su 16 |
| `bcwrlsi6c` | pytest del 14 novembre | 136 passati |
| (staccato) | `esecuzione/run_m8b_resume.sh 2` | **In corso** (sezione 2.2) |
| (staccato) | `esecuzione/battery_pauser.sh 62798` | **In corso**, finisce con il job |

## 4. Roadmap

Priorità: 1 alta, 2 media, 3 bassa.

### 4.1 Attività immediate (Fase A del piano v3)

| | Attività | Priorità | Dipende da | Stato | Risultato atteso |
|---|---|---|---|---|---|
| A0 | Leggere il pacchetto e verificare lo stato effettivo (git, `STATUS.txt`, test) | 1 | — | Da fare nella chat nuova | Fotografia dello stato per Davide |
| A1 | Finire M8b sui seed 1 e 2; `python outputs/fable_review/esecuzione/m8b_three_seeds.py`; `python scripts/5_savings_under_shift/savings_figures.py --results outputs/savings_under_shift_v2/results.csv --out-dir outputs/block3_review/figures_v2`; tasso di violazione di M8b sulle righe fuori dalla calibrazione | 1 | Il job | In corso | Guadagno di M8b e test della settimana 4 sulle 24 reti; voce X.02 del quaderno completata |
| A2 | Aggiornare README e `docs/findings.md` con le correzioni approvate | 1 | A1 (colonna M8b) | Da fare | Testi coerenti con il codice e con le figure nuove; numeri presi da `outputs/block3_review/` |
| A3 | Righe per il log del documento di progetto, nello stile dei testi per il professore, con il controllo dei caratteri vietati | 2 | A2 | Da fare | File in `outputs/fable_review/` da dare a Davide |
| A4 | Correggere `compare_b4.py` (o eliminarlo, i confronti sono già fatti); proporre a Davide di togliere `.gstack/` da `.gitignore` | 3 | — | Da fare | — |
| A5 | Review di ICLR 2027 e testo della 22325 | 2 | Davide | Bloccato | Critiche dei revisori per il test del revisore 2 |

> **Checkpoint 2b.** Fotografia dello stato a Davide, conferma del piano v3 e del suo ordine, ok per i piloti.

### 4.2 Attività pianificate (Fase B del piano v3)

| | Pilota | Priorità | Costo stimato | Risultato atteso |
|---|---|---|---|---|
| B1 | Bilancio computazionale dei savings sotto shift (oracoli per input, per exit, di soglia; prezzo statistico; netto della calibrazione) | 1 | Circa 1 ora, solo logit | La figura centrale: dove sta il margine perso |
| B2 | Fase di adattamento che ripaga (BN adaptation, M8b, BN più M8b, mappa sui logit, M8b economico, soglie per exit, M8a) | 1 | Due job da circa 45 minuti, al massimo 2 processi | La fase di adattamento da usare online, con payback e test della copia degli errori |
| B3 | Meccanismo: traffic shift contro reliability shift | 2 | Circa 30 minuti | Quale rimedio ha senso |
| B4 | Soglie per exit come adattamento minimo | 2 | Circa 45 minuti | Se 8 numeri riappresi bastano |

> **Checkpoint 3a.** `outputs/fable_review/03_direzioni.md` con esiti, matrice dei punteggi, revisore 2 e
> raccomandazione. Davide decide prima della Fase C.

### 4.3 Attività che dipendono dai risultati

| Fase | Dipende da | Contenuto |
|---|---|---|
| C. Parte online (settimane 5–7 del documento di progetto) | B2 e B4 (quale aggiornamento), decisione sul riferimento | Stream; controller con audit; tre versioni della fase di adattamento; M3–M7; S_0, S_1, S_2, frazione recuperata, payback; H5 |
| D. Requisiti per pubblicare | Ok di Davide; per ImageNet il download da parte sua | CIFAR-10.1 e corruzioni extra come conferma; latenza a batch 1; seconda architettura; ImageNet; AugMix; α vincolato e più seed |
| E. Paper al contrario, roadmap, nota per il professore | B e C | `04_paper_al_contrario.md`, `05_roadmap.md`, `06_nota_professore.md`, testo per il tracciamento |

### 4.4 Verifiche ancora necessarie

- Violazioni di M8b sui seed 1 e 2 fuori dalla calibrazione (sulle righe di eval il seed 2 è a 0,12).
- L'attribuzione del bound del controller contro le fonti primarie (le affermazioni di L4 sono RIPORTATE).
- I numeri del README dopo A2, con uno script come `outputs/fable_review/parte1/check_readme_numbers.py`.
- Le affermazioni bibliografiche delle note L1, L2, L4 usate nei testi finali: da ricontrollare sulla fonte.
- Lo stato di `origin/main` (non riverificato dal 10 novembre; un `git fetch` modifica i ref remoti, chiedere prima).

### 4.5 Checkpoint

| Checkpoint | Quando | Cosa si consegna |
|---|---|---|
| 1 | Fatto, 13 novembre | Revisione critica |
| 2 | Fatto, 14 novembre | Letteratura ed elenco dei piloti |
| 2b | Fine della Fase A | Fotografia dello stato, ok per i piloti |
| 3a | Dopo B1 e B2 | `03_direzioni.md` e decisione sulla fase di adattamento |
| 3b | Fine della Fase C | Risultati online |
| 3 | Fine della Fase E | Paper al contrario, roadmap, nota per il professore |

### 4.6 Obiettivi finali

- Consegna del progetto di Neural Networks (scadenza da chiarire con Davide).
- Un primo lavoro (workshop o TMLR), poi una pubblicazione tra ottobre e dicembre 2027.
- La tesi magistrale costruita su questo lavoro.

