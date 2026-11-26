# Ricostruzione di outputs/fable_review dopo la perdita del Mac (26 novembre 2026)

Il 26 novembre Davide ha perso il Mac su cui era stato fatto il lavoro. Su GitHub c'era tutto fino al commit `fa4648a`
del 16 novembre, ma questa cartella era in `.gitignore`. I file qui sono ricostruiti dalle chiamate della sessione
Claude `session_01AArfJrkP2FLpQZigKcWbfZ` (14-16 novembre), estratte dal coordinator del progetto nella libreria del
progetto (`recovery/`).

## Come

Gli eventi della sessione sono stati rieseguiti in ordine in una cartella separata: ogni Write, ogni Edit, ogni Read
completa (usata come stato vero del file) e i 21 snippet Python lanciati da Bash che riscrivevano un file (controllati
uno per uno: leggono un file, sostituiscono stringhe e lo riscrivono). Le letture parziali successive sono servite da
controllo e coincidono tutte con lo stato ricostruito. Le figure PNG vengono dalle letture delle immagini, quindi sono i
file originali.

## Cosa c'è

- Codice completo dei pilot B1, B2 e B3 (`pilots/`), dei controlli in `pilots/verifica_fable/`, della Fase A
  (`fase_a/`), del simulatore online e dei suoi job (`fase_c/sim.py`, `run.py`, `run_c06.sh`, `guard_calibrate.py`,
  figure), della Fase D (`fase_d/`, compresi CIFAR-10.1, latency e ImageNet).
- `fase_d/imagenet/imagenet_minimo.ipynb` rigenerato con `make_notebook.py` nella sua ultima versione (secondo run,
  teste allenate su fino a 200.000 immagini di ImageNet train). Istruzioni in `fase_d/imagenet/README_colab.md`.
- Documenti: `02b_piloti.md`, `03_direzioni.md`, `03b_fase_c.md`, `03c_fase_d.md`, testi di tracciamento e righe di log
  (`03*_tracciamento_proposto.md`, `*_righe_log_documento.txt`), `fase_a/A2_scostamenti_e_fonti.md`,
  `01_tracciamento_proposto.md`, il pacchetto `handoff/` del 14 novembre (file da 00 a 08), le note degli esecutori e le
  specifiche dei pilot.
- `quaderno.md`, tranne le righe 1-195 (vedi sotto). Dalla voce P1.14 in poi è completo.
- Figure originali: `pilots/B1_bilancio/b1_levels.png` (figura 5 del PDF al professore),
  `pilots/B2_adattamento/b2_frontier.png` (figura 6), `fase_c/timeline_main_sev3.png` (figura 7),
  `fase_c/timeline_guard_return.png` (figura 8), più `b2_gain_vs_compute.png`, `phases_sev3.png`,
  `timeline_return_sev3.png` e `timeline_mixed.png`. Le figure 5-8 sono copiate anche in `docs/figures/`.
- `documento_progetto_parziale.md`: righe 27-266 della copia locale del documento di progetto.

## Cosa manca

- Le righe 1-195 di `quaderno.md` (intestazione e voci da P1.01 a P1.13): mai stampate nella sessione. I loro esiti
  sono in `docs/findings.md`, sezione "Review of blocks 1 to 3".
- `01_revisione.md`, `02_letteratura.md`, le note `parte1/` e `parte2/`, `esecuzione/` (tranne `compare_b4.py`),
  `pilots/00_controlli_preliminari/`, `handoff/MASTER_PROMPT_DI_RIPRESA.md`, `handoff/piano_fable_5_2_v2_originale.md`:
  scritti nella sessione locale dell'11-14 novembre, che non è recuperabile. Gli script di `parte1/` erano già stati
  portati nella repository in `scripts/6_block3_review/`.
- `pilots/B2_adattamento/test_b2.py`: scritto dall'esecutore e mai stampato.
- Tutti i risultati (CSV, log, `STATUS.txt`, cache, `fase_d/imagenet/out/`) e i dati: vanno rigenerati rieseguendo.

## Prima di rieseguire

- I percorsi `/Users/davide/...` degli script shell sono già stati cambiati in `/Users/ddb/...`.
- `run_b2_bn.sh`, `run_b2_heads.sh`, `fase_c/run_c06.sh` e `fase_d/run_d01.sh` ora chiamano il Python del nuovo
  ambiente (`/Users/ddb/miniforge3/envs/eesavings/bin/python`, Miniforge installato il 26 novembre) al posto di
  quello del vecchio Mac (`/opt/homebrew/Caskroom/miniforge/base/envs/eesavings/bin/python`).
- Il 15 novembre il vecchio Mac ha avuto due kernel panic con job a 2 worker, e da allora i job pesanti andavano con 1
  worker (quaderno, voci C.18 e C.19).
