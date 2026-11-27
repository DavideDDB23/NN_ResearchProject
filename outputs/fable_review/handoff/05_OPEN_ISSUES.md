# 05. Problemi, rischi e questioni aperte

Stato al 14 novembre 2026. Per ogni voce: cosa, impatto, cosa fare. I rilievi R01–R24 sono quelli di
`outputs/fable_review/01_revisione.md`.

## 1. Bug conosciuti

| ID | Cosa | Impatto | Cosa fare |
|---|---|---|---|
| BUG1 | `outputs/fable_review/esecuzione/compare_b4.py` si ferma con `TypeError: sequence item 8: expected str instance, float found` dopo il primo confronto (chiavi con NaN nelle colonne `delta` e `n_cal`) | Nessuno sui risultati: i confronti sono stati rifatti a mano il 14 novembre, tutti con differenza 0 | Correggere le chiavi (convertire i NaN) o eliminare lo script |

Nel codice del progetto (`src/`, `scripts/1–6`) non sono stati trovati bug: righe rigenerate da zero uguali a 3e-9, 136
test che passano, porting identico all'originale.

## 2. Incongruenze tra testi, codice e output

| ID | Cosa | Cosa fare |
|---|---|---|
| INC1 | Il README descrive M1 con i "valid savings", ma `docs/figures/savings_by_method.png` è già la figura nuova con i savings lordi; `docs/figures/risk_by_method.png` non è citata | Fase A2 |
| INC2 | Frasi del README e dei findings da riscrivere (R12): "the best savings without adaptation", "It needs no labels", "illusory", "joint_sd keeps the most savings", "about 60% ... 40%", "Reason.", "passes in 1 of the 8 settings", "recovers a median of 11% to 25%", "adversarial and drifting streams" | Fase A2, testi pronti in `outputs/fable_review/01_tracciamento_proposto.md` sezione 3 |
| INC3 | Il README dice che i regimi hanno "the same total budget" degli script di Kubaty: vale solo per i tetti nominali, loro usano early stopping (epoche reali 523, 1610, 1166) | Fase A2 |
| INC4 | `docs/controller_bound.md`: attribuzione incompleta (mancano riferimenti completi, la Proposizione 3 di Gibbs et al. è in aspettativa, la prova ad alta probabilità segue Kiyani et al. Thm 5.1); la costante 2/3 potrebbe essere 1/3 | Da verificare sulle fonti primarie prima di cambiare il file (le affermazioni vengono da L4, RIPORTATE) |
| INC5 | Il documento di progetto online (copia in `outputs/fable_review/documento_progetto.md`) contiene ancora le letture vecchie del blocco 3: savings "illusori", 60/40, "soddisfatto solo per CIFAR10 con self distillation", e dà le violazioni di M2 "tra il 6% e il 10%" mentre il README dice 8–11% | Righe nuove per il log (Fase A3); la correzione del documento la decide Davide |
| INC6 | Il documento di progetto stimava il training delle teste in circa 650 forward complete su CIFAR100; misurato circa 30.900 | Da riportare nella nota al professore e nella Fase C |
| INC7 | Le note `outputs/fable_review/parte2/L2_openreview_neurips.md` e `L3_direzione_A.md` hanno in testa "Stato: in corso", ma sono complete | Aggiornare la riga |
| INC8 | `HANDOFF.md` alla radice è del 5 novembre: utile per la storia e la mappa della letteratura, ma alcune parti sono superate dai blocchi 2–3 e dalla revisione | Leggerlo come storico |
| INC9 | La memoria locale di Claude Code (`~/.claude/projects/-Users-davide-Desktop-NN-ResearchProject/memory/`) descrive la divisione Fable cervello e Opus esecutore; dal 14 novembre il modello principale è Opus | Aggiornata il 14 novembre con la regola "meno agenti"; un altro account potrebbe non vederla |

## 3. Risultati dubbi o da completare

| ID | Cosa | Perché è dubbio | Cosa fare |
|---|---|---|---|
| DUB1 | M8b sopra ε sulle righe di eval: 0,121 sul seed 2 (0,107 sul seed 0) | Le righe di eval sovrastimano le violazioni (per M2: 8,9% su eval, 6,7% fuori dalla calibrazione), ma 0,12 è un po' alto | Misurare fuori dalla calibrazione quando il job finisce |
| DUB2 | Payback di M8b | Il costo è misurato su 3 domini del regime joint, il guadagno è la media su regimi e corruzioni; non contano la ricorsione di L-BFGS e la validazione di L2 (pochi punti percentuali) | Dichiararlo; nella Fase B2 misurare costo e guadagno sugli stessi domini |
| DUB3 | Il "passa" di CIFAR10 joint_sd nel criterio della settimana 4 | 0,533, 0,489, 0,600 sui tre seed: sempre al limite | Riportarlo così, senza cambiare criterio |
| DUB4 | Il ruolo del segno di α nella varianza tra seed | Correlazione −0,54 su 18 punti; spiega al più un quarto dello scarto | Trattarlo come ipotesi; nei prossimi training vincolare α |
| DUB5 | Il riferimento più forte | Testato solo l'ensemble 7–9 (anche con i soli exit 1–6 usabili) e il taglio all'exit 8; non una testa finale riallenata | Se serve, pilota dedicato |
| DUB6 | Le affermazioni bibliografiche delle note L1, L2, L4 | RIPORTATE dagli agenti, non ricontrollate fonte per fonte | Ricontrollare quelle che finiscono nei testi |
| DUB7 | "Non trovato" sulla novità | Risultati negativi di una ricerca con punti ciechi (review di ICLR 2027, 22325, ICML 2026) | Ricontrollare prima di ogni invio |
| DUB8 | M8b sui seed 1 e 2 | 5 reti su 16 ancora da calcolare | Fase A1 |

## 4. Problemi metodologici

| ID | Problema | Stato |
|---|---|---|
| MET1 | **Il riferimento è debole** (R01): nei regimi congiunti l'exit 9 è al massimo accurato quanto l'exit 8; il consistency risk conta come errori anche i casi in cui sbaglia il riferimento | Da decidere per la Fase C quale riferimento e quale rischio usare; nelle Fasi B si riporta sempre l'accuracy |
| MET2 | **Esplorazione e conferma sugli stessi dati** (R13): le scelte del blocco 3 sono state fatte sulle 15 corruzioni standard | Le 4 corruzioni extra e CIFAR-10.1 restano chiuse per la conferma a metodo congelato |
| MET3 | **M2 è ottimistico** (R07): conosce il dominio e spende 3000 forward complete non sottratte | Si riporta con il suo costo e contro n_cal |
| MET4 | **Il bound del controller è largo** (R09) e vale per la media di lungo periodo, non per il rischio in una finestra; dopo uno shift c'è un transitorio sopra ε | Nella Fase C si riporta il rischio nel tempo e nella finestra peggiore |
| MET5 | **Una BN adaptation cambia il riferimento**, quindi rompe la condizione 4 del bound | Nella Fase C: o si adattano solo teste e soglie, o il cambio si tratta come un nuovo inizio del certificato |
| MET6 | **Pseudo replicazione**: gli split condividono immagini; corruzioni e severity della stessa immagine non sono indipendenti | Gestita con bootstrap gerarchico e a cluster (`scripts/6_block3_review/intervals.py`) |
| MET7 | **Tre seed** per regime: intervalli larghi e bootstrap con 3 unità al primo livello che li sottostima | Più seed nei prossimi training |
| MET8 | **α non vincolato** confonde i confronti tra regimi (R04) | Da cambiare nei training futuri |
| MET9 | **Il pooling delle teste non è contato** nei MAC (R18) | Dichiararlo; contarlo nella latenza |
| MET10 | **I savings in MAC non sono tempo reale** (R11) | Latenza a batch 1 nella Fase D |

## 5. Debolezze dell'implementazione (rilievo R22, solo elenco, nessuna riscrittura decisa)

- `analysis.measure` calcola `valid_savings`: una convenzione di reporting nel punto più basso della pipeline.
- `savings_figures.decomposition` rifà a mano `savings.savings_change_by_exit`, che così è usata solo dai test.
- `savings_figures.curves` costruisce M8b come M2 più un guadagno appaiato: la funzione più difficile da leggere.
- Il grafico a barre in `savings_figures.main` duplica gran parte di `grid_plot`.
- `savings_under_shift.py`: `agreement_losses` passa `None` come label; il nome della rete viene da
  `run_dir.split(os.sep)[-3:]` (si rompe con una barra finale).
- `fys.py` importa in testa `read_img_cls_data`, che serve solo a `img_cls_main`.
- `savings.RESNET20_CIFAR_CUM_MMACS` è usata solo dai test.
- Le teste hanno un'unità di confidence che il blocco 3 non usa, ma che è contata nei MAC (9,1% dei MAC delle teste su
  CIFAR10).
- `environment.yml` non fissa le versioni.
- `scripts/2_fys_reproduction/reproduce_fys.py` non ha `--out-dir`.
- Gli `args.json` delle 24 reti non hanno i campi di provenienza.

## 6. Limiti sperimentali

- Una sola architettura (ResNet20) e una sola scala (CIFAR); solo shift sintetici (CIFAR-C).
- Nessuna latenza reale; nessun esperimento online (stream).
- Il training delle reti non è riproducibile al bit (CUDA per il training, MPS per l'export).
- I controlli preliminari del 10 novembre (BN adaptation, patience, soglie per exit) sono stati fatti con un protocollo
  ridotto: BN adaptation e soglie per exit vanno rifatte nei piloti B2 e B4.

## 7. Decisioni non ancora prese

| Decisione | Chi | Note |
|---|---|---|
| Confermare il piano v3 e l'ordine dei piloti (B1 e B2 per primi) | Davide | Checkpoint 2b |
| Quale riferimento e quale rischio usare nella parte online | Davide, su proposta | MET1 |
| Valori di ε e δ per CIFAR10 e CIFAR100 | Davide | Aperta nel documento di progetto |
| Trigger della fase di adattamento: soglia su λ o test sequenziale | Dopo B2 | Aperta nel documento di progetto |
| Scadenza del progetto del corso | Davide | Non nota |
| ImageNet nel progetto o lasciato alla tesi | Davide e professore | Aperta nel documento di progetto |
| Scenario standard o device e server | Davide e professore | Domanda prevista nella nota al professore |
| Come presentare la fase di adattamento al professore, visto che il criterio non è passato | Davide | Nota al professore, Fase E |
| Commit delle modifiche e come dividerle | Davide | Niente è committato |
| Tenere o togliere `.gstack/` da `.gitignore` | Davide | — |

## 8. Attività che richiedono la conferma di Davide

- Ogni job oltre un'ora in locale e ogni uso di Colab, con stima di tempo e CU (il job M8b in corso è già approvato).
- Ogni scrittura in `docs/findings.md` che non sia il testo già approvato di `01_tracciamento_proposto.md`.
- Ogni modifica al documento di progetto online.
- Ogni commit, push, branch, stash, reset.
- Il codice nuovo in `src/` o `scripts/`.
- Il lancio dei piloti (Checkpoint 2b).
- L'invio di qualunque testo al professore (lo manda Davide).

## 9. Rischi

| Rischio | Probabilità | Effetto | Mitigazione |
|---|---|---|---|
| Limiti d'uso e crediti | Alta (successo più volte l'11–14 novembre) | Lavoro interrotto, agenti che ripartono da zero | Un agente alla volta, compiti stretti, note su file, compute in shell |
| Batteria e caricatore da 30 W | Alta con più di 2 worker su MPS | Job fermati dal watchdog | 2 worker con MPS, watchdog, job riprendibili |
| Erosione della novità | Media | Gli autori della 22466 lavorano su soglie per stadio (22472); Gibbs et al. e il gruppo di B-PAC indicano estensioni vicine | Ricontrollare arXiv e OpenReview a ogni checkpoint; puntare sul bilancio computazionale e sulla fase di adattamento |
| Critiche dei revisori | Alta | Riferimento debole, solo CIFAR, MAC invece di tempo, bound largo | Fase D; riportare accuracy e riferimenti alternativi; intervalli |
| L'adattamento non ripaga | Media | Il contributo sull'adattamento diventa un risultato negativo | È accettabile se riportato bene; B2 cerca varianti economiche; il piano B è la ricalibrazione o le soglie per exit |
| Scadenza del corso sconosciuta | Media | Priorità sbagliate | Chiederla subito |
| Perdita di dati | Bassa ma grave | `outputs/` (108 GB) e `data/` non sono in git; nessun backup noto | Chiedere a Davide se esiste un backup |
| Il job staccato viene ucciso | Media | 5 reti mancanti | Il job è riprendibile (salta le reti finite) |
| Una nuova istanza su un altro account non vede memoria e trascrizioni di questo account | Media | Contesto perso | Questo pacchetto è autosufficiente |

