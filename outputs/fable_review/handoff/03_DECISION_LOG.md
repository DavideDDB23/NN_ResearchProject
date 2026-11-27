# 03. Registro delle decisioni

Le decisioni sono in ordine cronologico, dalla più vecchia. Stato: **valida**, **superata** (con quale decisione),
**da rivalutare**. Le parole di Davide sono citate alla lettera (elenco completo in `08_CONVERSATION_TIMELINE.md`).
Le decisioni prima di questa sessione vengono da README, `docs/findings.md`, documento di progetto e `HANDOFF.md`.

## 1. Prima della sessione (fino al 10 novembre)

| ID | Decisione | Motivo | Alternative scartate | Stato |
|---|---|---|---|---|
| P01 | Linea di ricerca: risk control senza label di un early exit sotto shift, con savings computazionali e una fase di adattamento | Indicazione del professore (28 ottobre) e sua risposta alla proposta | L'idea iniziale di Davide era solo il monitoraggio con audit (poi trovato nella 22466) | Valida: è la bussola |
| P02 | ResNet20 multi exit, 9 exit, pooled head di Kubaty et al. 2025 | Costano lo 0,12–1,08% della rete; exit superficiali più accurati | Teste di Pomponi et al. 2022 (una testa costa il 41% della backbone; la rete con tutte le teste 3,3 volte una ResNet20) | Valida |
| P03 | α delle pooled head non vincolato, poi fisso dopo il training | Come SDN e Kubaty; le feature in cache valgono per l'adattamento | Vincolarlo in [0, 1] | Da rivalutare per i training futuri (46 teste su 192 con α < 0, confonde i confronti tra regimi) |
| P04 | Ricetta SGD di He et al., 320 epoche per regime, ultima epoca, nessun early stopping | Ricetta standard per ResNet20; la val resta pulita per le unità di confidence | AdamW con early stopping come Kubaty | Valida |
| P05 | Regime joint_sd aggiunto (Zhang et al. 2019, τ = 3, α = 0,5) | Variante con self distillation | — | Valida (confermato che non è in Kubaty et al.) |
| P06 | Unità di confidence: regressione logistica sulle feature, loss non bilanciata | Una soglia condivisa: il bilanciamento sposterebbe gli score di ogni exit in modo diverso | Confidence allenata con la rete (Pomponi) | Valida, ma il blocco 3 usa score sulle statistiche dell'uscita |
| P07 | Consistency risk, UCB con WSR, δ = 0,1, λ condivisa, griglia [1] + 1 − logspace(−4, 0, 200) | Come Fast yet Safe; griglia fitta vicino a 1 perché la max softmax su CIFAR è spesso sopra 0,99 | CRC (garanzia solo in media) | Valida; CRC va riportato accanto (R10) |
| P08 | Split adapt 4000, cal 3000, eval 3000; 5 split; stessi indici su CIFAR-C | Righe separate per adattamento, calibrazione e misura | — | Valida |
| P09 | Savings rispetto alla ResNet20 senza teste | Confronto onesto con la rete senza early exit | Rispetto all'exit 9 | Valida |
| P10 | Solo le 15 corruzioni standard nel blocco 3; le 4 extra per la conferma | Separare esplorazione e conferma | — | Valida |
| P11 | M8b con penalità L2 verso i pesi allenati e score fittato su righe non viste dalle teste | Senza L2 le teste vanno in overfitting; lo score fittato sulle stesse righe blocca ogni uscita | Senza L2; score sulle stesse righe | Valida |
| P12 | Criterio della settimana 4 fissato prima: recuperare 1/4 del calo su almeno metà dei casi a severity ≥ 3 | Evitare criteri a posteriori | — | Valida; esito: non superato (1 setting su 8, al limite) |
| P13 | Il bound del controller è presentato come adattamento di Kiyani et al. e Gibbs et al. | Non è un risultato nuovo | Presentarlo come nuovo (errore fatto in una bozza) | Valida |
| P14 | Training su Colab (A100) | Sul Mac un'epoca costa 18 s e due run non accelerano | Mac | Valida |
| P15 | Feature di CIFAR-C salvate solo per il seed 0 | 8,3 GB per rete | Tutte (133 GB in più) | Valida; per M8b sostituita da S12 (ricalcolo al volo) |

## 2. In questa sessione (11–14 novembre)

### S01. Seguire il piano v2, Parti 1–4, con i checkpoint
- **Data.** 11 novembre, primo messaggio: "Leggi per intero documenti/piano_fable_5_2.md prima di fare qualunque cosa.
  Poi segui le Parti da 1 a 4 nell'ordine, fermandoti ai checkpoint indicati. Rispondimi in italiano."
- **Conseguenze.** Parte 1 e Parte 2 fatte, Checkpoint 1 e 2 consegnati.
- **Stato.** Valida nella forma del piano v3 (le Parti 1 e 2 sono chiuse).

### S02. Verificare con codice indipendente invece di fidarsi del codice esistente
- **Data.** 11 novembre.
- **Cosa.** Piloti propri in `outputs/fable_review/parte1/`; la misura delle righe di `results.csv` riscritta da zero.
- **Motivo.** Standard 1.2.1 e 1.2.2 del piano: un controllo che usa lo stesso codice non trova i suoi errori.
- **Stato.** Valida.

### S03. Test UCB veloce (`_Kn(ε) > 0` invece della ricerca della radice)
- **Data.** 11 novembre.
- **Motivo.** Rendere possibili 100 split per dominio. Equivalenza verificata (36 calibrazioni reali, 160 matrici non
  monotone).
- **Conseguenze.** Portato in `analysis.ucb_index_fast` il 13 novembre.
- **Stato.** Valida.

### S04. Severità di R01 "Bloccante come decisione di progetto, non per i numeri"
- **Data.** 13 novembre, dopo l'obiezione del revisore indipendente.
- **Motivo.** La conclusione "i savings si dimezzano circa" regge con ogni riferimento; il motivo forte è la
  composizione dei disaccordi, non l'aumento dei savings con l'ensemble (per metà meccanico).
- **Stato.** Valida. La decisione su quale riferimento usare online è ancora aperta (`05_OPEN_ISSUES.md`).

### S05. Un solo revisore indipendente a fine Parte 1
- **Data.** 13 novembre. Regola 7 del piano v2.
- **Cosa è successo.** Due tentativi su Fable (effort high) si sono fermati sul limite d'uso senza conclusioni; il terzo
  su Opus 5.5 (effort medium, perimetro stretto, circa 30 chiamate) ha finito in 3,5 minuti.
- **Conseguenze.** Tre obiezioni verificate e integrate in `01_revisione.md`.
- **Stato.** Valida. Lezione: compito stretto e budget di chiamate esplicito.

### S06. "Fable cervello, Opus esecutore"
- **Data.** 13 novembre. Davide: "usa fable come ricercatore e orchestratore ma per fare le cose usa opus 5.5 extra, e
  poi fable verifica il risultato, questa sessioen sta consumando troppi crediti". Poi: "Si fable deve essere il
  cervello, il modello che usi per fare ml research, che pensa alle soluzioni, che pensa a possiibli nuove scoperte
  attentamente, poi una volta definito il tutto, opus esegue".
- **Interpretazione.** "extra" letto come effort extra high (`xhigh`); Davide non l'ha corretto.
- **Conseguenze.** Memoria salvata (`delegare-esecuzione-a-opus.md`).
- **Stato.** Valida come principio quando il modello principale è Fable; precisata da S09 e S14.

### S07. "Si applica tutto"
- **Data.** 13 novembre, risposta al Checkpoint 1.
- **Cosa approva.** Tutte le correzioni di `01_revisione.md` (R01–R24) e i testi di `01_tracciamento_proposto.md`:
  sostituire i valid savings di M1, eseguire M8b sui seed 1 e 2 (job oltre un'ora, approvato), riscrivere le frasi
  del README e dei findings, aggiungere i test, portare gli script nella repository, aggiungere la sezione ai findings.
- **Stato.** Valida. Eseguita in parte: codice e test fatti; README e findings no; M8b in corso.

### S08. Esecutori in parallelo per correzioni e letteratura
- **Data.** 13 novembre, ore 09:40. Un esecutore per codice e test, poi uno per i testi; quattro agenti di letteratura
  in parallelo. Tutti Opus 5.5, effort `xhigh`.
- **Cosa è successo.** Limite d'uso delle 5 ore colpito tre volte (09:44, poi dopo le riprese delle 14:13 e delle
  19:12); ogni ripresa ripartiva da zero. Codice e test fatti quasi del tutto; testi non fatti; letteratura: 2 note su 4
  utilizzabili. Circa 2,5 milioni di token senza risultato strutturato.
- **Stato.** Superata da S09.

### S09. Meno agenti, uno alla volta
- **Data.** 14 novembre, circa ore 01:27. Davide: "usa meno agenti, che stanno consumando tutti i crediti". Supera il
  messaggio precedente "Continua usando agenti in parallelo come hai fatto finora".
- **Cosa.** Un agente alla volta, compito stretto, note scritte su file man mano; il compute in job di shell.
- **Conseguenze.** I due agenti successivi (letteratura, direzione A e OpenReview) hanno finito al primo tentativo.
- **Stato.** Valida.

### S10. Script della revisione nella repository, output in cartelle nuove
- **Data.** 13–14 novembre, sotto S07.
- **Cosa.** `scripts/6_block3_review/` (stessa logica dei piloti), output in `outputs/block3_review/`; M8b dei seed 1 e
  2 in `outputs/savings_under_shift_v2/`, mai in `outputs/savings_under_shift/`.
- **Motivo.** Regola 4 del piano (non sovrascrivere gli output esistenti); evitare 133 GB di feature con
  `--recompute-features`.
- **Stato.** Valida.

### S11. La direzione A diventa un'analisi
- **Data.** 14 novembre, Checkpoint 2 (`02_letteratura.md`).
- **Motivo.** Le sue idee teoriche sono già in letteratura (22466, 22325, Einbinder et al. 2024).
- **Alternative.** Tenerla come contributo teorico (scartata: novità insufficiente).
- **Conseguenze.** Il contributo diventa la domanda del professore: savings computazionali sotto shift e una fase di
  adattamento al netto del costo.
- **Stato.** Valida. Davide non l'ha confermata alla lettera, ma subito dopo ha chiesto di ricordare la frase del
  professore, che va nella stessa direzione.

### S12. M8b sui seed 1 e 2 con ricalcolo delle feature
- **Data.** 14 novembre, 08:42 (primo tentativo, 4 worker) e 11:38 (ripresa, 2 worker, staccata dalla sessione).
- **Motivo.** R05; 4 worker scaricavano la batteria (watchdog al 30%, exit 143 alle 11:12).
- **Stato.** Valida, in corso.

### S13. Non aggirare la protezione di OpenReview
- **Data.** 14 novembre.
- **Cosa.** Le review di ICLR 2027 e il testo della 22325 restano da leggere dal browser di Davide.
- **Stato.** Valida.

### S14. Modello principale Opus 5.5
- **Data.** 14 novembre, circa ore 11:30, dopo un limite d'uso. Davide ha eseguito `/model claude-opus-5-5`.
- **Conseguenze.** Il modello principale fa ricerca ed esecuzione direttamente; agenti solo se servono davvero.
- **Stato.** Valida per questa sessione; nella chat nuova vale il modello che Davide sceglierà (regola 7 del piano v3).

### S15. Piano v3: tolte le parti fatte, completate le fasi successive, bussola del professore
- **Data.** 14 novembre. Davide: "Puoi continuare, ricordati il piano originale dove puoi cancellare quello fatto
  finora e lasciare e completare con le modifiche che servono le fasi successive. E ricordati anche la precisazione del
  professore".
- **Cosa.** `documenti/piano_fable_5_2.md` riscritto in v3 (copia della v2 in
  `outputs/fable_review/handoff/piano_fable_5_2_v2_originale.md`): Fase A (chiusure), Fase B (piloti B1–B4), Fase C
  (parte online del documento di progetto), Fase D (requisiti per pubblicare), Fase E (paper al contrario e nota).
- **Interpretazione.** "Il piano originale" letto come il piano che governa il lavoro (`piano_fable_5_2.md`), con dentro
  le fasi successive del documento di progetto (settimane 5–7). Davide non ha ancora visto la v3.
- **Stato.** Valida, da confermare con Davide.

### S16. Ordine dei piloti: B1 e B2 per primi
- **Data.** 14 novembre, nel piano v3.
- **Motivo.** Rispondono direttamente al professore (bilancio computazionale e fase di adattamento).
- **Alternative.** L'ordine proposto nel Checkpoint 2 era A e F per primi (`02b_piloti.md`), superato.
- **Stato.** Valida, da confermare al Checkpoint 2b.

### S17. Il metro comune: savings netti e frontiera accuracy contro compute
- **Data.** 14 novembre (`02b_piloti.md`, piano v3 sezione 6.5).
- **Motivo.** Quando il riferimento cambia (BN adaptation) i savings a consistency risk fissato non bastano.
- **Stato.** Valida, proposta.

### S18. Fermarsi prima dei piloti e preparare questo pacchetto
- **Data.** 14 novembre, ore 11:35 circa. Davide: "finisci l'analisi e fermati prima di mandare i piloti, mi serve
  tutto il contesto per mandarlo a un'altra chat nuova", con il testo della richiesta di handoff.
- **Cosa.** Analisi a tre seed parziale, ripresa del job M8b, piano v3, questo pacchetto in
  `outputs/fable_review/handoff/` (la regola 3 del piano mette i report di Claude in `outputs/fable_review/`).
- **Stato.** Valida.

### S19. Nessun commit
- **Regola 1 del piano.** Tutte le modifiche sono nel working tree. Davide decide se e quando committare.
- **Stato.** Valida.

## 3. Decisioni tecniche minori della revisione

- **Griglia di λ invariata** (R16): una griglia più fitta dà +0,002–0,003 ma porta le violazioni fuori dalla
  calibrazione dal 6,9% al 9,8%.
- **UCB resta il metodo principale**, CRC riportato accanto (R10).
- **M1 con i savings lordi** e il suo rischio, i valid savings solo come vecchia metrica (R02); già nel codice.
- **Intervalli con bootstrap gerarchico** (seed, corruzione, split) e a cluster (reti, corruzioni) (R06).
- **La scomposizione del calo nei due ordini**, o la media alla Shapley (R08).
- **Test rafforzati**: UCB su righe non monotone; controller con avversari adattivi e con il caso che deve fallire.

