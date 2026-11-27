# 00. Master handoff: savings computazionali degli early exit sotto shift

Pacchetto preparato il 14 novembre 2026 (circa ore 12) per passare il lavoro a una nuova istanza di Claude, che non ha
accesso alla conversazione originale. Questo documento è autosufficiente: gli altri file del pacchetto approfondiscono
singole parti.

Chi l'ha scritto: Claude nella sessione di Claude Code `47127c54-4d46-451d-9ed2-c2de7c18acba`, nella cartella
`/Users/davide/Desktop/NN_ResearchProject`. La sessione è partita l'11 novembre con il modello Fable 5.2
(`claude-fable-5-2`). Il 14 novembre, dopo un limite d'uso, Davide ha passato la sessione al modello Opus 5.5
(`claude-opus-5-5`), che ha scritto questo pacchetto.

Convenzione usata in tutto il pacchetto:
- **VERIFICATO**: rieseguito o letto negli output o nel codice da chi scrive;
- **RIPORTATO**: prodotto da un agente e controllato solo in parte;
- **DEDOTTO**: ragionamento;
- **IPOTESI**: da testare.

---

## 1. Il primo passo operativo

Prima di tutto, leggi questo documento per intero e poi `06_NEW_AGENT_INSTRUCTIONS.md`. Poi, dalla radice del repository:

```bash
cat outputs/savings_under_shift_v2/STATUS.txt
```

C'è un job in background (M8b sui seed 1 e 2, sezione 6.3) partito alle 11:38 del 14 novembre.
- Se l'ultima riga è `END exit=0 ... original_results_unchanged=yes`, il job è finito: esegui
  `python outputs/fable_review/esecuzione/m8b_three_seeds.py` e aggiorna l'analisi a tre seed (Fase A1 del piano).
- Se l'ultima riga è `STOPPED` o `END` con un exit diverso da 0, rilancia con
  `nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' caffeinate -i zsh outputs/fable_review/esecuzione/run_m8b_resume.sh 2 > /dev/null 2>&1 &`
  (salta da solo le reti finite).
- Se l'ultima riga è `PAUSED`, il job è sospeso dal pauser della batteria (sotto il 33%) e riparte da solo al 45%:
  basta tenere il Mac in carica e aspettare.
- Se l'ultima riga è `START`, `python_pid=...`, `PAUSER_START` o `RESUMED`, controlla con
  `pgrep -fl savings_under_shift.py`: se il processo c'è, il job è in corso; se non c'è, è stato ucciso senza scrivere
  lo stato (per esempio con lo spegnimento del Mac), e va rilanciato con il comando sopra.
- I progressi si contano con `grep -c "adaptation done" outputs/savings_under_shift_v2/run_m8b_seeds12_resume.log`
  (15 righe per rete, 75 in tutto).

Poi presenta a Davide una fotografia dello stato e **non lanciare i piloti** senza il suo ok (sezione 6.5).

---

## 2. Il progetto

- **Chi.** Davide De Blasio, studente di magistrale. Progetto del corso di Neural Networks del prof. Simone Scardapane
  (Sapienza), da estendere alla tesi magistrale.
- **Titolo.** "Compute savings degli early exit sotto shift".
- **L'idea.** Una early exit network con risk control senza label alla Fast yet Safe (Jazbec et al., NeurIPS 2024): una
  soglia condivisa λ sulla confidence degli exit, calibrata con un bound UCB in modo che il consistency risk (il
  disaccordo tra l'exit usato e l'exit finale) resti sotto ε con probabilità 1 − δ. Sotto distribution shift la
  garanzia si rompe, oppure va ricalibrata e i compute savings calano. Il progetto misura quanto calano, perché, e
  quanto recupera una fase di adattamento, al netto del suo costo.
- **Obiettivi.**
  - Consegna del progetto di Neural Networks (scadenza non nota: da chiedere a Davide).
  - Una pubblicazione tra ottobre e dicembre 2027 (`HANDOFF.md` alla radice); un workshop prima sarebbe un vantaggio.
  - Una tesi magistrale da 110 e lode. Estensioni possibili: finanza (`HANDOFF.md` §8), device e server (Pomponi et al.
    2024), soglie per exit online.
- **Il documento di progetto** concordato con il professore è un Claude Doc: "Compute savings degli early exit sotto
  shift", https://claude.ai/artifact/MNreFiy1XTNWcYZcUC8axi. Copia Markdown del 10–11 novembre in
  `outputs/fable_review/documento_progetto.md`. Contiene le domande RQ1–RQ3, le ipotesi H1–H5, il bilancio del compute
  (S_0, S_1, S_2, payback H_p), il sistema online, il bound del controller, i metodi M0–M8 e il piano di 10 settimane.

## 3. La bussola: la frase del professore

Il professore ha indicato la linea del risk control per early exit il 28 ottobre 2026. Dopo la proposta di inizio
novembre (`documenti/mail_prof_corretta.txt`) ha risposto, testualmente:

> "Come direzione va benissimo. Possiamo studiare soprattutto come cambiano i savings a livello computazionale sotto
> shift (considerando che devi anche aggiungere una fase di adattamento)."

Il 14 novembre Davide ha chiesto esplicitamente di ricordarla. Ogni fase deve rispondere a "come cambiano i savings
computazionali sotto shift" e includere o misurare una fase di adattamento, con il suo compute contato.

## 4. Il setting tecnico in breve

Dettagli in `01_TECHNICAL_STATE.md`.
- **Rete.** ResNet20 multi exit (backbone di He et al. 2016 con shortcut di tipo A, copiato da Pomponi et al. 2022) su
  CIFAR10 e CIFAR100, 9 exit. Exit 1–8: pooled head di Kubaty et al. 2025 (α·maxpool + (1−α)·avgpool a 4x4, poi
  Linear, più un'unità di confidence lineare con sigmoide); exit 9: GAP e Linear.
- **Regola di uscita.** Si esce al primo exit con score > λ in senso stretto; con λ = 1 nessun input esce prima.
- **Calibrazione.** UCB con il bound di Waudby-Smith e Ramdas (WSR), δ = 0,1, fixed sequence testing su
  λ ∈ [1] + 1 − logspace(−4, 0, 200), dall'alto verso il basso, fermandosi al primo fallimento.
- **Savings.** 1 − E[c_exit]/c_M0, con c_M0 = ResNet20 senza teste (40.551.040 MAC su CIFAR10, 40.556.800 su
  CIFAR100). Costo dell'exit l = blocchi 1..l più teste 1..l. I savings possono essere negativi.
- **Regimi** (Kubaty et al. 2025, 320 epoche, SGD di He et al.): joint, disjoint, mixed, joint_sd (joint con self
  distillation verso l'exit 9, α = 0,5, τ = 3; aggiunta nostra). 3 seed, 24 reti.
- **Split** del test (10.000 immagini): adapt 4000, cal 3000, eval 3000; 5 split casuali (3 per M8b); stessi indici
  su CIFAR-C (riga = (s − 1)·10000 + i). 15 corruzioni standard per l'analisi, 4 extra tenute per la conferma.
- **Metodi.** M1 λ pulita tenuta fissa; M2 λ ricalibrata sul dominio shiftato; M8a solo lo score agreement rifittato;
  M8b teste distillate verso l'exit 9 (con penalità L2 verso i pesi allenati), poi M2.
- **Score.** max softmax, temperature per exit, P(correct), P(agreement) (regressioni logistiche su 3 statistiche
  dell'uscita di ogni testa).

## 5. Storia del lavoro

| Data | Cosa | Dove |
|---|---|---|
| fine ottobre 2026 | Ricerca bibliografica e proposta; il professore indica la linea (28 ottobre) e risponde con la frase della sezione 3 | `HANDOFF.md`, `documenti/`, `/Users/davide/Desktop/Tesi/research_bundle/` |
| 6 novembre | Blocco 1: porting e riproduzione di Fast yet Safe sui logit ImageNet | README "Block 1", commit `ba9360d` |
| 6–7 novembre | Blocco 2: 24 reti allenate su Colab (A100), export di logit e feature | README "Block 2", commit `de7fc3e`, `6f90613` |
| 7–9 novembre | Blocco 3: savings a rischio fissato sotto shift (M1, M2, M8a, M8b); bound del controller | README "Block 3", `docs/controller_bound.md`, commit `32b1b79` |
| 9–10 novembre | Riorganizzazione della repository | commit `b643ab7`, `a3eadcf` (HEAD) |
| 10 novembre | Controlli preliminari e piano v1, poi v2 (11 novembre) | `outputs/fable_review/pilots/00_controlli_preliminari/`, `documenti/piano_fable_5_2.md` |
| 11–13 novembre | Parte 1 del piano: revisione critica, 24 rilievi | `outputs/fable_review/01_revisione.md`, `quaderno.md` P1.01–P1.18 |
| 13 novembre | Checkpoint 1 consegnato; Davide: "Si applica tutto" e la divisione Fable cervello, Opus esecutore | sezione 8 |
| 13–14 novembre | Esecutori Opus applicano codice e test (finiti in parte, tra limiti d'uso); ricerca bibliografica | `04_TASKS_AND_ROADMAP.md` |
| 14 novembre, notte | Davide: "usa meno agenti"; un agente alla volta | sezione 8 |
| 14 novembre, mattina | Parte 2 (letteratura) ed elenco dei piloti: Checkpoint 2 consegnato | `02_letteratura.md`, `02b_piloti.md` |
| 14 novembre | Davide: "Puoi continuare... ricordati il piano originale... e la precisazione del professore"; limite d'uso; modello passato a Opus 5.5; "continua" | `08_CONVERSATION_TIMELINE.md` |
| 14 novembre, ore 11–12 | Davide: "finisci l'analisi e fermati prima di mandare i piloti", e la richiesta di questo pacchetto. Fatto: M8b ripreso, analisi a tre seed parziale, piano v3, questo pacchetto | questo pacchetto |

## 6. Fotografia dello stato (14 novembre, ore 12)

### 6.1 Fatto e verificato

- **Blocchi 1–3** e i loro output (`outputs/savings_under_shift/results.csv`, 230.040 righe). Righe rigenerate da zero
  con una misura riscritta: uguali a 3e-9. Riproduzione di Fast yet Safe con 5 split: identica.
- **Parte 1, revisione critica.** 24 rilievi in `outputs/fable_review/01_revisione.md`; esperimenti in
  `outputs/fable_review/parte1/`. Un revisore indipendente (Opus) ha ricalcolato 34 numeri con codice suo: coincidono.
  Le sue tre obiezioni sono state verificate e integrate.
- **Correzioni al codice approvate il 13 novembre** (non committate):
  - `src/eesavings/analysis.py`: `ucb_index_fast` (stessa scelta di `ucb_index`, 5 volte più veloce) e commento di M2;
  - `scripts/5_savings_under_shift/savings_under_shift.py`: `--recompute-features` (feature di CIFAR-C ricalcolate da
    `model.pt` per i seed senza feature salvate), salto delle reti già fatte, scrittura atomica;
  - `scripts/5_savings_under_shift/savings_figures.py`: M1 con i savings lordi e una figura del rischio, M8b su più
    seed, `--out-dir`;
  - `scripts/6_block3_review/` (7 file): gli script della revisione portati nella repository;
  - test: 136 passano (erano 125), nessuno saltato (`outputs/fable_review/handoff/pytest_handoff.log`).
  - Verifiche: feature ricalcolate identiche a quelle salvate (0 differenze su 217.600.000 valori); righe M8b del seed
    0 riprodotte esattamente; `outputs/savings_under_shift/` mai modificato (md5 `0de4091121255c7f4a1b89920d763b2a`);
    gli output di `scripts/6_block3_review/` coincidono con quelli dei piloti (differenza 0); il ramo a più seed di
    `savings_figures.py` coincide con uno script indipendente.
- **Parte 2, letteratura.** `outputs/fable_review/02_letteratura.md`, note in `outputs/fable_review/parte2/`. Ricerca
  eseguita da agenti Opus; il punto più delicato (cosa dice la submission 22466 sul riferimento) verificato sul testo
  completo in locale.
- **Elenco dei piloti** con ipotesi e kill: `outputs/fable_review/02b_piloti.md`.
- **Piano aggiornato v3**: `documenti/piano_fable_5_2.md` (la v2 è in `outputs/fable_review/handoff/piano_fable_5_2_v2_originale.md`).

### 6.2 Implementato ma verificato solo in parte

- Le note bibliografiche L1, L2, L4 (`outputs/fable_review/parte2/`): RIPORTATE dagli agenti; lette, non ricontrollate
  fonte per fonte.
- Le righe M8b dei seed 1 e 2: prodotte dal codice verificato sul seed 0; i numeri dei seed 1 e 2 non sono stati
  ricalcolati con un metodo diverso (solo aggregati in modo indipendente).

### 6.3 In esecuzione

- **M8b sui seed 1 e 2** (rilievo R05, approvato). Job staccato dalla sessione:
  `outputs/fable_review/esecuzione/run_m8b_resume.sh`, 2 worker, avviato il 14 novembre alle 11:38:37, PID python
  62798. Stato in `outputs/savings_under_shift_v2/STATUS.txt`, log in
  `outputs/savings_under_shift_v2/run_m8b_seeds12_resume.log`.
  - Fatte prima della ripresa: 19 reti su 24 con righe M8b (le 8 del seed 0 originali, le 8 di CIFAR10 dei seed 1 e 2,
    e CIFAR100 disjoint seed 1, joint seed 1 e 2).
  - Mancano 5 reti: CIFAR100 disjoint seed 2, joint_sd seed 1 e 2, mixed seed 1 e 2. Misurato: 3 corruzioni su 15 in
    circa 18 minuti, quindi circa 90 minuti per rete e worker. Stima di fine tra le 16:00 e le 17:00 del 14 novembre,
    più le eventuali pause per la batteria.
  - Il primo tentativo (4 worker) è stato fermato dal watchdog della batteria alle 11:12 (exit 143). Il watchdog di
    `run_m8b_resume.sh` uccide il job sotto il 30%, perdendo le reti in corso.
  - Per evitarlo gira anche `outputs/fable_review/esecuzione/battery_pauser.sh` (avviato alle 11:57, staccato): sotto
    il 33% sospende il job con SIGSTOP, al 45% lo riprende con SIGCONT, e scrive `PAUSED` e `RESUMED` in `STATUS.txt`.
    Con 2 worker la batteria scendeva di circa 2 punti in 20 minuti (39% alle 11:38, 37% alle 11:56).
  - A fine job lo script scrive `outputs/savings_under_shift_v2/results.csv` (vecchie righe più le nuove M8b).

### 6.4 Da fare (pianificato, non implementato)

- **Fase A del piano v3**: completare l'analisi a tre seed; aggiornare README e `docs/findings.md` con le correzioni
  approvate (testi già scritti in `outputs/fable_review/01_tracciamento_proposto.md`); righe per il log del documento
  di progetto nello stile dei testi per il professore.
- **Fase B**: piloti B1 (bilancio computazionale), B2 (fase di adattamento che ripaga), B3 (meccanismo), B4 (soglie
  per exit). Nessuno è stato lanciato.
- **Fase C**: la parte online (stream, controller con audit, fase di adattamento online, M3–M7).
- **Fasi D ed E**: requisiti per pubblicare; paper al contrario, roadmap, nota per il professore.

### 6.5 Il punto esatto di interruzione

Checkpoint 2 consegnato. Davide ha detto di continuare, poi ha chiesto di fermarsi prima dei piloti per passare il
lavoro a una chat nuova. Quindi: Fase A in corso (A1 in esecuzione, A2–A4 da fare), Fase B non iniziata. Il prossimo
checkpoint è il **2b** del piano v3: fine della Fase A, fotografia a Davide e suo ok per i piloti.

### 6.6 Da correggere

- Il README descrive ancora M1 con i "valid savings", ma `docs/figures/savings_by_method.png` è già la figura nuova
  (M1 con i savings lordi). Si sistema con la Fase A2.
- `outputs/fable_review/esecuzione/compare_b4.py` si ferma con un TypeError dopo il primo confronto. Il confronto è
  stato rifatto a mano: tutto identico.
- `.gitignore` ha una riga `.gstack/` aggiunta dallo strumento di navigazione gstack: da decidere con Davide.
- Le note L2 e L3 hanno in testa "Stato: in corso", ma sono complete.

Elenco completo in `05_OPEN_ISSUES.md`.

## 7. Risultati chiave, con le fonti

ε = 0,05, δ = 0,1, max softmax, medie su 15 corruzioni e 3 seed, se non detto altrimenti.

**Il calo dei savings (RQ1).**
- Quota dei savings puliti che resta con M2 a severity 3: 0,591 [0,564, 0,617] su CIFAR10 e 0,530 [0,501, 0,558] su
  CIFAR100 (bootstrap gerarchico). A severity 5: 0,417 e 0,394. Fonte: `outputs/block3_review/intervals.log`.
- Tra le corruzioni da 0,41 (glass blur) a 0,95 (brightness); per famiglia, rumore 0,44 e 0,25, blur 0,53 e 0,55,
  digitale 0,64 e 0,60, meteo 0,72 e 0,65 (CIFAR10 e CIFAR100).
- Savings di M2 (pulito → severity 3 → severity 5), CIFAR10: joint 0,58 → 0,35 → 0,25; disjoint 0,38 → 0,20 → 0,14;
  mixed 0,56 → 0,33 → 0,22; joint_sd 0,58 → 0,36 → 0,27. CIFAR100: joint 0,24 → 0,13 → 0,10; disjoint 0,15 → 0,07 →
  0,05; mixed 0,24 → 0,12 → 0,09; joint_sd 0,29 → 0,17 → 0,13. Fonte: README, tabella del blocco 3.

**La rottura con la λ fissa (H2).** M1 sopra ε nel 58% (severity 1) e nel 90% (severity 5) dei casi su CIFAR10, con
rischio da 0,065 a 0,127; su CIFAR100 dal 29% al 55%, rischio da 0,048 a 0,070. M1 tiene savings lordi alti (0,46 e
0,35 su CIFAR10) comprati con un rischio sopra ε.

**M2 valido ma ottimistico.**
- Con 100 split per dominio la λ di M2 supera ε nel 6,7% e 5,7% dei casi sulle 7000 righe fuori dalla calibrazione
  (8,9% e 7,8% sulle 3000 di eval, 3,4% e 2,8% sulla popolazione). Nessun dominio su 1824 sopra 0,16.
- L'oracolo di soglia dà solo 0,009–0,015 in più: il prezzo statistico della ricalibrazione è piccolo.
- Netto delle 3000 forward di calibrazione su 10.000 input (seed 0, severity 3): 0,301 → 0,210 (CIFAR10), 0,127 →
  0,086 (CIFAR100).
- Con meno righe: savings puliti su CIFAR10 0,526 (3000), 0,460 (250), 0,403 (100); CRC 0,53 anche con 100 righe ma
  con il 38–47% delle calibrazioni sopra ε.

**Dove si perdono i savings (H1).** Su CIFAR10 nei regimi congiunti 0,24–0,26 del calo di circa 0,28 viene dagli exit
1–3. Lo shift a λ ferma vale il 61–71% del calo su CIFAR10 e l'85–88% su CIFAR100, secondo l'ordine della scomposizione.

**Il riferimento.** Nei regimi congiunti l'exit 9 è al massimo accurato quanto l'exit 8 (su CIFAR100 da 0,5 a 1,5 punti
sotto). Alla λ di M2 su CIFAR100 pulito, nei disaccordi ha ragione l'exit anticipato nel 47–50% dei casi e l'exit 9 nel
28–31% (regimi joint, mixed e disjoint; in joint_sd 39% e 38%); il sistema con early exit è più accurato dell'exit 9. Con un ensemble degli exit 7–9 come riferimento: savings
puliti su CIFAR100 da 0,231 a 0,293, di cui circa metà per un effetto meccanico (+0,032 se possono uscire solo gli exit
1–6).

**L'adattamento delle teste (M8b, H4, RQ3).**
- Guadagno lordo su M2 a severity ≥ 3 (max softmax): sui 5 setting con tre seed 0,058, 0,053, 0,053 (seed 0, 1, 2);
  scarto tra seed 0,005. Fonte: `outputs/block3_review/m8b_three_seeds_partial.log` (19 reti su 24).
- Criterio della settimana 4 (recuperare 1/4 del calo in almeno metà dei casi con severity ≥ 3), fissato prima: passa
  solo CIFAR10 joint_sd, al limite su ogni seed (0,533, 0,489, 0,600; in comune 0,54). Con lo score agreement non passa
  nessun setting.
- Compute del training delle teste: circa 2.600 forward complete su CIFAR10 e 30.900 su CIFAR100 (il documento di
  progetto stimava circa 650 su CIFAR100). Payback a severity 3: circa 80.000 input su CIFAR10 e 1,7 milioni su
  CIFAR100. Fonte: `outputs/block3_review/adaptation_cost.log`.
- M8a (solo lo score agreement rifittato) sotto M2 in 24 reti su 24 (−0,016 e −0,012).

**Gli score (H3).** Differenze tra i quattro score al massimo 0,04, indipendenti dalla griglia di λ. L'agreement aiuta
solo per joint_sd.

**I regimi.** joint_sd tiene più savings di joint su CIFAR100 (+0,036, errore standard 0,005, 3 seed su 3); su CIFAR10
la differenza (+0,011, errore 0,020) viene da un seed.

**Il bound del controller.** Corretto, regge contro 5 avversari adattivi, fallisce come deve se l'avversario vede la
moneta dell'audit. È largo: 0,207 (T = 5000), 0,156 (1e4), 0,081 (1e5), 0,059 (1e6) con ε = 0,05, π = 0,05, γ = 0,01,
δ = 0,1.

**La letteratura.** Già fatto da altri: il disaccordo con il riferimento non misura la qualità (22466); senza label il
disaccordo è il massimo identificabile (22325); accuracy gap con poche label in un early exit (Einbinder et al. 2024).
Non trovato al 14 novembre 2026: la misura dei savings computazionali sotto shift a garanzia mantenuta; il bilancio in
compute di una fase di adattamento dentro un risk control.

## 8. Le decisioni principali

Registro completo in `03_DECISION_LOG.md`. Le più importanti, dalla più recente:
1. **14 novembre: fermarsi prima dei piloti e preparare questo pacchetto** (Davide).
2. **14 novembre: piano v3** con le parti fatte tolte e le fasi successive completate, attorno alla frase del
   professore (Davide l'ha chiesto; scritto da Opus).
3. **14 novembre: modello principale Opus 5.5** (Davide, dopo un limite d'uso).
4. **14 novembre: meno agenti, uno alla volta** (Davide: "usa meno agenti, che stanno consumando tutti i crediti").
   Supera la richiesta precedente "continua usando agenti in parallelo".
5. **14 novembre: la direzione A diventa un'analisi**, non un contributo (dalla letteratura). Proposto nel Checkpoint 2;
   Davide non l'ha contestato e ha poi chiesto di concentrarsi sulla frase del professore.
6. **13 novembre: "Si applica tutto"** (Davide): tutte le correzioni della revisione approvate, compreso il job M8b sui
   seed 1 e 2 e la sostituzione dei valid savings.
7. **13 novembre: Fable cervello, Opus esecutore** (Davide). Vale ancora come principio quando il modello principale è
   Fable.
8. **Prima della sessione**: pooled head di Kubaty invece delle teste di Pomponi; ricetta SGD con 320 epoche e ultima
   epoca; α non vincolato e fisso dopo il training; consistency risk con UCB WSR; split 4000/3000/3000; criterio della
   settimana 4 fissato prima; il bound del controller presentato come adattamento.

## 9. I problemi aperti principali

Elenco completo in `05_OPEN_ISSUES.md`.
- **Il riferimento debole** (rilievo R01): resta da decidere quale riferimento e quale rischio usare nella parte online.
- **La fase di adattamento** che ripaga il suo costo non è ancora stata trovata: M8b recupera savings ma non si ripaga
  sugli orizzonti reali (pilota B2).
- **README e findings** non ancora aggiornati.
- **Le review di ICLR 2027** (pubbliche dal 5 novembre) e il testo della submission 22325 non sono stati letti:
  OpenReview blocca gli accessi automatici. Serve Davide dal browser.
- **Scadenza del corso** non nota.
- **Nessuna latenza reale misurata**; solo ResNet20 e CIFAR; dati di conferma ancora chiusi.

## 10. Come vuole lavorare Davide

Dettagli in `06_NEW_AGENT_INSTRUCTIONS.md`. In breve:
- Risposte in italiano, brevi e dirette, con le decisioni in cima.
- Seguire il piano e fermarsi ai checkpoint.
- Niente git che modifichi senza richiesta; non toccare `third_party/` e gli output esistenti.
- Pochi agenti, uno alla volta; il compute in job di shell; verificare sempre i risultati degli agenti.
- Rigore: ogni numero con una fonte, previsioni scritte prima, risultati negativi riportati, nessuna novità senza fonti
  primarie.
- Ogni job oltre un'ora o qualunque uso di Colab richiede il suo ok (il job M8b in corso è approvato).
- I testi per il professore seguono regole di stile precise (niente trattini, due punti, punti e virgola eccetera).

## 11. Se la nuova chat non vede questo Mac

Il pacchetto presuppone l'accesso alla cartella del progetto. Se la nuova istanza non ce l'ha, servono almeno:
- questo pacchetto (`outputs/fable_review/handoff/`);
- `documenti/piano_fable_5_2.md` (v3), `outputs/fable_review/01_revisione.md`, `01_tracciamento_proposto.md`,
  `02_letteratura.md`, `02b_piloti.md`, `quaderno.md`, `documento_progetto.md`;
- `README.md`, `docs/findings.md`, `docs/controller_bound.md`, `HANDOFF.md`;
- le note `outputs/fable_review/parte2/L1–L4`.

I dati (9,3 GB) e gli output delle reti (108 GB) non si possono trasferire con il pacchetto: senza di essi la nuova
istanza può pianificare e scrivere, non eseguire esperimenti. Elenco completo in `07_FILE_INDEX.md`.

## 12. Indice del pacchetto

| File | Contenuto |
|---|---|
| `00_MASTER_HANDOFF.md` | Questo documento |
| `01_TECHNICAL_STATE.md` | Ambiente, struttura, moduli, script, test, dati, output, stato di git, modifiche non committate |
| `02_RESEARCH_AND_REFERENCES.md` | Problema, ipotesi, risultati, letteratura, direzioni scelte e scartate |
| `03_DECISION_LOG.md` | Registro delle decisioni con contesto, motivi, alternative e stato |
| `04_TASKS_AND_ROADMAP.md` | Attività, agenti, job in background, dipendenze, checkpoint, roadmap |
| `05_OPEN_ISSUES.md` | Bug, problemi metodologici, incongruenze, rischi, decisioni da prendere |
| `06_NEW_AGENT_INSTRUCTIONS.md` | Come operare: regole, standard, preferenze di Davide, errori da evitare |
| `07_FILE_INDEX.md` | Indice dei file importanti, con percorsi e descrizioni |
| `08_CONVERSATION_TIMELINE.md` | I messaggi di Davide alla lettera, in ordine, con cosa ne è seguito |
| `MASTER_PROMPT_DI_RIPRESA.md` | Il prompt da incollare nella chat nuova |
| `piano_fable_5_2_v2_originale.md` | Copia integrale del piano v2, prima della riscrittura v3 |
| `pytest_handoff.log` | Esecuzione dei test del 14 novembre (136 passati) |

