# Piano di ripresa: savings computazionali degli early exit sotto shift, con una fase di adattamento

Progetto: "Compute savings degli early exit sotto shift" (Davide De Blasio, corso di Neural Networks del prof. Simone
Scardapane, Sapienza, da estendere alla tesi magistrale).

Versione 3 del 14 novembre 2026. Sostituisce la versione 2 dell'11 novembre, la cui copia integrale è in
`outputs/fable_review/handoff/piano_fable_5_2_v2_originale.md`. Le Parti 1 e 2 della v2 (revisione critica e
letteratura) sono fatte: qui ne restano solo gli esiti, con i rimandi ai file. Le fasi successive sono riscritte con le
modifiche che i risultati richiedono, attorno alla frase del professore. Le modifiche sono elencate in fondo, nel
registro delle revisioni.

## 0. Come usare questo documento

**Per Davide.** Il prompt per una chat nuova è in `outputs/fable_review/handoff/MASTER_PROMPT_DI_RIPRESA.md`.

**Per l'agente.**
- Prima di questo piano leggi `outputs/fable_review/handoff/00_MASTER_HANDOFF.md`: dice dove si è fermato il lavoro.
- Poi leggi questo piano per intero e seguilo dalla Fase A, fermandoti ai checkpoint.
- Dove questo documento e il codice non concordano, vale il codice, e la discrepanza va segnalata.
- `outputs/fable_review/02b_piloti.md` ha più dettagli sui piloti. Dove differisce da questo piano, vale questo piano.

---

## 1. La bussola

Dopo la proposta di inizio novembre (`documenti/mail_prof_corretta.txt`) il professore ha risposto, testualmente:

> "Come direzione va benissimo. Possiamo studiare soprattutto come cambiano i savings a livello computazionale sotto
> shift (considerando che devi anche aggiungere una fase di adattamento)."

Davide il 14 novembre ha chiesto di ripartire da qui. In pratica:
1. **La quantità principale sono i savings computazionali** di una early exit network con risk control senza label
   sotto shift: in MAC, e in tempo reale come controllo.
2. **Ogni risultato deve dire come cambiano i savings sotto shift e perché.**
3. **Il sistema deve avere una fase di adattamento**, e il suo compute va contato: savings lordi contro netti, payback.
4. **Teoria e nuove definizioni di rischio sono di supporto**, non il contributo principale. La Parte 2 ha mostrato che
   le idee teoriche sul rischio da certificare sono già in letteratura (sezione 4.2).

Ogni fase qui sotto deve rispondere a "come cambiano i savings sotto shift" e includere o misurare una fase di
adattamento. Le scelte di metodo (cosa adattare, quali score o soglie, quali dataset) possono cambiare con i risultati:
basta motivarle e tenerne traccia.

## 2. Il ruolo e gli standard

Ricercatore senior in machine learning e co-supervisore scientifico di Davide. Il compito è doppio: trovare ciò che è
sbagliato, debole o sopravvalutato, e trasformare il progetto in un contributo nuovo, solido e utile, che sia anche il
nucleo di una tesi da 110 e lode. Le competenze richieste sono elencate nella v2, sezione 1.1 (early exit e dynamic
networks, distribution-free UQ, online risk control, strumenti statistici, distribution shift, test-time adaptation,
efficienza reale).

Standard (v2, sezione 1.2, confermati):
1. Scetticismo verso i risultati, anche i propri: prima di credere a un numero, cerca il modo più semplice in cui
   potrebbe essere sbagliato.
2. Ogni numero ha una fonte: file, comando, commit o stato del working tree.
3. Tre livelli sempre separati: VERIFICATO, DEDOTTO, IPOTESI.
4. Novità solo su fonti primarie, scritta come "da quello che ho trovato al giorno X".
5. Uccidi le idee in fretta: pilota di al massimo un giorno, con un criterio di kill scritto prima.
6. Il test del revisore 2 per ogni contributo.
7. Una tesi centrale con due o tre contributi, non un elenco.
8. Consapevolezza del compute: stima tempo, CU e disco prima di ogni esperimento.
9. Semplicità: codice da studente di magistrale, leggibile e commentato.
10. Con Davide in italiano, breve e diretto, le cose da decidere in cima. Un risultato negativo riportato bene vale
    quanto uno positivo.

---

## 3. Il progetto in due minuti

### 3.1 Setting tecnico (invariato)

- **Rete.** ResNet20 multi exit su CIFAR10 e CIFAR100, con 9 exit (uno dopo ogni blocco). Exit da 1 a 8: pooled head
  (α·maxpool + (1−α)·avgpool a 4x4, Linear, più un'unità di confidence). Exit 9: GAP e Linear.
- **Regola di uscita.** Primo exit con score > λ (in senso stretto), λ condivisa da tutti gli exit.
- **Consistency risk.** 1[ŷ_exit ≠ ŷ_9], senza label.
- **Calibrazione.** UCB con il bound WSR (δ = 0,1) e fixed sequence testing su una griglia decrescente,
  [1] + 1 − logspace(−4, 0, 200).
- **Savings.** 1 − E[c_exit]/c_M0, con c_M0 la ResNet20 senza teste (circa 40,6 M MAC). Frazioni cumulative degli exit
  1–8: circa 0,13, 0,24, 0,36, 0,45, 0,56, 0,68, 0,77, 0,88. Le teste costano lo 0,12% su CIFAR10 e l'1,08% su
  CIFAR100 (MAC di conv e Linear; con il pooling contato 0,52% e 1,49%).
- **Regimi** (Kubaty et al. 2025, budget di 320 epoche): joint, disjoint, mixed, joint_sd (self distillation, α = 0,5,
  τ = 3, nostra aggiunta). 3 seed, 24 reti.
- **Split del test.** Adapt 4000, cal 3000, eval 3000; 5 split (3 per M8b); stessi indici su CIFAR-C.
- **Livelli di rischio.** ε = 0,02, 0,05, 0,1; i numeri sono a ε = 0,05 se non è detto altrimenti.
- **Metodi del blocco 3.** M1 λ pulita fissa; M2 λ ricalibrata sul dominio shiftato (riferimento ottimistico: dominio
  noto, 3000 forward complete); M8a solo lo score agreement rifittato; M8b teste distillate verso l'exit 9 con
  penalità L2 verso i pesi allenati, poi M2.
- **Score.** Max softmax, temperature per exit, P(correct) e P(agreement) appresi su 3 statistiche dell'uscita.

### 3.2 Obiettivi

- Consegna del progetto di Neural Networks (scadenza da chiarire con Davide).
- Una pubblicazione tra ottobre e dicembre 2027 (HANDOFF.md); un workshop prima sarebbe un vantaggio.
- Una tesi magistrale da 110 e lode. Estensioni possibili: finanza (HANDOFF.md §8), device e server (Pomponi et al.
  2024), soglie per exit online.

---

## 4. Cosa è già fatto

### 4.1 Blocchi 1–3 (6–9 novembre) e controlli preliminari (10 novembre)

Dettagli in README (sezioni Block 1, 2 e 3) e in `docs/findings.md`. In breve: porting esatto di Fast yet Safe; 24
reti allenate ed esportate; savings a rischio fissato sotto shift (M1, M2, M8a, M8b); bound del controller dimostrato
(`docs/controller_bound.md`, un adattamento, non un risultato nuovo).

### 4.2 Parte 1: revisione critica (11–13 novembre)

Deliverable: `outputs/fable_review/01_revisione.md` (24 rilievi). Esperimenti nel quaderno
`outputs/fable_review/quaderno.md` (voci P1.01–P1.18). Script portati nella repository in `scripts/6_block3_review/`,
output in `outputs/block3_review/`. Esiti principali:
- **Il codice è corretto e riproducibile.** 136 test passano; righe di `results.csv` rigenerate da zero uguali a 3e-9;
  riproduzione di Fast yet Safe identica; nessun leakage.
- **Il calo dei savings è confermato con intervalli.** A severity 3 resta 0,59 [0,56, 0,62] dei savings puliti su
  CIFAR10 e 0,53 [0,50, 0,56] su CIFAR100 (M2, ε = 0,05). Tra le corruzioni va da 0,41 (glass blur) a 0,95 (brightness);
  la famiglia rumore è la peggiore (0,44 e 0,25).
- **M2 è valido ma ottimistico.** Con 100 split per dominio la sua λ supera ε nel 6,7% e 5,7% dei casi sulle righe fuori
  dalla calibrazione. L'oracolo di soglia dà solo 0,009–0,015 in più. Su uno stream di 10.000 input i savings netti di
  M2 scendono del 30% per le 3000 forward di calibrazione.
- **I "valid savings" di M1 sono una metrica fragile.** Sostituiti dai savings lordi con il loro rischio.
- **Il riferimento è debole.** Nei regimi congiunti l'exit 9 è al massimo accurato quanto l'exit 8 anche sotto shift.
  Alla λ certificata su CIFAR100 pulito l'exit 9 ha ragione solo nel 28% dei disaccordi. Con un ensemble degli exit
  7–9 come riferimento i savings puliti su CIFAR100 passano da 0,231 a 0,293, di cui circa metà per un effetto
  meccanico (+0,032 senza sovrapposizione).
- **M8b non ripaga il suo compute.** Il training delle teste vale circa 2.600 forward complete su CIFAR10 e 30.900 su
  CIFAR100 (il documento di progetto stimava circa 650). A severity 3 si ripaga dopo circa 80.000 e 1,7 milioni di
  input dello stesso dominio.
- **Due affermazioni non reggevano.** joint_sd meglio di joint su CIFAR10 (un solo seed) e il "passa" del criterio
  della settimana 4. Su CIFAR100 il vantaggio di joint_sd è netto (+0,036, errore 0,005).
- **Il bound del controller è corretto ma largo:** 0,156 dopo 10.000 input e 0,081 dopo 100.000 (ε = 0,05, π = 0,05,
  γ = 0,01, δ = 0,1).
- **M8b su tre seed (parziale, 19 reti su 24, 14 novembre).** Il guadagno si trasferisce: 0,058 sul seed 0, 0,053 sui
  seed 1 e 2 (scarto 0,005). Il criterio della settimana 4 passa solo per CIFAR10 joint_sd, al limite su ogni seed
  (0,53, 0,49, 0,60).

### 4.3 Parte 2: letteratura (13–14 novembre)

Deliverable: `outputs/fable_review/02_letteratura.md`, note in `outputs/fable_review/parte2/`. Esiti principali:
- **Già fatto da altri:** il disaccordo con il riferimento non misura la qualità del task (submission ICLR 2027 22466);
  senza label il disaccordo è il massimo identificabile (22325); controllo dell'accuracy gap con poche label in un
  early exit (Einbinder et al. 2024); controllo online con audit e pesi inversi (Kiyani et al., Gibbs et al.);
  test-time adaptation delle teste di un early exit senza garanzie (TinyTTA, DAdEE).
- **Non trovato al 14 novembre 2026:** una misura dei savings computazionali di un early exit con risk control senza
  label sotto shift a garanzia mantenuta; il bilancio in compute di una fase di adattamento dentro un risk control;
  la scomposizione del consistency risk alla soglia certificata sotto shift; l'osservazione che distillare le teste
  verso un riferimento degradato recupera savings copiandone gli errori.
- **Conseguenza.** La direzione A della v2 (quale rischio certificare) diventa un'analisi dentro il bilancio dei
  savings. Il contributo è la domanda del professore.

### 4.4 Domande e ipotesi di partenza: stato al 14 novembre

| | Domanda o ipotesi | Stato |
|---|---|---|
| RQ1 | Quanto calano i savings sotto shift se il consistency risk resta sotto ε? | Risposta offline solida, con intervalli, per famiglia, contro n_cal e δ |
| RQ2 | Senza label e online, quanti savings perde un controller con audit rispetto alla soglia migliore, e in quanto si stabilizza? | Non iniziata (Fase C) |
| RQ3 | Quanto recupera una fase di adattamento al netto di audit e training, e dopo quanti input si ripaga? | Offline: lordo misurato su 3 seed (parziale); payback di M8b misurato; versione economica da trovare (Fase B2) |
| H1 | I savings calano con la severity soprattutto perché gli exit superficiali perdono traffico | Confermata su CIFAR10, in parte su CIFAR100 |
| H2 | Con la soglia fissa il rischio supera ε alle severity alte | Confermata (CIFAR10 58–90% dei casi, CIFAR100 29–55%) |
| H3 | Il calo dipende da regime e score; l'agreement regge meglio della max softmax | Il regime conta (CIFAR100); lo score poco (al massimo 0,04, indipendente dalla griglia) |
| H4 | L'adattamento recupera parte dei savings; il costo viene soprattutto dagli audit | Prima metà vera offline. Seconda falsa su CIFAR100 (domina il training), dipende dalla contabilità su CIFAR10 |
| H5 | Allenando solo sugli input arrivati in fondo si recupera meno, o il rischio sale | Non testata (Fase C) |

---

## 5. Regole operative (non negoziabili)

1. **Git.** Niente commit, push, branch, stash o reset senza una richiesta esplicita di Davide.
2. **`third_party/` è in sola lettura.**
3. **Repository pulita.** Report, piloti e file di prova in `outputs/fable_review/` (ignorata da git). Il codice entra in
   `src/` o `scripts/` solo dopo l'approvazione di Davide, nello stile esistente: commento in testa, funzioni brevi, un
   test in `tests/`.
4. **Output esistenti.** Mai sovrascrivere file in `outputs/cifar/`, `outputs/savings_under_shift/`,
   `outputs/block3_review/` o `outputs/savings_under_shift_v2/`. I nuovi risultati vanno in cartelle nuove.
5. **Compute locale.** Mac con chip M5 (MPS) e caricatore da 30 W.
   - Al massimo 3 processi pesanti; con MPS attivo al massimo 2 (il 14 novembre 4 worker hanno scaricato la batteria
     dall'80% a meno del 30% in 2,5 ore).
   - Controlla la batteria con `pmset -g batt` e fermati sotto il 30%; per i job lunghi usa un watchdog come quello di
     `outputs/fable_review/esecuzione/run_m8b_resume.sh`.
   - Usa `caffeinate -i`. Stima il tempo prima: alcuni script di revisione hanno richiesto 5 ore.
6. **Colab.** Colab Pro con A100, circa 200 CU. Il training lo lancia Davide da `notebooks/colab_training.ipynb`. Ogni
   job oltre un'ora in locale, o qualunque uso di Colab, richiede prima la sua approvazione, con stima di tempo e CU.
7. **Agenti e crediti.** Davide ha chiesto più volte di ridurre il consumo di crediti.
   - Un agente alla volta, con un compito stretto che finisce in un colpo, che scrive le sue note su file man mano.
   - Niente agenti in parallelo: il 13 novembre cinque agenti insieme hanno colpito tre volte il limite d'uso e circa
     2,5 milioni di token non hanno prodotto risultati.
   - Il compute puro va in job di shell in background, che non costano crediti.
   - Se il modello principale è Fable: Fable pensa e decide, gli agenti Opus (effort xhigh) eseguono, Fable verifica.
     Se il modello principale è Opus: lavora direttamente, e usa agenti solo quando servono davvero.
   - Verifica sempre tu stesso i risultati degli agenti prima di riportarli.
8. **Tracciabilità.** Ogni scoperta diventa una proposta di testo per `docs/findings.md` (in inglese, nello stile del
   file) e una riga per il log del documento di progetto (in italiano). Davide approva prima che si scriva in
   `docs/findings.md`. Il documento di progetto online (Claude Doc) non si modifica senza il suo ok.
9. **Testi per il professore** (regole di Davide, alla lettera): niente trattini di nessun tipo, due punti, punti e
   virgola, asterischi, hashtag o emoji; voce attiva e frasi semplici; condizionale in prima persona per i piani; si dà
   del lei; termini tecnici in inglese; citazioni come "Autore et al. anno", senza titoli; controlla i caratteri vietati
   con uno script prima di consegnare (c'è un esempio in `/Users/davide/Desktop/Tesi/research_bundle/scripts/check_chars.py`).
10. **Gemini.** Davide ha accesso a Gemini per controlli incrociati: se un risultato teorico o una scelta di metodo ne
    beneficia, prepara la domanda e chiedigli di farla.
11. **Navigazione web.** Per le pagine da navigare usa la skill `/browse` di gstack; mai gli strumenti
    `mcp__claude-in-chrome__*` (istruzione globale di Davide). Non aggirare protezioni anti bot (OpenReview).

## 6. Disciplina sperimentale

### 6.1 Esplorazione e conferma su dati diversi

| Uso | Dati |
|---|---|
| Esplorazione: piloti, scelte di metodo, iperparametri | Le 15 corruzioni standard, sulle reti esistenti |
| Conferma: una sola volta, a metodo congelato | Le 4 corruzioni extra (logit già esportati, mai usati per i savings), CIFAR-10.1 (da scaricare), reti con seed nuovi |

Prima di aprire i dati di conferma scrivi in un file cosa ti aspetti e con quale soglia. Le 4 corruzioni extra sono
una conferma debole (simili alle standard); lo shift naturale è quella forte.

### 6.2 Il rumore di fondo (misurato nella Parte 1)

| Fonte di variabilità | Deviazione standard dei savings di M2 a severity 3 |
|---|---|
| Tra i 3 seed | Da 0,002 a 0,008 in 7 setting su 8; 0,034 per CIFAR10 joint_sd (segno di α) |
| Tra gli split, a rete e corruzione fissate | 0,009 |
| Tra le corruzioni, a rete fissata | 0,064 (0,085 CIFAR10, 0,044 CIFAR100) |
| Tra i seed, del guadagno di M8b su M2 | 0,005 |

- L'unità di replicazione è il seed, poi la corruzione. Gli split non sono repliche indipendenti.
- Intervalli con bootstrap gerarchico (seed, corruzione, split) o test appaiati sui seed; il codice è in
  `scripts/6_block3_review/intervals.py`.
- Un effetto sotto 0,02 di savings, o sotto 2 deviazioni tra seed, è nullo finché non ci sono più seed.
- Riporta sempre anche il risultato per famiglia di corruzione (rumore, blur, meteo, digitale).

### 6.3 Il quaderno di laboratorio

`outputs/fable_review/quaderno.md`. Per ogni esperimento: data e domanda; previsione scritta **prima**; comando e
cartella di output; risultato in due righe; cosa cambia e passo successivo.

### 6.4 Baseline, limiti superiori, una cosa alla volta

Ogni metodo si confronta con la baseline più semplice, con il metodo più forte della letteratura applicabile e con un
oracolo. Un fattore per esperimento, confronti appaiati sugli stessi split; se cambi una metrica riporta la vecchia e
la nuova.

### 6.5 Il metro comune

Ogni risultato si riporta con due metri:
1. **Savings computazionali lordi e netti** a garanzia fissata (consistency risk ≤ ε con UCB), con il compute di
   calibrazione, audit e adattamento sottratto su un orizzonte H (10.000, 100.000 e 1 milione di input).
2. **La frontiera accuracy contro compute** al variare di λ (label usate solo per valutare), con sopra i punti scelti da
   ogni metodo. Serve quando il riferimento cambia (per esempio dopo una BN adaptation), perché allora i savings a
   consistency risk fissato non bastano.

---

## 7. Fase A: chiusure prima dei piloti

| | Cosa | Stato al 14 novembre | Dipende da |
|---|---|---|---|
| A1 | Finire M8b sui seed 1 e 2 e aggiornare l'analisi a tre seed | In corso: job staccato avviato alle 11:38, 5 reti di CIFAR100 mancanti | Batteria |
| A2 | Aggiornare README e `docs/findings.md` con le correzioni approvate il 13 novembre | Da fare (il codice è già aggiornato) | A1, per la colonna M8b |
| A3 | Righe per il log del documento di progetto, nello stile dei testi per il professore | Da fare | A2 |
| A4 | Piccole correzioni: `compare_b4.py` che si ferma; la riga `.gstack/` in `.gitignore` | Da fare o da decidere | Davide per `.gitignore` |
| A5 | Review di ICLR 2027 (22466, 22325, 22472) e testo della 22325 | Bloccato: OpenReview blocca gli accessi automatici | Davide, dal browser |

**A1.** Controlla `outputs/savings_under_shift_v2/STATUS.txt`. Se il job è finito con `exit=0`, esegui
`python outputs/fable_review/esecuzione/m8b_three_seeds.py` e
`python scripts/5_savings_under_shift/savings_figures.py --results outputs/savings_under_shift_v2/results.csv --out-dir outputs/block3_review/figures_v2`.
Se si è fermato, rilancialo con lo stesso script (salta le reti finite). Poi controlla il tasso di violazione di M8b
sulle righe fuori dalla calibrazione (nel parziale il seed 2 era a 0,12 sulle righe di eval).

**A2.** I testi proposti sono in `outputs/fable_review/01_tracciamento_proposto.md` (sezione findings, frasi da
riscrivere, test) e i rilievi in `outputs/fable_review/01_revisione.md`. Ogni numero nuovo deve venire da un output di
`outputs/block3_review/`. Attenzione: `docs/figures/savings_by_method.png` è già la figura nuova (M1 con i savings
lordi), mentre il testo del README descrive ancora i valid savings.

> **Checkpoint 2b.** Fine della Fase A: fotografia dello stato a Davide e suo ok per i piloti. Il Checkpoint 2 della
> v2 (letteratura ed elenco dei piloti) è stato consegnato il 14 novembre e Davide ha detto di continuare, poi ha
> chiesto di fermarsi prima dei piloti per passare a una chat nuova.

---

## 8. Fase B: piloti offline

**Ordine.** B1 e B2 per primi, perché rispondono direttamente al professore; poi B3 e B4. Tutti sulle 15 corruzioni
standard e sulle reti esistenti, ciascuno sotto un'ora sul Mac (B2 un po' di più). Output in
`outputs/fable_review/pilots/<nome>/`, voce nel quaderno con la previsione prima di lanciare.

### B1. Il bilancio computazionale dei savings sotto shift

- **Domanda.** Dove si perde il margine tra ciò che la rete potrebbe risparmiare sotto shift e ciò che M2 certifica?
- **I livelli**, per dataset, regime e severity (savings in MAC; oracoli scelti su metà delle righe e misurati
  sull'altra):
  1. M2 sul pulito;
  2. oracolo per input sul dominio shiftato (primo exit che concorda con l'exit 9), anche con budget ε (spendere ε di
     disaccordo sugli input dove uscire prima fa risparmiare di più);
  3. oracolo con soglie per exit (8 soglie sulla max softmax, rischio ≤ ε);
  4. oracolo di soglia condivisa (già misurato: 0,009–0,015 sopra M2);
  5. M2 con UCB a n_cal 3000, 1000, 250 e con CRC (già misurato in `outputs/block3_review/calibration_budget*`);
  6. M2 al netto delle forward di calibrazione su orizzonti di 10.000 e 100.000 input;
  7. accanto, per lettura: savings a parità di accuracy persa relativa (label solo per valutare), e con un riferimento
     senza sovrapposizione con gli exit usabili (già misurato in `outputs/block3_review/reference_*`).
- **Ipotesi.** Il salto più grande è tra l'oracolo per input e l'oracolo di soglia condivisa (lo score non riconosce
  gli input che concordano): almeno il 60% del margine totale a severity 3.
  *Previsione:* 0,30–0,40 di savings su CIFAR10 a severity 5, contro 0,015 di prezzo statistico.
- **Kill.** Nessuno: è una misura. È la figura centrale candidata.
- **Revisore 2.** "L'oracolo per input è irraggiungibile." Risposta: per questo c'è il livello 3, raggiungibile da una
  regola a 8 soglie.
- **Costo.** Circa un'ora, solo logit esistenti.

### B2. Una fase di adattamento che ripaga il suo costo

- **Domanda.** Quale adattamento recupera savings al netto del suo compute, e lo fa migliorando l'accuracy o copiando
  gli errori del riferimento?
- **Candidati**, stessi split, seed 0, regimi joint e joint_sd su entrambi i dataset, severity 3 e 5, 15 corruzioni:
  - (a) BN adaptation: statistiche ristimate sulle righe di adapt, sole forward (cambia anche il riferimento);
  - (b) M8b com'è;
  - (c) (a) e poi teste distillate verso il riferimento adattato;
  - (d) adattamento nello spazio dei logit: una mappa affine sui logit di ogni testa, fittata per distillazione
    (N·C² per valutazione invece di N·D·C);
  - (e) M8b economico: un solo L2 (0,01) e 5, 10, 20 iterazioni;
  - (f) soglie per exit riapprese sulle righe di adapt (da B4), l'adattamento più economico;
  - (g) M8a, lo score rifittato, come baseline che si sa perdere.
- **Cosa misurare.** Guadagno lordo su M2; compute in forward complete (righe di adattamento più training); payback
  H_p; savings netti su 10.000, 100.000 e 1 milione di input; accuracy di teste e sistema e accordo con l'exit 9, prima
  e dopo; savings certificati sull'accuracy persa (label solo per valutare); la frontiera accuracy contro compute.
- **Ipotesi D1.** Una variante economica, (d), (e) o (f), tiene almeno il 70% del guadagno di M8b con al più il 10% del
  suo compute, e porta il payback a severity 3 sotto 30.000 input su entrambi i dataset.
- **Ipotesi D2.** (c) supera (b) di almeno 0,02 di savings (adattano cose diverse).
- **Ipotesi D3 (copia degli errori).** Con M8b l'accordo delle teste con l'exit 9 sale molto più della loro accuracy, e
  meno di metà del guadagno sopravvive se si certifica l'accuracy persa. *Previsione:* vera su CIFAR100 sotto rumore.
- **Kill.** Se nessuna variante tiene almeno metà del guadagno con al più il 20% del costo, l'adattamento delle teste
  non ripaga sugli orizzonti reali: si riporta come risultato negativo e la fase di adattamento online usa la
  ricalibrazione o le soglie per exit. Se (c) non supera (b) di 0,02, la BN adaptation resta solo un modo per
  migliorare il riferimento e la frontiera.
- **Revisore 2.** "La letteratura TTA ha metodi più forti, e il payback dipende dall'implementazione." Risposta: curve
  guadagno contro compute, non un punto; un metodo alla Tent se le varianti lasciano margine.
- **Sotto pilota opzionale.** Con gli early exit attivi i blocchi profondi vedono solo gli input difficili: le
  statistiche BN stimate su quel traffico sono distorte, quelle stimate sugli audit casuali no. Misurare la differenza.
- **Costo.** Due job da circa 45 minuti (forward su MPS per la BN adaptation; adattamenti su CPU), al massimo 2 processi.
- **Esito atteso per il progetto.** La scelta della fase di adattamento da usare online (Fase C).

### B3. Il meccanismo: traffic shift contro reliability shift

- **Domanda.** Sotto shift i savings calano perché meno input superano la soglia (traffico) o perché a pari score gli
  input concordano meno (affidabilità)?
- **Misure.** Curva di affidabilità P(disaccordo | exit, score) sul pulito e sullo shiftato; controfattuale esatto nei
  due ordini (score shiftati con affidabilità pulita e viceversa, la λ che riporta il rischio a ε, i savings che ne
  seguono); AUROC della previsione di accordo per exit e severity; un classificatore dell'accordo con le statistiche di
  tutti gli exit già calcolati, allenato sulle righe di adapt, come limite più realistico dell'oracolo.
- **Ipotesi C1.** Il reliability shift spiega almeno metà del calo a severity 3 nei regimi congiunti su CIFAR10.
  *Previsione:* 50–65%.
- **Ipotesi C2.** Il classificatore con più exit recupera almeno 0,03 di savings sulla max softmax sotto shift.
  *Previsione:* incerta.
- **Kill.** Se i due ordini danno risposte opposte e nessuna misura distingue le due cause, resta la parte descrittiva.
- **Costo.** Circa 30 minuti.

### B4. Soglie per exit come adattamento minimo

- **Domanda.** Soglie diverse per exit, consapevoli del costo, battono la λ condivisa a parità di garanzia? Sotto shift
  basta riapprenderle (8 numeri invece di 100.000 pesi)?
- **Versioni.** E1: score calibrato più regola di Lagrange (uscire all'exit l se P(disaccordo) < μ·(c_9 − c_l)),
  certificata con UCB lungo μ. E2: cammino di soglie appreso sulle righe di adapt per budget di rischio crescenti, poi
  certificato con UCB e fixed sequence testing sulle righe di cal (valido perché il cammino non vede le righe di cal e
  parte da "nessuna uscita"). E3: limite superiore dal livello 3 di B1.
- **Ipotesi E1.** E2 guadagna almeno 0,02 sulla λ condivisa in almeno 5 setting su 8, sul pulito e a severity 3.
- **Ipotesi E2.** Un cammino appreso sul pulito e solo ricertificato sul dominio shiftato tiene almeno l'80% del
  guadagno del cammino riappreso.
- **Kill.** Se E2 guadagna meno di 0,02 nella maggior parte dei setting.
- **Revisore 2.** "Incrementale rispetto a Pareto Testing." Risposta: l'angolo è lo shift e il costo quasi nullo come
  fase di adattamento, confrontato con B2 nel bilancio netto. Il controllo preliminare 3 della v2 dice che una famiglia
  fissa proporzionale al costo non basta (da −0,06 a +0,04).
- **Costo.** Circa 45 minuti.

### Direzioni ridotte o chiuse

- **A (quale rischio certificare).** Non è più un contributo a sé (22466, 22325, Einbinder et al. 2024). Restano tre
  pezzi dentro B1 e B2: la scomposizione dei disaccordi alla λ certificata, i savings a parità di accuracy persa, la
  copia degli errori (D3).
- **F (prevedere i savings senza pagarli).** Chiusa come direzione: CRC con 100 righe dà già 0,53 di savings su CIFAR10
  pulito contro 0,54 con 3000, quindi prevedere deve battere una misura che costa 100 forward. Resta una correlazione
  descrittiva dentro B1 (savings di M2 contro accuracy dell'exit 9 tra i domini).
- **Il lemma "il consistency risk è il miglior certificato senza label"** si può scrivere come osservazione con la
  citazione della 22325, non come risultato.

> **Checkpoint 3a.** Dopo B1 e B2: `outputs/fable_review/03_direzioni.md` con esiti dei piloti, numeri, figure, comandi,
> matrice dei punteggi (criteri della v2, sezione 8.3: allineamento con il professore 25%, impatto 20%, novità
> verificata 20%, fattibilità 15%, rischio 10%, valore per la tesi 10%), critica del revisore 2 e raccomandazione
> sulla fase di adattamento da usare online. Davide decide prima della Fase C.

---

## 9. Fase C: la parte online (le settimane 5–7 del documento di progetto)

Qui vivono RQ2, RQ3, H4 e H5. Il disegno di partenza è nel documento di progetto (copia in
`outputs/fable_review/documento_progetto.md`, sezioni "Il bilancio del compute sotto shift", "Il sistema online",
"Garanzie", "Esperimenti"). Si tiene, con le modifiche sotto.

**Il sistema.** In modalità normale gli input escono con λ_t; una frazione π degli input usciti diventa un audit
(completato fino all'exit 9); un controller aggiorna λ_t con la regola di `src/eesavings/controller.py`. Quando un
trigger segnala lo shift (λ_t oltre λ̂ + κ, oppure un test sequenziale sul disaccordo) parte la fase di adattamento.

**I metodi** (documento di progetto): M0 full network; M1 Fast yet Safe statico; M2 soglia ottima sul dominio nuovo;
M3 monitor con fallback alla full network (come la 22466); M4 controller con audit, solo soglia; M5 controller più
adattamento con i soli input arrivati in fondo; M6 con i soli audit; M7 con entrambi; M8 adattamento offline con
accesso al dominio nuovo. Per M7 le tre versioni della fase di adattamento: fallback completo, audit rinforzati
(π_a per N_a input), adattamento continuo (ogni k input, π costante).

**Gli stream.** Salto, rampa di severity, gradini, ritorno al pulito, corruzioni miste; da 20.000 a 50.000 input, 50
ripetizioni, costruiti con gli indici di valutazione. Più un orizzonte lungo in stile CCC se il tempo lo permette.

**Le misure.** Consistency risk nel tempo rispetto a ε (media di lungo periodo e finestra peggiore); savings lordi e
netti; S_0 prima dello shift, S_1 stabilizzati senza adattamento, S_2 dopo l'adattamento; frazione recuperata
(S_2 − S_1)/(S_0 − S_1); payback H_p; accuracy contro le label (solo per valutare); forgetting al ritorno al pulito.

**Le modifiche rispetto al documento di progetto, dovute ai risultati:**
1. **L'aggiornamento deve essere economico.** Il documento stimava il training delle teste in circa 650 forward
   complete su CIFAR100; M8b ne costa circa 30.900. La fase di adattamento online usa la variante scelta in B2 (o le
   soglie per exit di B4), con il costo misurato e incluso in A_t.
2. **Il riferimento.** Il bound vale se backbone ed exit 9 non cambiano (condizione 4). Una BN adaptation cambia il
   riferimento: o si adattano solo le teste e le soglie (la garanzia resta), o il cambio di riferimento si tratta come
   un nuovo inizio del certificato e si riporta l'accuracy accanto.
3. **Il bound è largo sugli orizzonti di CIFAR** (0,156 dopo 10.000 input). Si riporta il rischio realizzato nel tempo,
   e opzionalmente si usa lo score come control variate (stima AIPW della 22466) per stringere il certificato.
4. **Le metriche** sono quelle della sezione 6.5, con i savings netti come metro principale.
5. **H5** si testa con M5, M6 e M7 sulla stessa fase di adattamento.
6. **Confronti con la letteratura.** Il monitor della 22466 (M3), un controller alla SSV o Gibbs (M4 lo è nella
   sostanza), una selezione alla B-PAC se il tempo lo permette.
7. **Allocazione degli audit.** p_l ∝ √(q_l / costo residuo) è Neyman con costi (Angelopoulos et al. 2025): si può
   usare come leva, non è un contributo.

**Costo.** Gli stream usano logit e feature in cache: tutto sul Mac, ore e non giorni, se l'aggiornamento è economico.
Le feature di CIFAR-C sono salvate solo per il seed 0; per gli altri si ricalcolano al volo (circa 20 secondi per
corruzione su MPS, vedi `savings_under_shift.py --recompute-features`).

> **Checkpoint 3b.** Fine della Fase C: risultati online, figura della timeline dei savings netti attorno allo shift con
> frazione recuperata e payback per le tre versioni della fase di adattamento.

---

## 10. Fase D: requisiti per pubblicare (dopo B e C, con l'ok di Davide)

- **Shift naturale.** CIFAR-10.1 (ed eventualmente CINIC-10) sulle reti esistenti: costo quasi zero, è anche il dato
  di conferma forte. Da scaricare (scrivere in un file `.part` e poi rinominare).
- **Conferma sulle 4 corruzioni extra**, a metodo congelato, con la previsione scritta prima.
- **Compute reale.** Latenza a batch 1 su CPU e MPS, con l'overhead delle teste e del controllo della soglia. Senza
  questo un revisore dirà che i savings in MAC non si traducono in tempo. Il pooling delle teste va contato.
- **Una seconda architettura su CIFAR** e la **scala ImageNet** (backbone preallenato con teste disjoint su feature in
  cache; ImageNet val, un sottoinsieme di ImageNet-C, V2, R, Sketch). ImageNet lo deve scaricare Davide; ImageNet-C
  pesa circa 66 GB. Piano di compute e disco per Colab con un'opzione minima e una completa.
- **Training robusto** (AugMix o consistenza tra exit sotto augmentation): 12 reti in più su Colab, stima delle CU
  prima. Kill se la quota di savings rimasta a severity 3 migliora meno di 5 punti.
- **Nei prossimi training** vincolare o fissare α delle pooled head (il segno di α è casuale e confonde i confronti
  tra regimi), e usare più seed.

## 11. Fase E: paper al contrario, roadmap e nota per il professore (la Parte 4 della v2)

1. `outputs/fable_review/04_paper_al_contrario.md`: titolo e abstract di 150 parole con i numeri veri; al massimo tre
   contributi; schizzo della Figura 1; tabella degli esperimenti necessari con lo stato; l'affermazione più debole e
   cosa la renderebbe solida.
2. La lista di controllo per una venue principale: almeno due architetture e due scale; uno shift sintetico e uno
   naturale; ablation; sensibilità a ε, δ, n_cal e tasso di audit; bilancio del compute completo con latenza reale;
   confronto con i metodi più vicini; intervalli corretti; limiti onesti; codice per riprodurre.
3. `outputs/fable_review/05_roadmap.md`: settimane, punti di decisione, compute per fase, rischi e piano B.
4. `outputs/fable_review/06_nota_professore.md`: cosa è emerso dal blocco 3, dalla revisione e dai piloti; come
   presentare la fase di adattamento visto che il criterio non è passato; la direzione proposta; due o tre domande
   (scala ImageNet, scenario device e server, scadenze). Davide la rivede e la manda lui.
5. Testo per il tracciamento: righe per `docs/findings.md` e per il log del documento di progetto.

> **Checkpoint 3.** Consegna tutto e fermati.

**Storyline candidata, da validare con B e C.** "Dove vanno i savings computazionali degli early exit sotto shift, e
quanto costa recuperarli": il calo misurato con intervalli (fatto), il bilancio che mostra dove sta il margine (B1), il
meccanismo (B3), una fase di adattamento giudicata con il bilancio netto (B2, B4), e la versione online con audit (C).
Venue: TMLR, oppure un workshop prima; NeurIPS o ICLR main solo se un rimedio dà un guadagno netto chiaro e c'è la
scala (Fase D). Le scadenze vanno verificate.

---

## 12. Calendario indicativo

| Fase | Durata | Compute |
|---|---|---|
| A. Chiusure | Mezza giornata, più il job M8b | Mac |
| B. Piloti B1–B4 | 2–3 giorni | Mac, al massimo 2–3 processi |
| C. Parte online | 1–2 settimane | Mac |
| D. Requisiti per pubblicare | Da pianificare | Mac e Colab (con approvazione) |
| E. Paper al contrario, roadmap, nota | 1 giorno | Nessuno |

## 13. Errori già fatti, da non ripetere

- **Novità.** Il 21 ottobre "nessuno lo fa così" era falso (la 22466 era pubblica dal 3 ottobre). Il 28 ottobre il
  controllo online con audit risultava "non trovato", ma B-PAC e SSV erano già accettati. A novembre le idee teoriche
  della direzione A sono risultate già in letteratura (22466, 22325, Einbinder et al.). Controlla sempre OpenReview,
  arXiv recente e le venue dell'anno.
- **L'allocazione degli audit con costi** sembrava un contributo, ma era Neyman con costi. **Il controller** è un
  adattamento.
- **Le teste adattate senza L2** vanno in overfitting; lo score fittato su righe viste dalle teste blocca ogni uscita.
- **Le premesse vanno controllate** con un'ora di prova prima di investire (la v1 del piano dava per probabili cose
  smentite).
- **Confronti non equi.** Un ensemble degli exit 7–9 come riferimento sembrava "migliore", ma metà del guadagno era
  meccanica: gli exit 7 e 8 votano nel riferimento con cui vengono confrontati.
- **Convenzioni di metrica.** I "valid savings" (savings azzerati dove il rischio supera ε) dipendevano dal rumore.
- **Spiegazioni causali su pochi punti.** Il segno di α spiega al più un quarto dello scarto tra i seed di CIFAR10
  joint_sd; era stato presentato come la causa.
- **Contabilità esplicita.** "H4 è falsa" dipendeva da come si contano gli audit.
- **Agenti in parallelo.** Cinque agenti insieme hanno colpito il limite d'uso tre volte e ripartivano da zero.
- **Hardware e ambiente.** Due download concorrenti hanno corrotto un file (scarica in `.part` e rinomina); con 8
  processi la batteria è scesa all'8%, con 4 worker e MPS dall'80% al 30% in 2,5 ore; MPS non è deterministico al bit
  in generale (ma la forward dell'export è risultata identica); i pesi salvati su CUDA si caricano con
  `map_location="cpu"`.
- **Shell su macOS.** sed non supporta `\b` (usa Python `re`); `grep -c` che restituisce 0 spezza le catene con `&&`.

## 14. Cosa NON fare

- Non cambiare criteri, metriche o ipotesi a posteriori per far passare un risultato. Se cambi una metrica, riporta
  entrambe.
- Non usare CIFAR-C (né altri domini di test) per scegliere iperparametri del training.
- Non aprire i dati di conferma durante l'esplorazione.
- Non presentare come nuovo ciò che la sezione 4.3 elenca come già fatto.
- Non presentare i controlli preliminari del 10 novembre come risultati (sono stati in buona parte confermati nella
  Parte 1, ma con il protocollo completo).
- Non lanciare esperimenti costosi prima che il pilota dica che ne vale la pena.
- Non riscrivere il codice esistente per gusto.
- Non rifare la Parte 1 o la Parte 2: sono fatte.

---

## Registro delle revisioni

- **Versione 1, 10 novembre 2026.** Prima stesura.
- **Versione 2, 11 novembre 2026.** Dopo la verifica contro codice, documento di progetto e dati: domande e ipotesi di
  partenza, disciplina sperimentale, punto sul riferimento, Parte 3 riscritta attorno ai controlli preliminari, paper al
  contrario. Copia integrale in `outputs/fable_review/handoff/piano_fable_5_2_v2_originale.md`.
- **Versione 3, 14 novembre 2026.** Su richiesta di Davide ("cancellare quello fatto finora e completare con le
  modifiche che servono le fasi successive", ricordando la frase del professore).
  - Parti 1 e 2 tolte e sostituite dai loro esiti (sezione 4).
  - La frase del professore diventa la bussola esplicita (sezione 1) e il metro comune (sezione 6.5).
  - Nuova Fase A di chiusure (M8b su tre seed, README e findings, log del documento di progetto).
  - Piloti riordinati: bilancio computazionale (B1) e fase di adattamento (B2) per primi; la direzione A ridotta a
    un'analisi; la direzione F chiusa.
  - Reintrodotta la parte online del documento di progetto (Fase C), con le modifiche dovute ai risultati.
  - Regole aggiornate: agenti e crediti, batteria e compute, navigazione web.
