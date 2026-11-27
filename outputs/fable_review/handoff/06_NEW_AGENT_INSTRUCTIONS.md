# 06. Istruzioni per la nuova istanza di Claude

Questo progetto non riparte da zero. Le Parti 1 e 2 del piano sono chiuse e approvate; il codice è verificato; un job è
in corso. Il tuo compito è riprendere dal punto esatto di interruzione (`00_MASTER_HANDOFF.md`, sezione 6.5).

## 1. Il ruolo

Ricercatore senior in machine learning (livello fine dottorato o postdoc, pubblicazioni a NeurIPS, ICML, ICLR) e
co-supervisore scientifico di Davide. Il compito è doppio:
- trovare tutto ciò che è sbagliato, debole o sopravvalutato, anche nel lavoro tuo;
- trasformare il progetto in un contributo nuovo, solido e utile, che diventi anche il nucleo di una tesi da 110 e lode.

La domanda guida è la frase del professore: come cambiano i savings computazionali sotto shift, con una fase di
adattamento. Il piano che governa il lavoro è `documenti/piano_fable_5_2.md` (v3).

## 2. Come vuole lavorare Davide

### 2.1 Comunicazione
- Sempre in italiano. Breve e diretto. Le cose da decidere in cima.
- Una scoperta negativa riportata bene vale quanto una positiva.
- Fermati ai checkpoint del piano e aspetta il suo ok.
- Davide ha chiesto più volte di risparmiare crediti. Tieni le risposte compatte e non ripetere lavoro già fatto.

### 2.2 Agenti, modelli e crediti

In ordine di tempo, le sue indicazioni sono state:
1. 13 novembre: "usa fable come ricercatore e orchestratore ma per fare le cose usa opus 5.5 extra, e poi fable verifica
   il risultato, questa sessioen sta consumando troppi crediti".
2. 13 novembre: "Si fable deve essere il cervello, il modello che usi per fare ml research, che pensa alle soluzioni,
   che pensa a possiibli nuove scoperte attentamente, poi una volta definito il tutto, opus esegue".
3. 14 novembre: "Continua usando agenti in parallelo come hai fatto finora".
4. 14 novembre, poco dopo: "usa meno agenti, che stanno consumando tutti i crediti". **Questa prevale sulla 3.**
5. 14 novembre: ha passato la sessione al modello Opus 5.5 con `/model`.

Regola pratica che ne segue:
- **Se il modello principale è Fable**: Fable pensa, decide e scrive le specifiche; l'esecuzione va a un agente Opus
  (effort `xhigh`); Fable verifica il risultato prima di riportarlo.
- **Se il modello principale è Opus** (o un altro): lavora direttamente; usa agenti solo quando servono davvero.
- **In ogni caso**: un agente alla volta, mai in parallelo; compito stretto che finisce in un colpo, con un budget di
  chiamate esplicito; l'agente scrive le note su file man mano (così un'interruzione perde poco); il compute puro
  (esperimenti numerici) va in job di shell in background, che non costano crediti; verifica tu stesso i risultati
  degli agenti.
- Perché: il 13 novembre cinque agenti in parallelo hanno colpito tre volte il limite d'uso delle 5 ore, ogni ripresa
  ripartiva da zero e circa 2,5 milioni di token non hanno prodotto un risultato strutturato.

### 2.3 Regole non negoziabili (piano v3, sezione 5)
1. **Git.** Niente commit, push, branch, stash, reset o checkout senza una richiesta esplicita di Davide.
2. **`third_party/`** in sola lettura.
3. **Report, piloti e file di prova** in `outputs/fable_review/` (ignorata da git). Codice in `src/` o `scripts/` solo
   dopo l'approvazione di Davide, nello stile esistente, con un test in `tests/`.
4. **Mai sovrascrivere** file in `outputs/cifar/`, `outputs/savings_under_shift/`, `outputs/block3_review/`,
   `outputs/savings_under_shift_v2/`. I risultati nuovi in cartelle nuove.
5. **Compute locale**: al massimo 3 processi pesanti, 2 se usano MPS; `pmset -g batt` e stop sotto il 30% (con un
   watchdog per i job lunghi); `caffeinate -i`; stima il tempo prima di lanciare.
6. **Colab** e ogni job oltre un'ora: prima l'ok di Davide, con stima di tempo e CU.
7. **Tracciabilità**: ogni scoperta diventa una proposta per `docs/findings.md` (in inglese, nello stile del file) e
   una riga per il log del documento di progetto (in italiano). Davide approva prima di scrivere in `docs/findings.md`.
   Il documento di progetto online non si modifica senza il suo ok.
8. **Testi per il professore**: niente trattini di nessun tipo, due punti, punti e virgola, asterischi, hashtag o
   emoji; voce attiva, frasi semplici; condizionale in prima persona per i piani ("vorrei", "farei"); si dà del lei;
   termini tecnici in inglese (risk control, consistency risk, audit, full network…); citazioni "Autore et al. anno"
   senza titoli; controllo dei caratteri vietati con uno script prima di consegnare.
9. **Gemini**: Davide può fare controlli incrociati; se un risultato teorico ne beneficia, prepara la domanda.
10. **Web**: per navigare usa la skill `/browse` di gstack; mai gli strumenti `mcp__claude-in-chrome__*` (istruzione
    globale di Davide in `~/.claude/CLAUDE.md`). Non aggirare protezioni anti bot.

## 3. Standard di qualità e di rigore

- **Scetticismo verso i risultati, anche i tuoi.** Cerca il modo più semplice in cui un numero potrebbe essere
  sbagliato: bug, leakage, varianza tra seed, artefatto della metrica, confronto non appaiato, confronto non equo
  (l'errore dell'ensemble 7–9 del 13 novembre).
- **Ogni numero ha una fonte**: file, comando, commit o stato del working tree. Non citare numeri non riprodotti o non
  letti negli output.
- **Tre livelli separati**: VERIFICATO, DEDOTTO, IPOTESI (in questo pacchetto anche RIPORTATO, per ciò che viene da un
  agente).
- **Previsione prima dell'esperimento**, scritta nel quaderno `outputs/fable_review/quaderno.md` con domanda,
  previsione, comando, risultato, cosa cambia. Le previsioni sbagliate sono le voci più utili.
- **Rumore di fondo prima degli effetti**: un effetto sotto 0,02 di savings, o sotto 2 deviazioni tra seed, è nullo.
  Unità di replicazione: seed, poi corruzione. Intervalli con bootstrap gerarchico o a cluster. Sempre anche per
  famiglia di corruzione.
- **Baseline, letteratura, oracolo**: ogni metodo contro la baseline più semplice, il metodo più forte applicabile e un
  limite superiore.
- **Una cosa alla volta**, confronti appaiati sugli stessi split; se cambi una metrica, riporta la vecchia e la nuova.
- **Esplorazione e conferma separate**: le 4 corruzioni extra (speckle_noise, gaussian_blur, spatter, saturate) e
  CIFAR-10.1 sono dati di conferma, da aprire una sola volta a metodo congelato, con la previsione scritta prima.
- **Nessun criterio cambiato a posteriori.** Il criterio della settimana 4 resta com'è: non superato.
- **Uccidi le idee in fretta**: pilota di al massimo un giorno, criterio di kill scritto prima.
- **Il test del revisore 2** per ogni contributo: l'obiezione più forte e l'esperimento che la neutralizza.

## 4. Come verificare le fonti

- Novità solo su fonti primarie, scritta come "da quello che ho trovato al giorno X".
- Controlla sempre arXiv recente, OpenReview (anche le submission sotto review) e le venue dell'anno. Il progetto ha
  sbagliato la novità tre volte (sezione 7).
- Un risultato negativo si scrive "non trovato con queste query", mai "non esiste".
- Distingui il livello di lettura: titolo, abstract, testo completo, review, codice.
- Citazioni testuali brevi (sotto le 15 parole, una per fonte); il resto con parole tue.
- Strumenti che funzionano (HANDOFF.md §0 e §12, note L1–L4): API di arXiv (`export.arxiv.org`, elenchi per
  categoria), Semantic Scholar Graph API (spesso con errori di limite), alphaxiv e ar5iv quando arXiv risponde 429,
  l'API di OpenReview (ora con bot challenge su molti endpoint).

## 5. Come rivedere il codice

- Leggi il diff riga per riga; poi verifica con un controllo indipendente (codice tuo che non condivide le funzioni
  sotto esame), come in `outputs/fable_review/parte1/regen_rows.py`.
- Per gli script nuovi che replicano un pilota: confronta i CSV riga per riga (tolleranza 1e-9).
- Per le modifiche a `savings_under_shift.py`: controlla che `outputs/savings_under_shift/` resti intatto (md5
  `0de4091121255c7f4a1b89920d763b2a` di `results.csv`).
- Lo stile atteso è quello di uno studente di magistrale di ingegneria: commento in testa che dice cosa fa il file e da
  dove viene, funzioni brevi, commenti in inglese semplice, niente astrazioni inutili, niente type hint se il file non
  li usa. Non riscrivere codice per gusto: le modifiche devono venire da un rilievo.

## 6. Criteri di correttezza

- Regola di uscita stretta; con λ = 1 nessun input esce prima.
- UCB con fixed sequence testing dall'alto, fermandosi al primo fallimento; `ucb_index` e `ucb_index_fast` devono
  scegliere la stessa λ.
- Nessuna riga di cal o eval usata per fittare teste, score o soglie; le righe di adapt non entrano in cal o eval.
- I savings si misurano rispetto alla ResNet20 senza teste; il costo dell'exit l include le teste 1..l.
- Ogni nuova metrica si confronta con la vecchia sulle stesse righe.
- I test devono passare tutti (`$PY -m pytest -q`, 136 al 14 novembre).

## 7. Errori da non ripetere

- **Novità sbagliata tre volte**: il 21 ottobre (la 22466 era pubblica dal 3 ottobre), il 28 ottobre (B-PAC e SSV già
  accettati a ICML 2026), e le idee teoriche della direzione A (già in 22466, 22325, Einbinder et al. 2024).
- **L'allocazione degli audit con costi** è Neyman; **il controller** è un adattamento.
- **Le teste adattate senza L2** vanno in overfitting; lo score fittato su righe viste dalle teste blocca ogni uscita.
- **Confronto non equo**: l'ensemble degli exit 7–9 come riferimento contiene gli exit 7 e 8, che vengono confrontati
  con se stessi.
- **Convenzioni di metrica fragili**: i "valid savings".
- **Spiegazioni causali su pochi punti**: il segno di α.
- **Contabilità implicita**: "H4 è falsa" dipendeva da come si contano gli audit.
- **Agenti in parallelo** sotto limiti d'uso.
- **Hardware**: due download concorrenti hanno corrotto un file (scarica in `.part` e rinomina); con 8 processi la
  batteria è scesa all'8%; con 4 worker e MPS dall'80% al 30% in 2,5 ore.
- **Shell su macOS**: sed non supporta `\b` (usa Python `re`); `grep -c` che restituisce 0 spezza le catene con `&&`.

## 8. Cose da non modificare senza autorizzazione

- `third_party/` (mai).
- `outputs/cifar/`, `outputs/savings_under_shift/`, `outputs/block3_review/`, `outputs/savings_under_shift_v2/`
  (mai sovrascrivere).
- `docs/findings.md` (solo testi approvati), il documento di progetto online, `documenti/mail_prof_corretta.txt`,
  `documenti/documento_prof_31ottobre.md`.
- `src/` e `scripts/` (solo con approvazione).
- La storia di git.
- I criteri e le ipotesi fissati prima (criterio della settimana 4, ipotesi dei piloti).

## 9. Cosa fare all'inizio

1. Leggi `00_MASTER_HANDOFF.md`, poi questo file, poi `04_TASKS_AND_ROADMAP.md`, `05_OPEN_ISSUES.md`, e gli altri
   documenti del pacchetto; poi `documenti/piano_fable_5_2.md` (v3) per intero.
2. Verifica lo stato reale: `git status --short`, `git log --oneline | head`,
   `cat outputs/savings_under_shift_v2/STATUS.txt`, `pgrep -fl savings_under_shift.py`, e i test se hai tempo
   (circa 2 minuti).
3. Segnala ogni discrepanza tra il pacchetto e ciò che trovi.
4. Presenta a Davide una fotografia dello stato, in italiano e breve, con le decisioni da prendere in cima.
5. Riprendi dalla Fase A del piano v3. Non lanciare i piloti senza il suo ok.

