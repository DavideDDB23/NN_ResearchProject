# Checkpoint 3a: esiti dei piloti B1 e B2 e direzioni

Stato al 15 novembre 2026, ore 11:05. Piano di riferimento: `documenti/piano_fable_5_2.md` (v3), Fase B. Il dettaglio
di ogni esperimento, con la previsione scritta prima, è in `outputs/fable_review/quaderno.md` (voci B1.01, B2.01,
B2.02, B2.03). Fino alla notte del 15 novembre ha orchestrato Fable (disegno, previsioni, verifiche; il codice di B1 e
B2 l'hanno scritto due esecutori Opus a effort xhigh, uno alla volta). Dalla mattina del 15 orchestra Opus 5.5, che ha
ricontrollato i testi, corretto otto frasi e aggiunto la replica B2.03. I job hanno girato in shell. Nulla è stato
committato e nessun file della repository fuori da `outputs/fable_review/` è stato toccato.

## Cose da decidere

1. **La fase di adattamento per la parte online (Fase C).** Proporrei una scala e un bias per classe sui logit di
   ogni testa, a backbone fermo, con 3.000 righe prese dal budget degli audit. Su 24 reti tiene tre quarti del
   guadagno di M8b e si ripaga dopo circa 28.000 input. La BN adaptation come secondo braccio, giudicata sulla
   frontiera accuracy contro compute. Dettagli in "Raccomandazione".
2. **I due metri.** Proporrei di riportare sempre insieme i savings certificati e l'accuracy a pari compute: in B2
   danno classifiche opposte.
3. **Il pilota B4 (soglie per exit certificate).** Proporrei di ridurlo alla misura già fatta in B1: il limite
   superiore è +0,017–0,019, sotto la soglia di 0,02 del piano in 5 setting su 8, e scala e bias fanno di più.
   **Aggiornato dopo B3**: la sua versione E1, la regola di uscita consapevole del costo, con uno score allenato
   solo sui dati puliti guadagna +0,009–0,014 sotto shift senza alcun costo. Quella parte va tenuta e misurata anche
   sul pulito.
4. **Il pilota B3 (riconoscere gli input che concordano).** Proporrei di farlo prima della Fase C: costa 30 minuti ed
   è lì che B1 mette circa il 90% del margine. Mi aspetto poco, e lo scriverei prima.
5. **Tracciamento.** Testi per `docs/findings.md` in `03_tracciamento_proposto.md`, righe per il log del documento di
   progetto in `03_righe_log_documento.txt` (controllo dei caratteri vietati passato). Non ho scritto nulla nei file
   della repository né nel documento online.
6. **Novità.** Per la fase di adattamento sui logit non ho fatto una ricerca in letteratura. Va fatta prima di
   presentarla come contributo (vedi "Revisore 2").
7. **Aperte dal Checkpoint 2b.** Se e come committare (nulla è committato: README e findings modificati nella Fase A,
   cartella `scripts/6_block3_review/` e test nuovi non tracciati); le righe per il log e le sei frasi superate del
   documento online; le review di ICLR 2027 dal tuo browser.
8. **Cache.** 12 GB di logit e feature con la BN adattata in `pilots/B2_adattamento/cache_bn/`, usati per l'ultima
   volta in B2.03. Si rifanno in 11 minuti. Li cancello?

## Cosa è stato fatto

| Pilota | Cosa misura | Casi | Compute | Verifica |
|---|---|---|---|---|
| B1 | Dove si perde il margine tra ciò che la rete potrebbe risparmiare e ciò che M2 certifica | 24 reti, pulito e 15 corruzioni a severity 1, 3, 5, 5 split | 12 minuti, CPU | M2 uguale a `results.csv` su 5.520 righe; ricalcolo separato su 16 casi, differenza 0 |
| B2 | Quale adattamento recupera savings al netto del suo compute | 4 reti (seed 0, joint e joint_sd), 15 corruzioni a severity 3 e 5, split 0 (37 configurazioni anche sullo split 1) | 11 minuti su MPS, 91 su CPU | M8b riprodotto al bit su 240 casi; ricalcolo separato su 6 casi, scarto massimo 0,0006 |
| B2.02 | Cosa fa la mappa affine sui logit | Gli stessi 120 casi | 5 minuti, CPU | Stessa misura di B2 |
| B2.03 | Le mappe sui logit su tutte le reti, e dopo la BN | 24 reti, 15 corruzioni a severity 3 e 5, split 0–2 (2.160 casi) | 25 minuti, CPU | M2 uguale a `results.csv` su 2.160 righe |

## Esiti in breve

### B1: dove sta il margine

Savings medi a severity 3 (ε = 0,05, 24 reti; `pilots/B1_bilancio/b1_summary.log`, `pilots/verifica_fable/b1_extra.log`):

| Livello | CIFAR10 | CIFAR100 |
|---|---|---|
| Oracolo per input, con budget ε | 0,718 | 0,510 |
| Oracolo per input, disaccordo zero | 0,677 | 0,466 |
| 8 soglie per exit, confronto equo | 0,349 | 0,151 |
| Soglia condivisa, scelta allo stesso modo | 0,329 | 0,134 |
| M2 con 3.000 righe | 0,311 | 0,123 |
| M2 con 250 righe | 0,262 | 0,096 |

- **Il prezzo statistico pesa il 2–4% del margine** (0,015 e 0,009).
- **Le soglie per exit chiudono il 5%**: +0,019 [0,012, 0,027] e +0,017 [0,014, 0,020] a pari rischio.
- **Il resto, circa il 90%, sta nel riconoscere quali input concordano** con l'exit 9: tra la migliore soglia
  condivisa e l'oracolo per input c'è il 96–98% del margine, e le soglie per exit ne prendono il 5%.
- **Sotto shift la rete perde poca capacità di uscire presto**: all'oracolo resta l'89% e l'81% dei savings puliti,
  a M2 il 59% e il 53%.
- **A parità di accuracy persa i savings non calano**: con il vincolo relativo resta 0,85 e 1,05 dei savings puliti.
  Alla λ di M2 su CIFAR100 il sistema è più accurato dell'exit 9 a ogni severity (+0,6–0,7 punti).
- **Savings netti di M2**: su 10.000 input conviene calibrare con 500 righe (0,271 contro 0,217 con 3.000 su
  CIFAR10 a severity 3); da 100.000 in su con 3.000.

### B2: quale adattamento ripaga (4 reti, seed 0)

Guadagno di savings su M2, split 0, 30 casi per cella, intervallo al 95% sulle corruzioni
(`pilots/verifica_fable/b2_extra.log`, `b2_mech_read.log`). M2 è a 0,342 e 0,150 a severity 3.

| Configurazione | CIFAR10, sev 3 | CIFAR100, sev 3 | Righe | Training (forward) |
|---|---|---|---|---|
| M8b com'è | +0,053 [0,043, 0,063] | +0,022 [0,015, 0,032] | 4.000 | 2.569 e 30.262 |
| Teste, 10 iterazioni, un solo L2 | +0,045 | +0,008 | 3.000 | 73 e 731 |
| Mappa affine sui logit, L2 scelto | +0,048 [0,040, 0,056] | +0,025 [0,017, 0,035] | 4.000 | 38 e 5.679 |
| Scala e bias per classe sui logit | +0,045 [0,039, 0,052] | +0,015 [0,013, 0,017] | 3.000 | 0,4 e 11 |
| Scala e bias per classe sui logit | +0,046 [0,039, 0,052] | +0,007 [0,003, 0,011] | 750 | 0,1 e 3 |
| Solo bias per classe | +0,031 | +0,010 | 3.000 | 0,2 e 4 |
| Solo una temperatura per testa | +0,019 | +0,002 | 3.000 | 0,05 e 0,6 |
| Mappa affine, 5 iterazioni, righe gratis | +0,023 | +0,003 | 0 | 0,3 e 44 |
| BN con 4.000 righe | −0,005 [−0,039, +0,031] | +0,003 [−0,008, +0,013] | 4.000 | 0 |
| BN e poi teste | +0,036 | +0,010 | 4.000 | 2.541 e 29.240 |

- **Quasi tutto il guadagno di M8b è una correzione lineare dei logit.** La mappa affine tiene il 91% su CIFAR10 e
  il 114% su CIFAR100. Su queste 4 reti scala e bias per classe tengono l'85% su CIFAR10 (l'86% con 750 righe, con
  un training ventimila volte più piccolo) e il 65% su CIFAR100.
- **Il secondo split conferma**: su 148 confronti il guadagno cambia in media di 0,003 e al più di 0,011; la
  correlazione di rango tra i due split va da 0,93 a 0,99.
- **A pari audit.** Con 750 righe per adattare e 2.250 per calibrare, invece di 3.000 per calibrare: +0,039–0,043
  su CIFAR10 a severity 3 senza un audit in più, cioè +0,027–0,030 netti già su 10.000 input. Su CIFAR100 +0,006.
- **Righe gratis** (input che arrivano comunque all'exit 9): rendono meno delle stesse righe prese a caso, in 22
  confronti su 24 fuori dal rumore. Con L2 scelto danno tra il 55% e il 95% del guadagno; con L2 fisso dal 10% a
  quanto le righe casuali. Costano quasi nulla.
- **BN adaptation**: accuracy dell'exit 9 +7,3 e +8,0 punti a severity 3, +16,2 e +12,2 a severity 5. Savings
  certificati fermi in media, −0,07 sotto rumore su CIFAR10. Con 500 righe il 93–98% del guadagno di accuracy.
- **I due metri danno classifiche opposte.**

| | Savings certificati, guadagno su M2 | Accuracy a pari compute di M2 |
|---|---|---|
| M8b, CIFAR10 sev 3 / sev 5 | +0,053 / +0,094 | +0,8 / 0,0 punti |
| M8b, CIFAR100 sev 3 / sev 5 | +0,022 / +0,053 | −0,2 / −0,3 punti |
| BN, CIFAR10 sev 3 / sev 5 | −0,005 / +0,018 | +7,7 / +16,3 punti |
| BN, CIFAR100 sev 3 / sev 5 | +0,003 / −0,004 | +8,0 / +12,5 punti |
| BN e poi teste, CIFAR10 | +0,036 / +0,058 | +8,4 / +16,6 punti |
| BN e poi teste, CIFAR100 | +0,010 / +0,006 | +7,8 / +12,3 punti |

- **Su CIFAR100 il guadagno delle teste è copia di errori.** Con M8b l'accuracy delle teste non sale (+0,0 e −0,6
  punti) e l'accordo sale di 4,9 e 9,8 punti. Tra i nuovi accordi il riferimento sbaglia nel 78% e 89% dei casi (92%
  e 97% sotto rumore). Certificando l'accuracy persa il guadagno sparisce o cambia segno (+0,014 [−0,006, +0,036] e
  −0,020). Su CIFAR10 a severity 3 no: l'accuracy delle teste sale di 2,8 punti e il guadagno resta.
- **Un riferimento degradato è più facile da imitare.** Per questo migliorare il riferimento con la BN abbassa i
  savings certificati sotto rumore, e BN e poi teste dà meno savings delle teste da sole (−0,017, −0,036, −0,012,
  −0,047).

### B2.03: le mappe sui logit su 24 reti

Guadagno su M2 e quota del guadagno di M8b, split 0–2, 540 casi per cella, intervallo al 95% a due vie su reti e
corruzioni (`pilots/verifica_fable/b2_mech24_read.log`, `b2_mech24_extra.log`).

| | CIFAR10, sev 3 | CIFAR10, sev 5 | CIFAR100, sev 3 | CIFAR100, sev 5 |
|---|---|---|---|---|
| M8b | +0,045 | +0,078 | +0,019 | +0,043 |
| Scala e bias, 3.000 righe | +0,034, 77% [64, 88] | +0,046, 59% [48, 72] | +0,014, 76% [53, 111] | +0,022, 51% [40, 64] |
| Scala e bias, 750 righe | 70% | 56% | 29% | 37% |
| Solo bias, 3.000 righe | 63% | 50% | 43% | 38% |
| Solo temperatura, 3.000 righe | 9% | 13% | 14% | 1% |

- **La raccomandazione regge su tre seed e quattro regimi.** Quota di scala e bias per seed a severity 3: 81%, 75%,
  74% su CIFAR10; 72%, 78%, 77% su CIFAR100. Guadagno positivo nel 92–97% dei casi.
- **Payback a severity 3**: circa 27.300 input su CIFAR10 e 28.900 su CIFAR100, contro 86.000 e 1,7 milioni di M8b.
  Training: 0,4 e 11 forward complete, contro 2.604 e 30.860. Con 750 righe 7.500 su CIFAR10 (70% del guadagno);
  su CIFAR100 750 righe non bastano (29%).
- **La garanzia regge.** Sopra ε nell'11,5%, 11,5%, 6,7%, 5,6% dei casi contro il 13,3%, 8,5%, 11,1%, 8,5% di M2
  sugli stessi casi; tutte le differenze dentro il rumore; sopra ε di più di due errori standard al più l'1,1%.
- **Sui due metri**: +0,75 punti di accuracy al costo di M2 su CIFAR10 a severity 3, −0,14 e −0,21 su CIFAR100.
- **La temperatura da sola non basta**: aiuta solo in joint_sd. Sulle 4 reti di B2 sembrava valere quanto le soglie
  per exit; era un confronto sbagliato (vedi "Errori").
- **BN e poi scala e bias** (seed 0, joint e joint_sd): +0,038 e +0,059 di savings su CIFAR10 con +8,4 e +17,0
  punti di accuracy; +0,012 e +0,004 su CIFAR100 con +8,0 e +12,5 punti.

### B3, dopo il checkpoint: traffico contro affidabilità, e gli score appresi

Svolto il 15 novembre su indicazione di Davide (`pilots/verifica_fable/b3_read_part1.log`, `b3_read_part2.log`;
voce B3.01 del quaderno). 24 reti; M2 coincide con `results.csv` su 720 righe.

- **Il calo ha due cause di peso simile.** Su CIFAR10, nei regimi joint e joint_sd a severity 3, l'affidabilità (a
  pari score gli exit concordano meno con l'exit 9) spiega il 37% [25%, 48%] e il 39% [30%, 47%] del calo nei due
  ordini; il traffico (gli exit perdono confidenza) il resto. Su CIFAR100 il 31% e il 47%. L'ipotesi C1 (almeno
  metà) è falsa. La quota dell'affidabilità cresce con la severity ed è massima sotto rumore.
- **A λ ferma il traffico non alza il rischio.** Alla λ pulita, sui dati shiftati, il rischio sale a 0,101 su
  CIFAR10 a severity 3; con gli score shiftati ma l'affidabilità pulita resterebbe a 0,047. La ricalibrazione corregge
  l'affidabilità; il traffico perso non lo recupera.
- **Gli score appresi riconoscono poco meglio gli input che concordano** (AUROC +0,005 a severity 3). Con la λ
  condivisa perdono contro la max softmax (−0,010/−0,015, come M8a). **Con la regola consapevole del costo** (si
  esce all'exit l se P(disaccordo) < μ·(c_9 − c_l)) guadagnano +0,019 su CIFAR10 e +0,016 su CIFAR100 se allenati
  sulle righe di adapt, e +0,008–0,014 se allenati solo sui dati puliti, cioè senza costo. L'ipotesi C2 (almeno
  0,03) è falsa.

## Le ipotesi

| Ipotesi | Esito | Numero che decide |
|---|---|---|
| B1: il salto tra oracolo per input e soglia condivisa vale almeno il 60% del margine | Vera | 96,4% e 97,7% |
| D1: una variante economica tiene il 70% del guadagno con il 10% del compute e payback sotto 30.000 su entrambi i dataset | Falsa sulle 4 reti di B2; sulle 24 reti scala e bias la rispettano al limite | 4 reti: 65% e 33.700 su CIFAR100. 24 reti: 77% e 76%, payback 27.300 e 28.900 |
| D2: BN e poi teste supera le teste di 0,02 | Falsa, segno opposto | −0,017 e −0,012 a severity 3 |
| D3: l'accordo sale molto più dell'accuracy e meno di metà del guadagno sopravvive con l'accuracy persa | Vera su CIFAR100 e sotto rumore, falsa su CIFAR10 a severity 3 | Vedi sopra |
| Kill di B2: nessuna variante tiene metà del guadagno con il 20% del costo | Non scatta | Su CIFAR100: 93% del guadagno con il 10% del costo |
| Kill di B4: le soglie per exit guadagnano meno di 0,02 nella maggior parte dei setting | Scatta sul limite superiore | 3 setting su 8 sopra 0,02 |
| C1: l'affidabilità spiega almeno metà del calo a severity 3 nei regimi congiunti su CIFAR10 | Falsa | 37% e 39% nei due ordini |
| C2: lo score con più exit recupera almeno 0,03 di savings sotto shift | Falsa | +0,019 e +0,016, e solo con la regola consapevole del costo |

Su D1 sulle 24 reti: scala e bias sono stati scelti dopo aver visto B2, quindi B2.03 è una replica su 20 reti nuove
di una scelta esplorativa, sempre sulle 15 corruzioni standard. Il payback sta appena sotto 30.000 e gli intervalli
delle quote scendono al 64% e al 53%. Non lo chiamerei una conferma.

Le previsioni. Per B1 ne hanno retto quattro su sei; una era troppo alta di 0,01 e quella sulle soglie per exit era
sbagliata. Per B2 e B2.02 ne ha retto circa metà; gli errori più grossi sono per difetto sulla mappa affine (prevedevo
50–70%, è il 91%) e sull'idea che le righe difficili fossero le migliori da cui imparare. Per B2.03 hanno retto quelle
sulle quote, sui seed e sulla BN; erano sbagliate quella sulla temperatura e quella su D1 su CIFAR100.

## Le direzioni, con i punteggi

Criteri e pesi della v2 del piano. Punteggi da 1 a 5, da discutere. Per il rischio, 5 vuol dire rischio basso.

| Direzione | Professore 25% | Impatto 20% | Novità verificata 20% | Fattibilità 15% | Rischio 10% | Tesi 10% | Totale |
|---|---|---|---|---|---|---|---|
| A. Il bilancio dei savings con due metri (B1, B2, poi online) | 5 | 3 | 3 | 5 | 4 | 4 | 4,00 |
| B. Fase di adattamento minima sui logit, con il bilancio netto | 5 | 3 | 2 | 5 | 4 | 4 | 3,80 |
| C. BN adaptation con ripartenza del certificato | 3 | 4 | 2 | 4 | 3 | 3 | 3,15 |
| D. Riconoscere gli input che concordano (pilota B3) | 3 | 4 | 2 | 4 | 2 | 3 | 3,05 |
| E. Soglie per exit certificate (pilota B4) | 3 | 2 | 2 | 4 | 2 | 2 | 2,55 |

- **A** è la domanda del professore e la Parte 2 non ha trovato lavori che la misurano. Il limite è la scala.
- **B** risponde alla seconda metà della frase del professore. Il rischio è sceso dopo la replica su 24 reti (3 → 4).
  La novità non è verificata: vedi sotto.
- **C** serve soprattutto a mostrare che i due metri divergono. Come metodo non è nuova.
- **D** ha il margine più grande sulla carta, ma i controlli fatti finora sullo score sono stati negativi.
- **E** ha un limite superiore piccolo, e scala e bias certificati con la UCB fanno di più su CIFAR10 (+0,034
  contro +0,019) e quasi altrettanto su CIFAR100 (+0,014 contro +0,017), su baseline non identiche.

## Revisore 2

- **"È CIFAR con una ResNet20."** Vero. B1 e le mappe sui logit hanno 24 reti e tre seed; le varianti sulle teste e
  la BN solo 4 reti del seed 0. Prima di scrivere servono la Fase D e, per le teste, i seed 1 e 2 (feature da
  ricalcolare, circa 20 secondi per corruzione).
- **"Che il disaccordo non misuri la qualità del task lo dice già la submission 22466."** Sì, come avvertenza su un
  caso. Qui c'è la misura: quanto compute costa, su 24 reti, e l'effetto opposto dei due adattamenti sui due metri.
- **"La mappa sui logit è il matrix scaling di Guo et al. 2017 con una distillazione."** È la critica più seria alla
  direzione B (l'attribuzione a Guo et al. è a memoria, da verificare). La Parte 2 ha trovato TinyTTA (adattamento
  delle sole teste, senza risk control né bilancio) e Kaplan 2026 (riadattare la testa vale poco contro ricalibrare
  la soglia, su un LLM): tutte e due note riportate da agenti. L'angolo che resta è il bilancio in compute dentro un
  risk control e la regola sul backbone fermo. Non ho cercato i lavori di test-time adaptation che agiscono solo
  sull'uscita: va fatto prima di ogni affermazione.
- **"La BN adaptation è nota, e Mehra et al. 2022 la usano già sui multi exit sotto CIFAR-C."** Vero. Non è un
  contributo: è il controllo che mostra perché i savings a consistency risk fissato non bastano.
- **"Gli oracoli sono irraggiungibili."** L'oracolo per input sì. Le 8 soglie no: sono una regola esplicita, e
  guadagnano 0,02.
- **"Il payback dipende da come si contano gli audit."** Sì. Per questo ci sono due conti (incrementale e pieno), la
  lettura a pari audit e i savings netti su tre orizzonti.
- **"Scala e bias li avete scelti guardando i risultati."** Vero: sono nati in B2.02, dopo B2. La replica B2.03 è su
  20 reti nuove ma sulle stesse corruzioni. La conferma vera chiede le 4 corruzioni extra e CIFAR-10.1, a metodo
  congelato (Fase D).
- **"Le righe gratis violano l'ipotesi della calibrazione."** Le teste dipendono dalle righe di cal attraverso la
  sola λ di M2. Sui dati non si vede (7,3% di casi sopra ε contro 8,3% di M2), ma una garanzia rigorosa chiede
  righe di cal nuove o una correzione. Da chiudere se la variante entra nella parte online.
- **"Sono corruzioni sintetiche."** Vero: lo shift naturale è nella Fase D.

## Raccomandazione per la fase di adattamento online

1. **Cosa adattare.** Una scala e un bias per classe sui logit di ogni testa, per distillazione verso l'exit 9.
   Backbone ed exit 9 fermi, quindi la condizione del bound resta valida e la mappa si fonde nel layer della testa
   senza costo a inferenza.
2. **Con quali righe.** 3.000 righe del budget degli audit. Su CIFAR10 bastano 750 (70% del guadagno di M8b contro
   77%), su CIFAR100 no. Le righe che arrivano comunque all'exit 9 come fonte in più, sapendo che rendono meno.
3. **Come ricalibrare.** Sui logit salvati degli audit già pagati: con scala e bias costa meno di un decimo di forward
   completa (con le teste intere costerebbe 3 forward su CIFAR10 e 32 su CIFAR100).
4. **Cosa non usare.** M8b com'è: su CIFAR100 a severity 3 non si ripaga nemmeno su un milione di input. La mappa
   affine piena con L2 fisso e poche righe su CIFAR100: fa danno. La sola temperatura: aiuta solo in joint_sd.
5. **Secondo braccio.** BN adaptation (500 righe danno il 93–98% del guadagno di accuracy), trattata come ripartenza
   del certificato (3.000 audit per ricalibrare) e seguita da scala e bias. Su CIFAR10 tiene l'85–90% del guadagno
   di savings di scala e bias da soli e aggiunge 8–17 punti di accuracy; su CIFAR100 a severity 5 perde quasi tutto
   il guadagno di savings.
6. **Metri.** Savings netti sugli orizzonti e accuracy a pari compute, sempre insieme.
7. **Per H5** (M5, M6, M7): B2 dà già il confronto offline tra righe gratis e audit a pari numero.

## Limiti

- Le varianti sulle teste e la BN: un seed, due regimi, uno split per la maggior parte delle configurazioni. Le mappe
  sui logit: 24 reti, tre split.
- Tutto offline: il dominio è noto e le righe sono i.i.d.
- La scelta di L2 = 0,01 per le varianti a L2 fisso era scritta prima dei risultati e su CIFAR100 non regge.
- B2.02 e B2.03 sono esplorazione aggiunta dopo aver visto B2, non conferma.
- Il costo di training di M8b nei payback viene da 3 domini del seed 0 joint (`adaptation_cost.csv`).

## Errori in questo giro

- **Confronto non equo in B1** (Fable). Le 8 soglie contro la soglia condivisa senza controllare il rischio su righe
  nuove: avrebbe dato +0,027 con 8 setting su 8 sopra 0,02. L'ha fatto notare la nota dell'esecutore; corretto prima
  del job completo.
- **Contabilità di B2** (Fable). La BN stimata dal traffico non era gratis e mancava la seconda calibrazione. Anche
  questo l'ha segnalato l'esecutore; corretto prima di guardare i risultati.
- **Confronto non equo in B2.02** (Fable). La temperatura su joint e joint_sd del seed 0 contro la media di B1 su
  tutti i regimi. Smentito da B2.03.
- **Frasi troppo forti o imprecise** (Fable), corrette da Opus nella revisione del 15 novembre, con una nota nel
  quaderno: le righe gratis (55–95% solo con L2 scelto); il margine che sta nel riconoscere gli input (circa 90%, non
  96–98%); "nulla batte M2 su 100.000 input su CIFAR100"; "nessun adattamento si ripaga sotto 100.000 input"; il
  costo di ricalibrare con scala e bias; il training ventimila volte più piccolo (solo con 750 righe); "con 750 righe
  conviene il solo bias".
- **Orari stimati nel quaderno** invece che letti: tre volte Fable, una volta Opus. Corretti.
- **Un errore nello script di sintesi** di B2 (i netti a pari audit di alcune configurazioni), trovato e corretto
  dopo il job.

## Comandi e file

Dalla radice della repository, con il Python dell'ambiente `eesavings`:

- B1: `python outputs/fable_review/pilots/B1_bilancio/b1_levels.py --workers 3`, poi `b1_summary.py`.
- B2: `zsh outputs/fable_review/pilots/B2_adattamento/run_b2_bn.sh 0`, poi `run_b2_heads.sh 3 --splits 0` (e
  `run_b2_heads.sh 2 --splits 1`), poi `b2_summary.py`; i seguiti con `b2_mech.py` e `b2_mech24.py --workers 3`.
- Verifiche e letture con codice separato: `outputs/fable_review/pilots/verifica_fable/` (`b1_check.py`,
  `b1_extra.py`, `b2_check.py`, `b2_extra.py`, `b2_mech_read.py`, `b2_mech24_read.py`, `b2_mech24_extra.py`, con i
  log accanto).
- Figure: `pilots/B1_bilancio/b1_levels.png`, `pilots/B2_adattamento/b2_frontier.png`,
  `pilots/B2_adattamento/b2_gain_vs_compute.png`.
