# Handoff ricerca: progetto Neural Networks con il prof. Scardapane (risk control per early exit)

Aggiornato al 31 ottobre 2026 (sezione 12 per le novità). Questo documento riassume tutta la conversazione fatta finora (contesto, messaggi del professore, idee, ricerche, verifiche, errori) per continuare il lavoro in Claude Code senza perdere nulla.

## 0. Come usare questo documento

- Metti questo file nella cartella del progetto (per esempio come `HANDOFF.md`, oppure copialo dentro `CLAUDE.md`) e scompatta accanto `research_bundle.zip`. I percorsi tra backtick si riferiscono alla cartella del bundle.
- Prima di lavorare leggi le sezioni 1, 2, 3 e 6. La sezione 4 è la mappa della letteratura, la 5 la lettura completa della submission ICLR 2027 numero 22466.
- Regola fondamentale: ogni affermazione di novità va verificata su fonti primarie (OpenReview API, arXiv nuove uscite, proceedings ICML, NeurIPS, ICLR) e scritta come "da quello che ho trovato al giorno X". I due errori fatti finora (sezione 9) vengono da ricerche che non avevano guardato OpenReview e ICML 2026.
- Strumenti che funzionano per cercare: WebFetch su `https://api2.openreview.net/notes/search?term=...&group=ICLR.cc/2027/Conference&limit=100` (anche con altri gruppi), Semantic Scholar Graph API (`https://api.semanticscholar.org/graph/v1/paper/arXiv:XXXX/citations?fields=title,year,venue,externalIds&limit=1000&offset=...`), alphaxiv.org e ar5iv per leggere arXiv quando arXiv risponde 429. Google Scholar e DBLP sono spesso bloccati. Le pagine forum di OpenReview hanno un controllo del browser: i PDF vanno scaricati a mano da un browser.

## 1. Contesto

### 1.1 Chi e cosa

- Davide De Blasio, laurea magistrale Engineering in Computer Science and AI alla Sapienza. Deve fare un progetto (research option) per il corso di Neural Networks del prof. Simone Scardapane. Il progetto è estendibile alla tesi magistrale, con l'obiettivo di una pubblicazione tra ottobre e dicembre 2027. Interesse di lungo periodo per la quant finance.
- Il gruppo di Scardapane lavora da anni su early exit e conditional computation (dettagli in 4.6). Il principale autore del gruppo sugli early exit è Jary Pomponi.

### 1.2 Cronologia

1. Metà ottobre 2026. Davide propone due idee.
   - Idea 1 (finanza): le reti che prevedono l'esito di un passive limit order (fill, cancellazione, adverse selection) restano affidabili sotto un regime shift? Dati Hyperliquid, crash del 10 ottobre 2025, confronto tra modello base, wallet history e temporal GNN sul grafo dei wallet.
   - Idea 2 (metodologica): una early exit network può accorgersi senza label che uno shift ha rotto il risk control di Fast yet Safe? Audit random sulla full network con probabilità nota, inverse probability weighting, sequential test anytime valid, fallback alla full network quando scatta l'allarme.
2. Primo messaggio del professore (verbatim): "Idealmente vanno entrambe bene (su early exit ci abbiano lavorato, non ho capito se anche questa sarebbe legata al campo applicativo? tra l'altro in questo caso avremmo una possibile collaborazione in Spagna). Posso chiederti un breve documento descrittivo su entrambe per capire meglio input, idee, ecc.? Privilegia la chiarezza e lo scriverlo tu sull'usare troppo Claude. :-)"
3. Intorno al 22 ottobre Davide manda un documento breve sulle due idee. Il testo è in Appendice A e nel file `verifiche/audit4/documento_per_il_professore.txt`. Quel documento diceva per l'idea 2 "Da quello che ho trovato nessuno lo fa così": oggi sappiamo che non è vero (sezione 9).
4. Risposta del professore del 28 ottobre (verbatim): "Ciao Davide, Il focus su finance potrebbe andare bene per la tesi, ma leggendo con calma forse lo eviterei per il progetto di neural networks, in quanto sposterebbe troppo tempo su dati, baseline non neurali, ecc. Potremmo perseguire la seconda opzione che si lega bene a quello che abbiamo fatto in passato per poi eventualmente specializzarla al dataset finanziario in seguito? Ci sono alcuni paper che fanno risk control su early exits, e.g. https://arxiv.org/html/2405.20915v2, puoi provare ad impostare nuovamente l'idea su queste linee ed eseguendo un check di letteratura più ampio?"
   - Nota: il link è proprio Fast yet Safe, la base dell'idea 2.
5. 28 ottobre, primo giro di ricerca (sei ricercatori in parallelo), report `reports/Risk control per early exit.md`. Scoperta principale: la submission ICLR 2027 numero 22466 copre già il monitoraggio con audit. Scritta una bozza di nuovo documento per il professore (Claude Docs, "Early exit con risk control sotto shift"), verificata da un revisore indipendente (`verifiche/audit5/verifica.md`) ma mai inviata e ormai superata.
6. 29 e 30 ottobre, secondo giro: sweep sistematico delle venue ML e applicate, alberi di citazione completi, controllo di novità mirato, lettura completa della submission 22466 (PDF fornito da Davide). Scoperte: B-PAC e SSV (ICML 2026) e Gibbs et al. (arXiv, 22 ottobre 2026) fanno già il controllo online della soglia con audit, fuori dagli early exit. Note in `research_notes/Check completo risk control early exit/`.
7. Stato al 30 ottobre: l'idea finale non è ancora fissata. La proposta corrente è nella sezione 6 e va sottoposta a un red team prima di scrivere il documento per il professore.

### 1.3 Preferenze di Davide e regole di stile per i testi al professore

Preferenze generali: risposte brevi e dirette; ogni affermazione va confermata con ricerche web prima di accettarla.

Regole di stile date da Davide per i documenti destinati al professore (da rispettare alla lettera):
- Voce attiva, frasi dirette e concise, linguaggio semplice e concreto, ritmo vario (frasi brevi e lunghe), tono naturale e conversazionale, realistico, niente marketing né frasi da AI ("è importante notare", "rivoluzionario", "ottimizzare", "aumentare", "non è solo, ma").
- NESSUN tipo di trattino (né "-", né en dash, né em dash), nessun due punti, nessun punto e virgola, nessun asterisco, nessun hashtag, nessuna emoji. Attenzione ai nomi con trattino: si scrive CIFAR10, CIFAR100C, Wild Time, CAPE Net, goal oriented.
- Preferire affermazioni definitive ai condizionali per i fatti, ma per i piani lo stile di Davide è il condizionale in prima persona ("vorrei", "farei", "allenerei"). Al professore si dà del "lei".
- Termini tecnici in inglese invece delle traduzioni: risk control (non "garanzia"), risk level, consistency risk, disagreement, audit (non "controlli"), detection delay, false alarm(s), false alarm control, full network, classifier, random, high probability, data stream, time to fill, wallet history, market context, nodes ed edges, price time priority, crash, inverse probability weighting.
- Citazioni: solo quelle essenziali, scritte come "Autore et al. anno" con link sul nome. Niente titoli di paper (contengono due punti e trattini).
- Il professore ha chiesto chiarezza e un testo scritto dallo studente, non "troppo Claude".

## 2. Sintesi in poche righe

- L'idea 2 originale (monitoraggio senza label con audit random e test anytime valid) è coperta dalla submission ICLR 2027 numero 22466 (pubblica su OpenReview dal 3 ottobre 2026).
- Anche il passo successivo che avevamo proposto (aggiornare online la soglia con gli audit, alla ACI, con garanzia sotto shift arbitrario) esiste già, ma fuori dagli early exit: B-PAC e SSV (ICML 2026) per il routing tra due modelli, Gibbs et al. (arXiv 2610.31662) per decisioni generiche con audit, CARE in versione i.i.d.
- Restano aperti, per le reti multi exit:
  1. una misura sistematica di quanto il risk control di Fast yet Safe si rompe sotto shift, in funzione di come sono stati allenati gli exit (joint, disjoint, mixed) e della calibrazione per exit;
  2. un confronto diretto tra i controller esistenti (Fast yet Safe statico, monitor e fallback della 22466, selezione alla B-PAC, controller alla SSV o Gibbs) su reti multi exit sotto corruption shift;
  3. l'allocazione degli audit quando il loro costo dipende dalla profondità dell'exit o dall'uplink (device e server): Gibbs et al. assumono audit a costo uguale, la 22466 dice che l'allocazione ottima degli audit resta aperta;
  4. soglie per exit controllate online (per esempio lungo un Pareto path offline), aperto ma incrementale.
- Proposta corrente (da validare con red team): misura (1) come novità principale e parte "neural", confronto (2) come benchmark, contributo di metodo su (3) nell'ambientazione device e server del gruppo, (4) come estensione. Finanza per la tesi.

## 3. Evoluzione dell'idea e stato della novità

| Formulazione | Stato al 30/10/2026 | Lavori che la coprono o la toccano |
|---|---|---|
| (a) Monitor senza label con audit random e test anytime valid (idea 2 originale) | Fatta | Submission ICLR 2027 numero 22466 (audit randomizzati, allarme LR e betting, confidence sequence sotto drift, escalation degli audit, fallback permanente) |
| (b) Controllo online della soglia con audit e pesi inversi, garanzia sotto shift arbitrario | Fatta come meccanismo, fuori dagli early exit | Gibbs, Han, Tibshirani, Zrnic 2026 (arXiv 2610.31662); SSV (ICML 2026); B-PAC (ICML 2026); CARE (arXiv 2610.09043, i.i.d.); ConformalDAgger (ICLR 2025, ACI con feedback intermittente e pesi inversi) |
| (b) applicata a reti multi exit con più di due exit | Aperta come applicazione, quasi diretta con una soglia condivisa | B-PAC scrive "extending it to multiple models remains unsolved"; la 22466 usa una cascata a 3 stadi ma solo per monitorare |
| (c) Covariate shift con importance weighting | Aperta ma debole | Per il consistency risk basta ricalibrare sugli input target eseguendo la full network |
| (d) Risk control per classe o per gruppo | Parziale | SAGE-CP, Ringel et al. (ICML 2024) |
| (e) Regressione e time series | Parziale | Fast yet Safe (segmentazione, generazione), Conformal Certification (arXiv 2610.19189), TRACE-CRC; niente in finanza |
| (f) Device e server con audit come offload forzato | Aperta come combinazione con risk control | HIL-N e HIL-F (TMLCN 2024) fanno offload forzati random pesati 1 su epsilon ma per il regret; Conformal Lyapunov Optimization (TSP 2025) fa online CRC con label a ogni passo e senza early exit |
| (g) Soglie per exit (vettore) online | Aperta ma incrementale | Offline: Pareto Testing (ICLR 2023), SAFE-KD, MultiRisk; la submission 22472 analizza soglie condivise contro per stadio, offline e senza garanzia |
| Misura della rottura di Fast yet Safe sotto shift per training regime | Nuova (nessuno l'ha pubblicata) | Kubaty et al. (ICML 2025) confrontano i regimi ma senza shift né agreement; Rethinking Calibration (ICML 2026) ha uno shift DomainNet ma senza agreement né risk control; Mehra et al. 2022 mostrano che l'inconsistenza tra exit cresce sotto corruzione |
| Allocazione degli audit con costo variabile (profondità o uplink) | Aperta nel contesto early exit | Gibbs et al. lo indicano come lavoro futuro; la 22466 dice "optimal audit allocation remains open"; Xu et al. (NeurIPS 2024) danno l'allocazione ottima a costo uguale |

## 4. Mappa della letteratura

Legenda: [PR] peer reviewed con venue, [pre] preprint, [sub] submission sotto revisione. Le note complete con citazioni testuali sono in `research_notes/` (vedi Appendice C).

### 4.1 Risk control e garanzie offline sugli early exit (la linea indicata dal professore)

| Lavoro | Cosa garantisce | Label | Ipotesi | Link |
|---|---|---|---|---|
| CATs, Schuster, Fisch, Jaakkola, Barzilay [PR EMNLP 2021] | Consistenza con la full network almeno 1 meno epsilon, conformal | No | Scambiabilità | https://aclanthology.org/2021.emnlp-main.406 |
| CALM, Schuster et al. [PR NeurIPS 2022] | Learn then Test su consistenza testuale o di rischio, token per token | Solo per la variante di rischio | i.i.d. | https://arxiv.org/abs/2207.07061 |
| Pareto Testing, Laufer-Goldshtein, Fisch, Barzilay, Jaakkola [PR ICLR 2023] | LTT su configurazioni multidimensionali (early exit, token pruning, head pruning in BERT) | Sì | i.i.d., offline | https://ICLR.cc/virtual/2023/poster/10902 |
| Anytime classification con conditional monotonicity, Jazbec et al. [PR NeurIPS 2023] | Monotonicità condizionale delle previsioni lungo gli exit | | | (vedi note) |
| Fast yet Safe, Jazbec, Timans, Hadži Veljković, Sakmann, Zhang, Naesseth, Nalisnick [PR NeurIPS 2024] | Soglia condivisa calibrata con CRC (in aspettativa) o UCB WSR (alta probabilità, delta 0,1); quattro rischi (performance gap o consistency, su predizioni 0 1 o distribuzioni Brier) | No per il consistency risk | i.i.d. | https://arxiv.org/abs/2405.20915 |
| EENN-AVCS, nested prediction sets, Jazbec et al. [PR UAI 2024] | Insiemi annidati anytime valid lungo gli exit di un input, modello bayesiano | Iperparametri | Parametrica | https://proceedings.mlr.press/v244/jazbec24a.html |
| Ringel et al. [PR ICML 2024] | LTT sul gap di accuracy, exit nel tempo (early classification) | Sì | i.i.d. | https://proceedings.mlr.press/v235/ringel24a.html |
| UAT, Bajpai e Hanawal [PR NeurIPS 2025] | Bandit online sulla confidence; dichiara anche un limite sul rischio (Theorem 4.1) valido solo se un proxy di affidabilità allenato offline con label stima bene la correttezza; non dimostrato sotto shift | Sì per il proxy | Proxy accurato | https://arxiv.org/abs/2509.23666 |
| Wynn et al. [PR ICML 2026, PMLR 306] | LTT con loss relativa al modello zero shot, soglia condivisa | Sì | i.i.d. | https://arxiv.org/abs/2510.02480 |
| SAFE-KD [pre, feb 2026] | CRC sull'errore, una soglia per exit; errori per exit su CIFAR10C | Sì | Scambiabilità | https://arxiv.org/abs/2602.03043 |
| Selection-Safe Conformal Early Exit [pre 14 ott 2026, anche sub ICLR 2027] | Copertura conformal valida per ogni regola di stop | Sì | Scambiabilità | https://arxiv.org/abs/2610.20773 |
| SAGE-CP [pre, in revisione TMLR] | Copertura per rotta | Sì | Scambiabilità | https://www.alphaxiv.org/abs/2610.23864 |
| CAPE-Net [pre, ago 2026, Research Square] | LTT e Holm sulla precisione per classe, exit sul numero di pacchetti; CUSUM senza controllo dei false alarm | Sì | Split temporali | https://www.researchsquare.com/article/rs-10360175/latest.pdf |
| Compute-Adaptive Risk Control [PR IEEE Access 2026] | Certificazione offline del compute | | | abstract S2 (vedi `citation_trees.md`) |
| Adaptive Computation with Conformal Certification [pre, anche sub ICLR 2027] | Certificazione conformal del calcolo adattivo | | | https://arxiv.org/abs/2610.19189 |
| Time-Uniform Guarantees for Transformer Early Exit via Martingale Stopping [sub ICLR 2027] | Regola di stop per input con bound di Ville lungo i layer | | | OpenReview API (vedi note) |
| EoE-Safe [sub ICLR 2027] | Early exit con garanzie (solo abstract) | | | OpenReview API |
| PAC Bayes per early exit | Bound da 1,5 a 4 volte lasco | | | https://www.alphaxiv.org/abs/2604.15764 |

Altri citati nelle note: MultiRisk (soglie multiple offline, https://www.alphaxiv.org/abs/2512.24587), Active Spiking Perception, RACE-AIMC, Aegis-ES, CRC under non monotone losses (https://arxiv.org/abs/2604.01502, utile perché il rischio degli early exit non è sempre monotono nella soglia).

Fast yet Safe in dettaglio (base del progetto):
- Ogni input esce al primo exit la cui confidence supera una soglia lambda unica per tutti gli exit. Si sceglie la lambda ammissibile più bassa (più risparmio), con ripiego su lambda uguale a 1, cioè la full network, se nessuna è ammissibile. Il consistency risk sostituisce la label con l'output dell'ultimo exit, quindi la calibrazione chiede solo input.
- Esperimenti: ImageNet (MSDNet, DViT, L2W-DEN, Dynamic Perceiver), Cityscapes, CNN DM e SQuAD con CALM, CelebA, IWSLT (solo v2). Media su 100 split casuali calibration e test.
- v2 del 4 novembre 2024 (quella linkata dal professore): aggiunge uno studio di speculative decoding (BiLD) e una sezione Discussion che dice "A key limitation of our work is the reliance on a single shared exit threshold among layers". Frase esatta sui lavori futuri: "Lastly, relaxing the i.i.d assumption on calibration and test data could help extend risk-controlling EENNs to scenarios with test-time distribution shifts [86] or to online updating strategies [23]." Il riferimento [23] è il Rolling Risk Control di Feldman et al.
- Codice: https://github.com/metodj/RC-EENN (ultimo commit 10 giugno 2024, nessuna licenza, logit ImageNet precalcolati su Google Drive in formato pickle (logits, targets), nessun esperimento di shift).

### 4.2 Fermare il ragionamento degli LLM, cascate e routing con garanzie (lavori adiacenti)

- Ragionamento: Thought calibration [PR EMNLP 2025], Conformal Thinking [PR ICML 2026, https://arxiv.org/abs/2602.03814, ispirato a Jazbec et al. 2024], ORCA [PR COLM 2026, modalità senza label rispetto alla risposta a budget pieno], PAC Reasoning [PR NeurIPS 2026, https://arxiv.org/abs/2510.09133, senza label, offline], Statistical Early Stopping for Reasoning Models [PR ICML 2026], Knowing When to Quit [PR ICLR 2026], SAFER [PR ICLR 2026], Conformal Language Modeling [PR ICLR 2024], CES "Calibration-Free Early Stopping for Chain-of-Thought Reasoning via E-processes" (Lee, Han, Kim, KAIST) [pre 18 ott 2026, https://arxiv.org/abs/2610.25212, due e processes su un segnale per chunk, senza calibrazione]. Sub ICLR 2027: Supervise What You Certify, Risk Control for Iterative Reasoning, SANE, One Hit Detection. Preprint: LYNX, MARS, MiCP.
- Cascate e routing: Fisch et al. [PR ICLR 2021], Trust or Escalate [PR ICLR 2025, oral], C3PO [PR NeurIPS 2025, senza label con il modello più potente come riferimento, offline, https://openreview.net/forum?id=e4IlBqhbTO], Conformal Arbitrage [PR NeurIPS 2025], LEC e RACER [PR ICML 2026], Conformal Cascade [pre, https://www.alphaxiv.org/abs/2607.25018, scrive "covariate shift deployment needs online recalibration"], CAb conformal alignment edge cloud [pre, https://arxiv.org/abs/2510.17543, offline, lavoro futuro "anytime valid guarantees for sequential inputs"], CR2 [pre, https://arxiv.org/abs/2605.12001], CCPO [PR AAAI 2026, offline], Conformal LLM routing [ACL 2026 SRW, offline]. Sub ICLR 2027: CRISP, CopRoute, DCMR. Online senza garanzie sul rischio: Online Cascade Learning [PR ICML 2024, no regret], CCO [PR ICML 2026].
- Serving: KV cache eviction con LTT, Redline, CARE per VLA, ARC Decode (tutti offline).

### 4.3 Monitoraggio e controllo online con audit (i lavori più vicini all'idea)

**Submission ICLR 2027 numero 22466**, "Auditing Early Stopping after Deployment: Sequential Inference with Selective Feedback" (anonima, OpenReview id wbXdPogQnU, creata il 15 settembre 2026, pubblica dal 3 ottobre 2026). Letta per intero: vedi sezione 5.

**B-PAC**, "Anytime Safe PAC Efficient Reasoning", Yu, Zeng, Zhu, Huang, Zeng, Jing [PR ICML 2026, PMLR 306], https://arxiv.org/abs/2601.22446
- Routing tra un LLM che ragiona (thinking) e uno che non ragiona. Loss misurata rispetto al modello thinking, quindi senza label.
- Online: la loss si osserva solo quando si chiama il modello costoso, con probabilità nota, inclusa un'esplorazione random (0,7 nel warm up, poi 0,05). Inverse propensity weighting.
- Una test supermartingale per ogni soglia candidata, soglia riselezionata a ogni passo (la più alta che passa il test).
- Garanzia anytime P(per ogni t, R_t(u_t) minore o uguale epsilon) maggiore o uguale 1 meno alfa sotto query i.i.d. Per stream arbitrari (Thm 5.1) limita solo un rischio cumulativo pesato e per una regola meno efficiente; la regola consigliata è valida sotto shift solo empiricamente.
- Limite dichiarato: "extending it to multiple models remains unsolved". Non cita lavori sugli early exit. Gruppo SUSTech (PAC Reasoning, G PAC, C PAC, B PAC, HyPAC, RACER): il più probabile a estendere ai multi exit.

**SSV**, "When to Trust the Cheap Check", Kiyani, Noorani, Pappas, Hassani [PR ICML 2026, poster sul sito ICML, spotlight su OpenReview], https://arxiv.org/abs/2602.17633
- Due soglie adattate online alla ACI, aggiornate solo nei round in cui si interroga il verificatore forte, con importance weighting; si muovono in entrambe le direzioni.
- Bound ad alta probabilità a orizzonte fisso, senza ipotesi sullo stream. Errori normalizzati sull'esito del verificatore, non per input servito. Nessuna confidence sequence anytime.
- Il suo Thm 5.1 ha la stessa forma del bound che avevamo derivato per il nostro controller (sezione 6.3): quel bound va presentato come adattamento, non come risultato nuovo.

**Gibbs, Han, Tibshirani, Zrnic**, "Reliable AI-Assisted Decisions via Selective Auditing" [pre, 22 ott 2026], https://arxiv.org/abs/2610.31662 (testo completo su https://www.alphaxiv.org/abs/2610.31662)
- Una soglia di confidence che si muove online: sotto la soglia ogni decisione viene sottoposta ad audit, sopra la soglia si fa uno spot check con probabilità nota gamma, con pesi inversi.
- Garanzia principale sulla frazione di errori non sottoposti ad audit, in aspettativa, per stream avversari: |Err(T) meno alfa| minore o uguale (1/eta più 1/gamma)/n_T. Variante "total error" (sezione 5.1): errori non sottoposti ad audit per round sotto alfa più (1/eta più 1/gamma)/T, che è proprio il target del nostro controller, in aspettativa.
- Variante ottima primal dual su una griglia di soglie con tasso T alla meno 1/3 e lower bound (i.i.d.); la variante calibrata aggiorna ogni soglia della griglia da un solo audit.
- Limiti dichiarati: garanzie solo in aspettativa, auditor umano o "more capable AI verifier", nessun risparmio di compute, "all audits have the same cost". Lavori futuri: auditor con costi diversi, regole di audit con "more than a scalar confidence".
- Cita SSV ma non B-PAC, la 22466, HIL-N, ConformalDAgger.

**CARE** "Careful Judge" [pre], https://arxiv.org/abs/2610.09043: acquisizione random delle label con bound sul rischio per passo, stream i.i.d.
**Audited Selective Verification** [pre], https://arxiv.org/abs/2607.13221: audit random con un solver di riferimento completo, validità dichiarata sotto shift arbitrario (screening di sistemi elettrici).
**ConformalDAgger**, Zhao et al. [PR ICLR 2025], https://arxiv.org/abs/2410.08852: ACI con feedback intermittente a probabilità nota e pesi inversi, bound di lungo periodo che cresce con max gamma/p.

Monitoraggio e strumenti sequenziali:
- Podkopaev e Ramdas [PR ICLR 2022], https://arxiv.org/abs/2110.06177: monitoraggio del rischio di un modello in produzione, richiede label (anche in ritardo).
- Timans et al. [PR UAI 2025], https://proceedings.mlr.press/v286/timans25a.html: testing by betting, H0 E[z_t | F_{t-1}] minore o uguale epsilon per ogni t, Ville, ritardo O((log(1/delta) più T)/(lambda mu)) che cresce con il changepoint T; richiede label; "we leave the discussion on model updating, or safe adaptation, for future work". Codice https://github.com/alextimans/risk-monitor (AGPL, ultimo commit 25 marzo 2026).
- Schirmer, Jazbec, Naesseth, Nalisnick [PR NeurIPS 2025], https://arxiv.org/abs/2507.08721: incertezza come proxy della loss; "verifying Assumption 1 for a given loss proxy still requires a labeled test stream".
- Xu, Karampatziakis, Mineiro [PR NeurIPS 2024], https://arxiv.org/abs/2406.10490: active anytime valid RCPS, label interrogate con probabilità q(x) maggiore o uguale q_min e pesi inversi; politica ottima q* proporzionale a radice di E[R^2 | x] (costi uguali).
- PPRM, Zhang et al. [pre, accettato ICML 2026 secondo il commento arXiv], https://arxiv.org/abs/2602.02229: limiti anytime valid combinando label sintetiche e poche label vere.
- Amoukou et al. [PR NeurIPS 2024]: stimatore dell'errore allenato. Anytime valid CRC, Hultberg et al. 2026, https://arxiv.org/abs/2602.04364.
- e detector, Shin, Ramdas, Rinaldo (controllano il tempo medio al false alarm, ritardo uniforme nel changepoint). Martingale conformi senza parametri (ott 2026), https://arxiv.org/abs/2610.17343.
- Online risk control: ACI, Gibbs e Candès [PR NeurIPS 2021], https://arxiv.org/abs/2106.00170; DtACI [PR JMLR 2024]; conformal PID [PR NeurIPS 2023]; passi decrescenti [PR ICML 2024]; Rolling Risk Control, Feldman, Ringel, Bates, Romano [TMLR 2023 secondo un indice terzo], https://arxiv.org/abs/2205.09095. Tutti chiedono feedback a ogni passo.
- Sub ICLR 2027: Deployment Risk Control for Sequential Prediction Systems (desk rejected), Online Conformal Shielding (soglia da feedback parziale). Certified Adaptive Refresh [pre], https://arxiv.org/abs/2605.29139. DISCERN [pre], https://arxiv.org/abs/2609.17560. Mirror online CP con feedback intermittente [pre], https://arxiv.org/abs/2503.10345.
- Shift: weighted CP (Tibshirani et al. 2019), weighted CRC (Angelopoulos et al. ICLR 2024), Barber et al. (Annals of Statistics 2023), conditional conformal (Gibbs, Cherian, Candès), GDE e agreement on the line, Detectron (ICLR 2023).
- Vincoli di design segnalati: Blindfold (gli audit random sono non distorti solo se il sistema non sa quando è sottoposto ad audit; la latenza può rivelarlo); Doula e Mühlhäuser (cambiare impostazioni di inferenza come quantizzazione o temperatura rompe la scambiabilità).

### 4.4 Early exit sotto shift e adattamento online senza garanzie sul rischio

- Evidenze empiriche: lo shift muove gli exit in due direzioni opposte. Il consistency risk si scompone in somma su l di P(exit = l) per P(previsione dell'exit l diversa dall'ultimo exit | exit = l).
  - Mehra et al. [pre 2022], https://arxiv.org/abs/2212.01562: SDN su CIFAR10C e CIFAR100C, AugMix contro training standard, più underthinking e overthinking, costo dell'oracolo da 23,4% a 39,4% (VGG 16).
  - Apparate [PR SOSP 2024], https://arxiv.org/abs/2312.05385: con una sola taratura iniziale la concordanza con il modello completo scende a 84,5% (CV) e 86,8% (NLP) contro un obiettivo del 99%; manda tutti gli input fino in fondo, quindi non risparmia compute.
  - UAT su BLIP 2 con rumore: la baseline Fast yet Safe esce di più e il gap BLEU 4 sale da 1,5 a 3,7. CapEEN [PR Findings EMNLP 2024]: esce di meno sotto rumore ma il gap cresce.
  - ADEx [PR IFIP Networking 2026]: sotto drift di popolarità delle classi l'uso del ramo principale passa da 7,46% a 46,14%.
  - Rethinking Calibration for Early-Exit Neural Networks [PR ICML 2026, PMLR v306 kubaty26a, arXiv 2508.21495; la v1 era "Failure Prediction Is a Better Performance Proxy" con Jazbec, workshop NeurIPS 2025]: Appendice F con shift DomainNet su accuracy, ECE, EEFP; nessun agreement tra exit, nessun risk control, training regime non variato.
  - Nessun paper misura il consistency risk di una soglia Fast yet Safe calibrata su dati puliti al crescere della severity.
- Bandit e metodi online senza garanzie sul rischio: UEE-UCB [AIMLSystems 2022], AdaEE e UCBEE [ICC 2023], SplitEE e I-SplitEE, CeeBERT [PR Findings ACL 2024, https://arxiv.org/abs/2405.15039, solo regret], A-CapEEN. Dove la reward è documentata vedono l'ultimo layer solo per gli input non usciti, quindi un exit sicuro ma sbagliato non viene penalizzato (vale per CeeBERT; per UAT solo se lo segnala il proxy).
- Usano l'agreement con la full network ma senza garanzie: Apparate, FREE [PR EMNLP 2023], ADEx, LEV. Drift detector: CAPE-Net (CUSUM), ADEx (regola sulle frequenze).
- Training e adattamento: Kubaty et al. "How to Train Your Multi-Exit Model?" [PR ICML 2025, https://proceedings.mlr.press/v267/kubaty25a.html, con Pomponi]: confronta joint, disjoint, mixed (più branch wise, separate, alternating, mixed gradual in appendice), NON self distillation; nessuno shift, nessuna metrica di agreement, nessuna calibrazione o risk control. EENet [PR WACV 2024], JEI DNN, DDA [PR CVPR 2021], DAdEE [PR Findings EMNLP 2024, riusa la soglia della sorgente], MOOD [PR CVPR 2021], Meronen et al. [PR WACV 2024, ECE di MSDNet su CIFAR100 da 0,182 a 0,017].
- Sistemi senza garanzie: Tabi [EuroSys 2023], CEED [INFOCOM 2025], E3 [SenSys 2025], HELIOS [MLSys 2026].
- T-TAMER, Yang, Zhang, Morgenstern, Xu [PR ICLR 2026]: garanzie di approssimazione nel caso peggiore per regole di exit (non statistiche); versione workshop "Hold That Exit" (NeurIPS 2025 SPIGM). Decision centered teamwork in Dynamic Early Exiting Networks [workshop ICML 2026]: conformal solo per migliorare likelihood.

### 4.5 Edge e device server

- HIL-N e HIL-F, Moothedath, Champati, Gross [PR IEEE TMLCN vol. 2 pp. 280 297, 2024, DOI 10.1109/TMLCN.2024.3366501]: con probabilità epsilon il device fa offload anche quando accetterebbe la risposta locale; pesi 1/epsilon per una stima non distorta della loss; output del server come ground truth; regret O(n^{2/3}) che conta il costo degli audit. È il meccanismo "audit uguale offload forzato", usato per il regret e non per il risk control. Seguiti: Hedge-HI [PR MobiHoc 2024], H2T2 [PR AAAI 2026, https://arxiv.org/html/2509.15674v3, robustezza allo shift solo empirica].
- HI-LCB [pre, https://arxiv.org/abs/2508.08985]: soglia di offload online con regret O(log T), senza offload forzati, assume accuracy crescente con la confidence. "Learning to Offload with Low Regret under Asymmetric Misclassification Costs", Agarwal, Sutar, Moharir [workshop ICML 2026 AI4NextG, https://icml.cc/virtual/2026/78459].
- Conformal Lyapunov Optimization, Binucci, Simeone, Banelli [PR IEEE TSP vol. 73 pp. 4521 4538, 2025, https://arxiv.org/abs/2503.00486]: online CRC più Lyapunov nell'inferenza edge gerarchica, garanzia di lungo periodo deterministica, ma label vere a ogni passo e niente early exit. Versione multiutente [PR IEEE FMEC 2025].
- Gruppo Simeone: SpikeCP [PR IEEE JSTSP], neuromorphic split computing con LTT [PR IEEE TSP 2024], OCP-PLS [PR PMLR 271, 2025, https://proceedings.mlr.press/v271/hou25a.html, online CP con feedback occasionale dal cloud], C-SQS [workshop NeurIPS 2025 AI4NextG, aggiornamento conformal online senza label per speculative decoding edge cloud], CD-CRC [pre].
- RACE [PR IEEE Trans. Services Computing, set 2026]: budget di hallucination edge cloud online, senza early exit, full text a pagamento.
- Pacheco, Couto et al.: calibrazione e offloading (GLOBECOM 2021, ICC 2021, ITU J FET 2024, JNCA 2023), nessuna garanzia.

### 4.6 Il gruppo di Scardapane

- Early exit (Pomponi su quattro di questi):
  - 2020: "Differentiable branching in deep networks for fast inference" (ICASSP); "Why should we add early exits to neural networks?" (Cognitive Computation, https://arxiv.org/abs/2004.12814, cita la "guaranteed accuracy of the intermediate predictions" come tema aperto); early exit per fog IoT (Information Sciences).
  - 2021: rete multi exit calibrata per cellule uroteliali (focal loss e temperature scaling; aggiungere exit spesso peggiora l'ECE).
  - 2022: confidence nei modelli multi exit (Entropy, Pomponi).
  - 2024: early exit ricorsivi per goal oriented communications, Pomponi, Merluzzi, Devoto, Pontes Mota, Di Lorenzo, Scardapane (Asilomar 2024 secondo la homepage, https://arxiv.org/abs/2412.19587). Exit sul device e su un server "that can take over computation whenever needed", offloading con Q learning su canale Rayleigh, ResNet 20 con 9 exit su CIFAR10, "accuracy cannot be directly assessed", margin come proxy, obiettivo "guaranteeing a minimum target level of performance" senza garanzia formale. È il lavoro più vicino al progetto.
  - 2025: NACHOS (IEEE TNNLS); Kubaty et al. ICML 2025 (Pomponi coautore, non Scardapane).
  - 2026: survey sulle dynamic neural network (Image and Vision Computing, https://arxiv.org/abs/2501.07451: "The domain of Early Exits lacks a standardized experimental protocol").
- Conditional computation: survey 2024 (https://arxiv.org/abs/2403.07965: controllare il tempo di inferenza è "still an open challenge", split computing e semantic communication come campi), LRCC (https://arxiv.org/abs/2610.08858), D2DMoE (NeurIPS 2024), Adaptive Computation Modules (AAAI 2025), activation sparsity (ICLR 2026), token e layer selection (GLOBECOM 2024, IEEE TMLCN 2026, Neurocomputing 2025, Neural Networks 2025).
- Nessun lavoro del gruppo usa conformal prediction o risk control. La linea Di Lorenzo e Merluzzi ha "reliability guarantees" come vincoli medi di lungo periodo alla Lyapunov su proxy come l'entropia, con ipotesi verso l'accuracy che "are not formally proved". Nessuna tesi del gruppo sugli early exit nell'archivio.
- Progetti: CENTS (2024 2026, PI Scardapane, interno Sapienza), 6G GOALS (EU, fino al 31 dicembre 2026, PI Di Lorenzo, Scardapane task leader; la visione chiede robustezza "with respect to possible distribution shifts").
- Ponti: Wójcik e Szatkowski (coautori di Scardapane) e Kubaty (coautore di Pomponi) hanno firmato con Jazbec (primo autore di Fast yet Safe) il lavoro sulla failure prediction per early exit. Binucci e Banelli (Conformal Lyapunov Optimization) hanno firmato con Di Lorenzo e Merluzzi un lavoro del 2023; il poster di dottorato di Binucci elenca "reliability guarantees via adaptive conformal prediction" come lavoro pianificato.
- Spagna: la collaborazione citata dal professore non è identificabile da fonti pubbliche. Link documentati: talk di Scardapane al Computer Vision Center (UAB Barcellona, 2023) e un paper ICML 2025 sugli early exit nel continual learning con Szatkowski; Soret (Málaga) e Kountouris (Granada) nell'orbita 6G GOALS; tesi del 2022 con Universitat de Barcelona. Va chiesto al professore.

### 4.7 Strumenti statistici di base

Learn then Test (Angelopoulos et al., https://arxiv.org/abs/2110.01052), Conformal Risk Control (Angelopoulos et al., ICLR 2024, https://arxiv.org/abs/2208.02814), RCPS (Bates et al. 2021, JACM), confidence sequences (Howard et al. 2021; Waudby-Smith e Ramdas 2024, aGRAPA e ONS), active statistical inference con AIPW (Zrnić e Candès 2024), on off observation control (Banerjee e Veeravalli 2012 e 2013), RouteCert (Baghi 2026), disuguaglianza di Freedman per martingale con varianza limitata.

## 5. La submission ICLR 2027 numero 22466 letta per intero

Testo estratto in `submission_22466/paper.txt` (65 pagine, 9 di corpo più appendici). Punti chiave:
- Setting (sezione 3): per ogni soggetto il sistema osserva stadi x_{i,1..Rmax}; una regola pi_lambda propone uno stadio di consegna d_i; il riferimento è l'output a budget pieno y_{i,Rmax}. Loss L_i(lambda) = 1[rho(y_{i,d}, y_{i,Rmax}) maggiore di m], zero se non si ferma prima. Candidati ordinati lambda(0) (never stop) prima di tutti. La regola deployata lambda_i è F_{i-1} misurabile (può cambiare nel tempo). Rischio condizionato R_i(lambda) e rischio cumulativo medio R̄_t^dep. Per il disagreement esatto, R_i limita anche l'errore di task in eccesso rispetto al riferimento, non quello assoluto.
- Proposition 1: senza audit alcune violazioni sono indistinguibili da deployment che rispettano il target (stessa legge dei dati osservati).
- Audit (4.1): A_i ~ Bernoulli(p_i), p_i H_i misurabile in [p_min, 1]; se A_i = 1 si completa e si consegna l'output completo (l'audit è anche un intervento); per le completion naturali A_i = p_i = 1. "A completed trajectory also reveals losses of fixed candidates whose decisions and outputs can be replayed from it" (un audit rivela la loss di ogni candidato). Audit uniformi a tasso p conservano una frazione 1 meno p del risparmio atteso (Proposition 4).
- Proposition 2: lower bound informativo per shift silenziosi, n maggiore o uguale kl(1 meno beta, delta)/(p kl(r, alfa)).
- Theorem 1: allarme likelihood ratio su una griglia {0,1; 0,2; 0,4; 0,8; 0,95} con alfa 0,05, restart, delay E[(tau meno s')+] minore o uguale (log(G/(delta w')) più log(r_g/alfa))/(p I_g).
- 4.3 e Theorem 2: incrementi AIPW Z_i = f_i più (A_i/p_i)(L_i meno f_i), con f_i predizione della loss (di default 1 meno la max softmax, oppure calibrazione isotonica online); confidence sequence empirical Bernstein sul rischio cumulativo valida sotto drift arbitrario e sotto aggiornamenti predicibili di lambda_i e p_i. Allarme betting opzionale (nessun bound sul delay).
- 4.4: allocazione degli audit guidata dall'evidenza (smooth escalation p = p_floor più (p_max meno p_floor) phi(e); variante two threshold).
- 4.5 e Algoritmo 1 e 2: allarmi LR e betting a delta/2 ciascuno; quando un allarme scatta, fallback permanente (latched) alla regola never stop; altrimenti si tiene lambda_i oppure si usa il "stationary selector" opzionale (Theorem 3): sceglie l'ultimo candidato la cui upper CS e quelle di tutti i precedenti sono sotto alfa, valido solo se i soggetti sono i.i.d.
- Esperimenti: cascata pre allenata su ImageNet valutata su Imagenette (ResNet 18, ResNet 50, ConvNeXt Base, exit con max softmax maggiore o uguale 0,7), MRI low field M4Raw, Qwen2.5 Math 1.5B su GSM8K e MATH 500 con riferimento la plurality di 64 campioni. Stream da 3000 (6000 per LLM) con cambio a 1500, alfa = delta = 0,05. Cambi: risoluzione 160 e 128 pixel, sostituzione con AlexNet. Tabella 1 (128 pixel, costo extra 0,03): smooth escalation con entrambi gli allarmi ha disagreement post cambio 0,076, restricted mean delay 825, rilevati 90 su 100, costo post 0,665.
- Discussione e limiti: nessuna risposta uniformemente migliore; violazioni piccole possono sfuggire; l'allarme betting non ha bound sul delay; "optimal audit allocation remains open"; la qualità del riferimento va valutata a parte.
- Cosa NON fa: non adatta la soglia sotto shift con una garanzia (il selettore vale solo i.i.d. e il fallback è permanente), nessun recupero di compute dopo il cambio, nessun costo degli audit dipendente dalla profondità o dalla comunicazione, nessun training degli exit, nessuna rete multi exit vera (usa una cascata di tre modelli diversi), non cita B-PAC né SSV.
- Probabilmente degli stessi autori: submission 22472 "What Drives Early Stopping Gains? Threshold Policies and an MRI Case Study" (creata 5 minuti dopo), analisi offline di soglie condivise contro per stadio, senza garanzia.

## 6. Proposta corrente per l'idea finale (da sottoporre a red team)

### 6.1 Titolo di lavoro

"Online risk control per reti multi exit sotto shift, con audit a costo variabile" (in inglese: "Cost aware audits for online risk control of multi exit networks under distribution shift").

### 6.2 Componenti

1. Misura (novità principale e parte "neural" del progetto). Allenare reti multi exit su CIFAR100 con regimi diversi (joint, disjoint, mixed come in Kubaty et al. 2025, con Pomponi) e con o senza calibrazione per exit (temperature scaling), calibrare la soglia con Fast yet Safe (UCB, non CRC) su dati puliti e misurare il consistency risk realizzato su CIFAR100C al crescere della severity (e CIFAR10C, CIFAR10.1 per la variante CIFAR10). Scomporre il rischio in distribuzione degli exit e disagreement condizionato. Domanda: quale training rende il risk control più robusto allo shift? Nessuno l'ha pubblicato al 30 ottobre 2026.
2. Benchmark (trasferimento, non novità). Confronto sugli stream multi exit: Fast yet Safe statico; monitor della 22466 con fallback permanente ed escalation; selezione anytime alla B-PAC (i.i.d.); controller alla SSV o Gibbs (ACI con pesi inversi, bidirezionale); un bandit alla UAT o CeeBERT. La confidence sequence della 22466 (Theorem 2) fa da certificato per qualunque controller predicibile.
3. Contributo di metodo (aperto). Allocazione degli audit con costo variabile: in una rete multi exit completare un input uscito all'exit l costa i blocchi rimanenti (più costa se esce presto); nel device server costa anche l'uplink. Probabilità di audit p(x) dipendente da confidence e costo, mantenendo la validità dei pesi inversi o AIPW, per minimizzare la varianza della stima del rischio (e quindi il ritardo o il margine del controller) a parità di budget. Gibbs et al. indicano gli auditor con costi diversi come lavoro futuro; la 22466 dice che l'allocazione ottima resta aperta.
4. Estensione opzionale. Soglie per exit lungo un Pareto path: offline si costruisce un cammino di vettori di soglie che finisce nella full network; online un controller unidimensionale si muove sul cammino; poiché un audit rivela la loss di ogni punto del cammino, il cammino si può anche riadattare online. La garanzia di lungo periodo resta valida per qualunque cammino predicibile che finisca nella full network (derivazione nostra da verificare). Risponde al limite dichiarato di Fast yet Safe (soglia unica).
5. Ambientazione del gruppo. Versione device e server degli early exit ricorsivi (Pomponi et al. 2024): audit uguale offload forzato, costo uplink. Da posizionare rispetto a HIL-N (regret), Conformal Lyapunov Optimization (label a ogni passo) e B-PAC o SSV (due modelli, nessun costo di comunicazione).
6. Tesi. Finanza (sezione 8).

### 6.3 Matematica disponibile (stato: derivazioni nostre, da verificare)

- Controller alla ACI con audit: lambda_{t+1} = max(0, lambda_t più gamma (B_t z_t/pi meno epsilon)), con z_t = 1[exit usato con soglia lambda_t in disagreement con l'ultimo exit], B_t ~ Bernoulli(pi) indipendente dall'input e dal passato, confidence minore o uguale a 1 (quindi con lambda maggiore o uguale a 1 nessun exit scatta e z = 0). Deterministicamente (1/T) somma B_t z_t/pi minore o uguale epsilon più (1 più gamma/pi)/(gamma T). Con Freedman, con probabilità almeno 1 meno delta a T fisso, (1/T) somma z_t minore o uguale epsilon più (1 più gamma/pi)/(gamma T) più radice(2 (1 meno pi) log(1/delta)/(pi T)) più 2 log(1/delta)/(3T). Verificato da un revisore indipendente anche con simulazione su tre avversari non anticipanti (eccesso al 95% per radice di pi T tra 0,2 e 0,6). Serve Freedman (Azuma dà 1/pi per radice di T). Ha la stessa forma del Thm 5.1 di SSV e della sezione 5.1 di Gibbs et al. (in aspettativa): va presentato come adattamento.
- Con pi = 0,05 e delta = 0,1 il termine dei pesi inversi vale circa 3 punti percentuali a T = 10^5 e circa 1 a T = 10^6: ridurre la varianza (control variate AIPW con la confidence, audit non uniformi) è necessario.
- Il controller non garantisce il rischio istante per istante; un monitor sopra (22466 o Timans) va tarato su un livello un po' sopra il target del controller, perché il controller lascia il rischio condizionato sopra il target per parte del tempo.
- Allocazione con costi (ipotesi da verificare): minimizzare E[q(x)/p(x)] con vincolo E[p(x) c(x)] minore o uguale B, dove q(x) è la probabilità di disagreement prevista e c(x) il costo dell'audit, dà p*(x) proporzionale a radice di q(x)/c(x), troncata in [p_min, 1]. Con costi uguali si ritrova la forma di Xu et al. Verificare se esiste già in active inference con costi.
- Pareto path: per un cammino predicibile con estremo sicuro (full network) il bound di lungo periodo si applica all'indice del cammino. Da dimostrare con cura.

### 6.4 Come scrivere la novità (onesto)

"Il risk control offline sugli early exit esiste (CATs, CALM, Fast yet Safe). Il monitoraggio con audit è nella submission 22466. Il controllo online con audit esiste per due modelli e per decisioni generiche (B-PAC, SSV, Gibbs et al. 2026). Da quello che ho trovato al 30 ottobre 2026 nessuno misura quanto il risk control di Fast yet Safe si rompe sotto shift in funzione del training degli exit, nessuno confronta questi controller su reti multi exit, e nessuno tratta audit con costo che dipende dalla profondità o dall'uplink."

Da citare per forza: Fast yet Safe, 22466, B-PAC, SSV, Gibbs et al. 2026, HIL-N, Conformal Lyapunov Optimization, Kubaty et al. 2025, Pomponi et al. 2024, Timans et al. 2025.

### 6.5 Rischi della proposta

- La parte di metodo potrebbe essere vista come trasferimento di strumenti noti (soprattutto se l'allocazione con costi esiste già in active inference).
- Il gruppo SUSTech (B-PAC) e il gruppo di Nalisnick (Fast yet Safe, Timans, Schirmer) hanno tutti gli ingredienti per estendere ai multi exit: conviene muoversi in fretta o puntare su ciò che loro non fanno (training regime, costi, device server).
- Su CIFAR10 il consistency risk potrebbe essere troppo piccolo: per questo il nucleo è CIFAR100.
- Consistency non è accuracy: se lo shift peggiora la full network servono label (secondo livello, utile anche per la tesi).

## 7. Piano sperimentale, codice e dati

- Piano di circa 10 settimane su una GPU (Colab o simile):
  1. Settimana 1: rilettura di 22466, Gibbs et al., SSV, B-PAC; riproduzione di Fast yet Safe sui logit ImageNet pubblici con il consistency risk (CPU, minuti).
  2. Settimane 2 e 3: reti multi exit su CIFAR100 con i regimi di training (MSDNet-PyTorch, 7 exit, 300 epoche; oppure teste SDN su ResNet congelata, 25 epoche; oppure ResNet 20 a 9 exit come nel lavoro ricorsivo). Stime GPU da meno di un'ora a qualche ora, da confermare con un run di un'epoca.
  3. Settimana 4: argmax e confidence per exit su test pulito e CIFAR100C (corruzioni per 5 severità).
  4. Settimana 5: studio offline, consistency risk per corruzione, severità e regime, scomposto.
  5. Settimane 6 a 8: stream alla Timans (nessuno shift, shift brusco presto e tardi, gradini, rampa di severità, regimi markoviani), tasso di audit da 0,01 a 0,2, da 50 a 100 ripetizioni, tutti i controller, allocazione uniforme contro con costi (CPU).
  6. Settimana 9: variante device server con costo uplink (parametri di canale dalla letteratura, per esempio Pacheco: 93 Mbps e 12 ms di RTT contro 42 Mbps e 213 ms).
  7. Settimana 10: scrittura.
- Metriche: consistency risk realizzato nel tempo (calcolato offline eseguendo la full network su tutto), media di lungo periodo e finestra peggiore rispetto a epsilon, compute risparmiato al netto degli audit, false alarm e detection delay per i metodi con monitor, accuracy del sistema accanto a quella della full network.
- Risorse:
  - RC-EENN (Fast yet Safe): https://github.com/metodj/RC-EENN, formato pickle (logits, targets).
  - risk-monitor (Timans): https://github.com/alextimans/risk-monitor, stream da predizioni in cache, T = 1500, 50 trial, finestre 0, 10, 50, 200, 365, scommesse aGRAPA con clipping 0,5/epsilon, burn in 100.
  - Codice supplementare anonimo della 22466 su OpenReview (simulatore, metodi, script).
  - MSDNet-PyTorch (MIT), Shallow-Deep Networks (MIT), airbench (ResNet 18 su CIFAR10 in 7 minuti su A100), AnytimeClassification (AGPL, checkpoint MSDNet CIFAR su Drive da verificare), L2W-DEN, Dynamic Perceiver, DVT, DeeBERT e PABEE, CALM t5x, LayerSkip.
  - Dati: CIFAR10C (Zenodo 2535967, CC BY 4.0, 2,9 GB), CIFAR100C (Zenodo 3555552), CIFAR10.1 v6 (2000 immagini), ImageNet-C (66 GB, Zenodo 2235448), ImageNet-V2, -R, -Sketch, Wild Time (Yearbook, FMoW-Time), DomainNet.
- Simulazioni sintetiche già fatte (`verifiche/audit2/a4_metodo_idea2.md`, `a4_sim_idea2.py`): calibrare con UCB invece che CRC (con CRC P(R maggiore di epsilon) 0,37 contro 0,07, e falsi allarmi 13,6% contro 2,8% in 5000 audit senza shift); con scommesse su tutta la storia il ritardo dopo uno shift tardivo ha un pavimento che non scende con più audit (27.413 passi con 5% di audit contro 16.994 con 100%), le finestre mobili lo tolgono; ricalibrare sugli ultimi 1000 audit prima dell'allarme porta P(R_post maggiore di epsilon) a 0,70 contro 0,049 con audit freschi.

## 8. Estensione per la tesi: finanza (idea 1, solo contesto)

- Report completo dell'idea 1: `reports/Ordini limite ed early exit.md` e note in `research_notes/Ordini limite ed early exit/`.
- Dati: dataset Zenodo "An Open Book" di Albers, Cucuringu, Howison, Shestopaloff (CC BY 4.0, 195,3 GB, https://zenodo.org/records/18184441): order status e book diffs solo per dicembre 2025 su BTC, ETH, SOL; per ottobre 2025 solo i trade (oltre 250 perpetual), con buchi documentati nel README (ottobre 2 ore mancanti, novembre 10 ore, gennaio 1 ora, ore non specificate). Fill gratuiti con indirizzo, flag crossed (taker), liquidazioni e ADL da Hydromancer Reservoir dal 28 luglio 2025 (requester pays).
- Shift naturale: crash del 10 ottobre 2025, 19,13 miliardi di dollari liquidati in 24 ore (CoinGlass via Decrypt); su Hyperliquid picco tra 21:16 e 21:27 UTC con circa 2,1 miliardi di liquidazioni e circa 16 round di ADL (arXiv 2602.15182). Da controllare se le 2 ore mancanti di ottobre cadono la sera del 10.
- Letteratura idea 1: Arroyo et al. 2024 (fill probability con NN), Fabre e Ragel 2023, KANFormer (Zhong et al. 2025), Cartea et al. 2026 (toxic trades di un broker FX), Zhai 2026 e Barone e Lillo 2026 (Hyperliquid), Albers et al. 2025 (fill contro rendimento su Binance), DeepLOB, LOBCAST ("all models exhibit a significant performance drop when exposed to new data"), LightTrader (DeepLOB in 296 microsecondi).
- Legame con il progetto: lo stesso controller ha una garanzia di lungo periodo per qualunque sequenza, mentre la calibrazione i.i.d. di Fast yet Safe non regge (simulazione AR(1) con rho 0,99: P(R maggiore di epsilon) 0,33 contro 0,1 nominale) e il monitor di Timans dà allarmi con regimi persistenti (26% e 58,5% degli stream). I markout arrivano in ritardo ma il consistency risk non li richiede: servono solo per il secondo livello sull'accuracy. Una classe rara come l'adverse selection può restare nascosta in un rischio 0 1 piccolo: nella tesi serve controllo per classe. Nessun early exit in profondità trovato per order book o serie finanziarie.

## 9. Errori già fatti e correzioni da applicare

- Il controllo di novità del 21 ottobre aveva detto "nessuno lo fa così" per l'idea 2: sbagliato, la 22466 era su OpenReview (pubblica dal 3 ottobre). Il documento inviato al professore contiene ancora quella frase.
- Il primo giro del 28 ottobre aveva detto che il controllo online della soglia con audit "non era stato trovato": sbagliato, B-PAC e SSV erano accettati a ICML 2026 e Gibbs et al. è uscito il 22 ottobre.
- Correzioni emerse dalla verifica della bozza del 28 ottobre (`verifiche/audit5/verifica.md`), da applicare a qualunque nuovo testo:
  1. UAT ha anche una garanzia sul rischio (Theorem 4.1), ma solo tramite un proxy allenato con label; CeeBERT ha solo regret.
  2. La 22466 non "si ferma all'allarme": fa escalation degli audit e consegna l'output completo sugli audit; non ricalibra la soglia sotto shift.
  3. La 22466 è pubblica dal 3 ottobre 2026 (odate), creata il 15 settembre.
  4. CATs assume dati scambiabili, gli altri i.i.d.: scrivere "i.i.d. o scambiabili".
  5. La frase sulle due direzioni dello shift va citata (Mehra et al. 2022).
  6. Il monitor sopra il controller va tarato un po' sopra il target.
  7. Nei testi incollati come testo semplice i cancelletti dei titoli Markdown violano la regola sugli hashtag.
- La bozza "Early exit con risk control sotto shift" (Claude Docs, `verifiche/audit5/documento_idea2_rivista.md`) presenta il controller come contributo nuovo: va riscritta secondo la sezione 6.

## 10. Lacune note della ricerca

- NeurIPS 2026 main track non scansionato (lista leggibile solo fino ad "Att..."; un titolo da aprire: "Anytime Valid PAC Bayes Certificates for Adaptive Test Time Scaling").
- Submission ICLR 2027 controllate solo per parole chiave (query troncate a circa 30 risultati). TMLR, AISTATS 2026, UAI 2025 e 2026, workshop OpenReview 2022 2026 solo per parole chiave. COLT 2026 non controllato. Google Scholar e DBLP bloccati.
- arXiv: le nuove uscite sono coperte fino al 28 ottobre 2026.
- Letti solo in parte: Gibbs et al. sezione 4 ed esperimenti; SSV sezione 5 (teorema preso dalla prova in appendice); RACE (a pagamento); diversi lavori solo dall'abstract (CARE, Redline, Compute-Adaptive Risk Control, EoE-Safe e altri sub ICLR 2027).
- Alberi di citazione completi solo su Semantic Scholar (Fast yet Safe 43 citanti, CALM 422, CATs 95, Podkopaev e Ramdas 54, EENN-AVCS 12, anytime 21, UAT 4, Timans 11, Schirmer 12; alberi grandi di CRC 445, LTT 365, ACI 732, RRC 55 filtrati per titolo).
- Non verificato: se l'allocazione degli audit con costi esiste già in active inference o nel survey sampling (è il punto da controllare per primo prima di proporla come contributo).

## 11. Prossimi passi per Claude Code (checklist)

1. Red team indipendente sulla proposta della sezione 6 (novità, correttezza, fattibilità in 10 settimane, interesse per il professore). Chiedere esplicitamente di cercare lavori su "cost aware auditing", "cost sensitive active inference", "importance sampling with costs" e su multi exit con audit.
2. Leggere per intero: Gibbs et al. (2610.31662, sezioni 4, 5 ed esperimenti), SSV (sezione 5), B-PAC (Thm 5.1 e sezione esperimenti), 22466 appendice A (Algoritmo 2, Theorem 3, Proposition 3 e 4), Kubaty et al. ICML 2025, Rethinking Calibration ICML 2026, Pareto Testing, Xu et al. NeurIPS 2024 (politica ottima).
3. Sweep di aggiornamento: arXiv dal 28 ottobre in poi, lista NeurIPS 2026 completa, OpenReview ICLR 2027 con query esaustive (early exit, audit, risk control, multi exit, cascade).
4. Decidere il contributo di metodo (costi degli audit contro Pareto path) e dimostrare i bound (controller adattato, allocazione con costi, cammino).
5. Esperimento pilota: Fast yet Safe su CIFAR100C con teste SDN, consistency risk contro severità per due regimi di training.
6. Scrivere il nuovo documento per il professore nello stile della sezione 1.3: cosa c'è già (citando 22466, B-PAC, SSV, Gibbs et al., HIL-N), cosa resta aperto, l'idea, il piano, il legame con il gruppo e con la finanza, la domanda sulla collaborazione in Spagna e sulla preferenza tra ambientazione standard e device server. Prima di inviarlo, verifica indipendente di ogni citazione e controllo dei caratteri vietati con uno script.

## 12. Aggiornamento del 31 ottobre 2026 (leggere prima della sezione 6)

Report completo: `reports/Ricerca estesa 31 ottobre - idea finale.md`. Script: `scripts/sweep_2026_10_31/`.

- Sweep esaustivi: ICLR 2027 quasi completo con abstract (34.697 record), ICLR 2026, ICML 2026, NeurIPS 2025, TMLR, AISTATS e UAI 2026, workshop NeurIPS e ICML 2026 (116.348 record), titoli NeurIPS 2026 accettati (9.278), arXiv fino al 29 ottobre. L'endpoint `notes/search` di OpenReview via curl non è troncato e accetta `limit=1000` con `offset`.
- C1 (misura di Fast yet Safe sotto shift per training regime) resta nuova. Più vicino: Mitra et al. 2026 (cascate sotto perturbazioni, arXiv 2608.17711).
- La sezione 6.3 (allocazione degli audit con costo, p* proporzionale a radice di q/c) NON è un contributo: è Neyman con costi (Cochran) e c'è già in ML (Angelopoulos et al. 2025, arXiv 2506.07949; AM-PPI; OPAL; soprattutto ReSight, sub ICLR 2027 4ihDGJ5CAY, che completa run fermati presto campionando per predizione e costo residuo con pesi inversi). OMVV (sub ICLR 2027, arXiv 2610.02632) estende SSV a più verificatori con costi.
- Nuova idea raccomandata: misura (C1) più controller (C2, trasferimento) più **adattamento online delle teste di exit con il final exit come target, alimentato da completion e audit random**, backbone e final exit congelati. Argomento chiave: le sole completion (ADEx, IFIP Networking 2026; Online Cascade Learning, ICML 2024) non vedono mai gli input accettati in disaccordo, dove vive il consistency risk; gli audit sì. La garanzia resta valida per aggiornamenti predicibili (adattamento, non teorema nuovo).
- Rischio: Kaplan 2026 (arXiv 2609.14144) trova su LLM che la soglia per deployment vale molto e che specializzare un readout già allenato su tutto il traffico vale circa 2 punti; un readout allenato su un solo tipo è però quasi inutile sugli altri. Evidenza mista. Cancello con pilota offline in settimana 4; piano B nel report (6.7).
- Allineamento con la mail del 28 ottobre (report, sezione 1): il documento del 23 ottobre (`Tesi.pdf`) presentava l'idea 2 come rilevamento (allarme e fallback); il professore indica la linea del risk control. La Discussion di Fast yet Safe v2 elenca come lavori futuri la soglia condivisa, lo shift a test time ([86] Prompt Risk Control) e l'aggiornamento online ([23] Feldman et al., TMLR 2023): la nuova idea risponde a questi punti. La frase "nessuno lo fa così" va corretta: la 22466 era pubblica dal 3 ottobre.
- Spagna: da non considerare per ora (indicazione di Davide, 31 ottobre).
- Correzione: Gibbs et al. non riallenano il modello; il "lavoro futuro con auditor a costi diversi" va ricontrollato sul PDF.

- Verifica del 31 ottobre sul documento per il professore: Mitra et al. 2026 esiste (arXiv 2608.17711, workshop GREEN-AI a ECML PKDD 2026); Nie et al. 2024 raccolgono dimostrazioni anche con salti DAgger a probabilità decrescente; UAT è l'eccezione online al risk control offline; aggiunta la sub ICLR 2027 35881 (Intrinsic Self-Distillation for TTA); Wild Time sostituito da ImageNetV2 e ImageNet R. Documento: `documenti/documento_prof_31ottobre.md` e Claude Docs https://claude.ai/artifact/6p4BCFvoe3pBKzwQRakjPK

- Verifica del 2 novembre (`reports/Verifica stato dell'arte 2 novembre.md`): catalogo completo dei lavori di risk control su early exit (27 righe), punti aperti di Fast yet Safe uno per uno, novità riconfermata (RQ1 nuova, RQ3 nuova come combinazione, RQ2 trasferimento). Legame con il gruppo rafforzato: Pomponi, Scardapane, Uncini (Entropy 2022) allenano un confidence score per exit con BCE contro la label e scrivono che fissare le soglie per exit è difficile; Scardapane et al. ICASSP 2020 (differentiable branching); Lilli, Giarnieri, Scardapane 2021 (multi exit calibrata). Proposta: riadattare online le teste di confidence del gruppo contro l'agreement con il final exit, su ResNet20 multi exit come nei lavori del gruppo. CALM allena già offline un early exit classifier contro l'oracolo di consistenza.

## 13. Risposta del professore e documento di progetto (3 novembre 2026)

- Mail inviata a inizio novembre (versione finale nella conversazione del 2 novembre): idea 2 reimpostata sul risk control, misura sotto shift, controller con audit, adattamento delle teste di confidence (Pomponi et al. 2022) contro l'agreement con il final exit.
- Risposta del professore (verbatim): "Come direzione va benissimo. Possiamo studiare soprattutto come cambiano i savings a livello computazionale sotto shift (considerando che devi anche aggiungere una fase di adattamento)."
- Documento di riferimento del progetto (Claude Docs): "Compute savings degli early exit sotto shift", https://claude.ai/artifact/MNreFiy1XTNWcYZcUC8axi. Contiene domande e ipotesi, notazione (S_P, R_P, scomposizione per exit, Delta S_shift e Delta S_adapt), bilancio del compute (savings netti = lordi meno audit meno adattamento, E[O_audit] = pi S_lordi, payback H_p), costi ResNet20 multi exit (40,6 M MAC, exit a 0,13 0,24 0,36 0,45 0,56 0,68 0,77 0,88), sistema online con fase di adattamento, garanzie con quattro condizioni, esperimenti M0 a M8, piano di dieci settimane con il test della quarta settimana, rischi e decisioni aperte. Il 5 novembre il documento è stato completato con il contesto del 2 novembre (check di letteratura, punti aperti di Fast yet Safe, legame con Pomponi et al. 2022), un approfondimento sui savings (due cause del calo, savings illusori a soglia ferma, scomposizione per exit, transitorio del controller, audit con probabilità per exit alla Neyman con costi, tre versioni della fase di adattamento), la sezione sulla tesi, la sfumatura su Kaplan e le fonti mancanti.

## Appendice A. Documento inviato al professore (versione tenuta da Davide, circa 22 ottobre)

Titolo e paragrafo introduttivo ("Due idee di ricerca. Le due idee sono indipendenti. La prima è più applicata e riguarda la finanza, la seconda è più metodologica. In fondo scrivo anche come si potrebbero unire per la tesi.") più il testo seguente.

Idea 1. Limit order e regime shift

Le neural networks che prevedono se un passive limit order verrà eseguito o cancellato restano affidabili quando il mercato cambia regime all'improvviso? E sapere chi scambia con chi le rende più stabili?

Un passive limit order resta nell'order book finché qualcuno lo esegue o lo cancella. Quando viene eseguito si parla di fill. Il fill tende ad arrivare proprio quando il prezzo sta per muoversi contro l'ordine. Questo si chiama adverse selection, e i trade eseguiti così sono i toxic trades.

Esistono reti che stimano la fill probability, ma o sono allenate su un solo periodo senza un test su un altro, o sono testate sui mesi successivi senza guardare i cambi di regime. C'è anche chi toglie proprio i giorni di volatilità estrema. Da quello che ho trovato nessuno ha studiato se queste previsioni restano affidabili con un regime shift. I lavori più vicini stimano il time to fill con statistiche sui trader che mandano gli ordini, oppure studiano i toxic trades di un broker di foreign exchange, ma senza cambi di regime e senza grafi. Su Hyperliquid, un exchange di perpetual futures, c'è chi usa la wallet history per l'adverse selection, ma non ho trovato neural networks per gli order outcomes.

Userei i dati pubblici di Hyperliquid su Zenodo. Ci sono il lifecycle di ogni ordine (da quando entra nel book a quando viene eseguito o cancellato) su BTC, ETH e SOL per dicembre 2025 e i trade con le controparti di oltre 250 perpetual da ottobre 2025 a gennaio 2026, quindi dovrebbe esserci anche il crash crypto del 10 ottobre 2025 (a ottobre mancano 2 ore di trade, quindi poi in caso devo controllare quali). Ogni ordine è legato a un wallet pseudonimo, perciò si può costruire il grafo di chi scambia con chi.

Vorrei confrontare tre modelli. L'input di base è l'ordine con il market context, l'output è la probabilità che venga eseguito e che il prezzo si muova contro subito dopo. Il primo modello vede solo questo. Il secondo aggiunge la wallet history. Il terzo usa una temporal GNN con i wallet come nodes e i trade tra maker e taker (chi mette l'ordine nel book e chi lo esegue) come edges. Molti edges sono meccanici, perché la controparte la decide la price time priority. Per questo il confronto che conta è tra terzo e secondo modello, perché dice se il grafo aggiunge davvero qualcosa. Allenerei su un periodo e guarderei come cambia la calibration con il regime.

Per il progetto del corso di NN farei solo l'adverse selection su BTC, con i tre modelli, qualche giorno prima e dopo il 10 ottobre. Il primo passo sarebbe costruire il dataset dei fill passivi con i markout, cioè quanto si muove il prezzo dopo il fill. Lascerei alla tesi la parte su fill e cancellazioni di dicembre.

I limiti principali sono tre. Il primo è che il crash è uno solo, quindi non so se i risultati valgono per altri regime shift. Il secondo è che il lifecycle pubblico c'è solo per dicembre 2025 e solo per tre asset, quindi attorno al crash posso studiare solo l'adverse selection e non la fill probability. Il terzo è che le cancellazioni non sono un censoring semplice, perché molte le decide il trader in base al prezzo atteso, quindi tratterei fill e cancellazioni come competing risks.

Idea 2. Early exit che si accorgono da soli degli shift

Una early exit network può accorgersi da sola, senza label, che uno shift ha rotto il suo risk control?

In una early exit network ci sono classifier anche nei layer intermedi. Ogni input si ferma al primo exit abbastanza sicuro, così si risparmia compute. Fast yet Safe, un paper recente, calibra l'exit threshold in modo che la frazione di input in cui l'exit dà una previsione diversa dalla full network (la rete completa fino all'ultimo layer) resti sotto un risk level fissato. Questo è il risk control. Non servono label, perché il riferimento è l'output finale della rete.

Il problema è che il risk control è dimostrato solo se i dati di calibration e di test sono i.i.d., e il paper lascia lo shift come lavoro futuro. Con uno shift il risk control può smettere di valere e non si può verificare, perché quando un input esce prima l'output finale non viene mai calcolato.

L'idea sarebbe fare a meno delle label, perché un riferimento esatto c'è già ed è l'output finale della rete, basta calcolarlo ogni tanto. Su una piccola frazione di input campionati random, che chiamo audit, farei arrivare il forward fino all'ultimo layer, mentre la risposta dell'exit parte comunque subito, e confronterei i due output. La frazione di disagreement è proprio il consistency risk che Fast yet Safe tiene sotto il risk level, quindi gli audit alimentano un sequential test. Il test si aggiorna a ogni audit e tiene bassi i false alarms per tutta la durata del monitoraggio. Appena ci sono abbastanza prove che il consistency risk ha superato il risk level, scatta l'allarme e la rete passa alla full network, dove il disagreement è zero per costruzione ma si perde il risparmio di compute. Gli audit devono essere random con una probabilità nota e mai zero, altrimenti la stima è biased. Gli input incerti si possono sottoporre ad audit più spesso, pesando ogni audit con l'inverso della sua probabilità (inverse probability weighting).

Esiste già un paper che monitora un risk control di questo tipo con un sequential test che tiene bassa la probabilità di false alarm, ma servono le label e non è pensato per le early exit. Un altro lavoro fa a meno delle label usando l'incertezza come proxy dell'errore, ma con un'assunzione che per essere verificata richiede comunque label. Tra i lavori sugli early exit i più vicini sono uno che confronta gli exit con la full network su tutti gli input, quindi senza risparmiare compute, e uno che rileva lo shift con dei cumulative sum charts su exit per flussi di rete, ma senza un false alarm control. Da quello che ho trovato nessuno lo fa così.

Bisogna determinare quanti audit servono per accorgersi in tempo dello shift. Più audit fanno scattare l'allarme prima, ma costano parte del compute risparmiato. Con il test standard il detection delay cresce anche con quanto tardi arriva lo shift, quindi i bet del test, cioè quanto punta a ogni audit, andrebbero stimati su una sliding window degli ultimi audit. Il contributo sarebbe ridurre gli audit, per esempio scegliendo quali input sottoporre ad audit in base alla confidence degli exit, senza perdere il false alarm control. Scegliere in base alla confidence si usa già per ottenere label e non è detto che qui batta gli audit uniformi, quindi lo confronterei con quelli.

Il limite principale è che il risk control riguarda la consistency con la full network e non l'accuracy, quindi il monitor (l'insieme di audit e test) può non far scattare l'allarme mentre l'accuracy cala. Se lo shift peggiora anche la full network, servono alcune label.

Per il progetto servono una early exit network e un data stream che cambia. Allenerei una MSDNet su CIFAR10 e la calibrerei con la versione di Fast yet Safe high probability, così il monitor difficilmente fa scattare l'allarme prima dello shift. Prima misurerei se il risk control si rompe davvero su CIFAR10C, cioè CIFAR10 con corruptions di severity crescente. Non è scontato, perché se gli exit intermedi diventano meno sicuri più input arrivano all'ultimo layer, dove non c'è disagreement. Poi aggiungerei il monitor e passerei a Wild Time, che ha uno shift naturale anno per anno. Misurerei i false alarms, il detection delay e quanto compute resta risparmiato. Il codice di Fast yet Safe è pubblico.

L'idea nasce metodologica, ma si lega a due campi. Il primo è la finanza, come banco di prova con shift reali e improvvisi. Il secondo è l'edge inference, dove la rete gira in parte sul device e in parte su un server. Nel vostro lavoro sulle early exit per le goal oriented communications l'accuracy non si misura a run time e si usa il margin come proxy. Il server può già finire il forward dai risultati intermedi, quindi se il device gliene mandasse una piccola frazione random si stimerebbe, senza label e con risk control, quanto gli exit sul device si discostano dalla full network. Mandare questi risultati costa un po' di banda, ma solo per quella frazione.

Mi farebbe piacere capire di che collaborazione si tratta in Spagna.

Le due idee insieme

Per la tesi vorrei unire le due idee. Userei una piccola early exit network che stima se un fill passivo su Hyperliquid subirà adverse selection. La allenerei e la calibrerei sui fill pubblici, poi la monitorerei durante il crash con gli audit dell'idea 2, così il test gira su uno shift reale. Prima però misurerei quanti false alarms dà il monitor nei periodi senza shift, perché i dati di mercato non sono i.i.d. e il test lo assume. Per sapere se un allarme è giusto basta calcolare la full network offline su tutti i fill. I markout fanno da label e servono solo a capire se peggiora anche la full network. Non ho trovato lavori che lo facciano e mi piacerebbe sapere cosa ne pensa.

## Appendice B. Documenti e artifact prodotti

- Documento lungo iniziale (Claude Docs, non più usato) e versione breve "Due idee di ricerca (versione breve)" (Claude Docs, https://claude.ai/artifact/PdaerMEqgLapCkww3CE86W).
- Bozza superata del 28 ottobre "Early exit con risk control sotto shift" (Claude Docs, https://claude.ai/artifact/Rkhw3dCp5wwT9Kcf3PfkeU; testo in `verifiche/audit5/documento_idea2_rivista.md`).
- Report: `reports/Ordini limite ed early exit.md` (idea 1 e prima analisi dell'idea 2, 19 ottobre), `reports/Risk control per early exit.md` (primo giro sul risk control, 28 ottobre; parzialmente superato dalle scoperte del secondo giro).

## Appendice C. Contenuto del bundle `research_bundle.zip`

- `HANDOFF.md`: questo documento.
- `reports/`: i due report completi.
- `research_notes/Ordini limite ed early exit/`: note del primo lavoro (idea 1, idea 2, fit con il professore).
- `research_notes/Risk control per early exit/`: primo giro sul risk control (ee_risk_control, shift_online_monitoring, ee_under_shift, scardapane_group, benchmarks_protocol, citations_novelty).
- `research_notes/Check completo risk control early exit/`: secondo giro (ml_venues_sweep con tabelle per venue, applied_venues_sweep, citation_trees con conteggi, novelty_final_idea con verdetti per componente e bozze di frasi di novità).
- `verifiche/`: audit dei documenti (audit2: letteratura, dati, metodo e simulazione dell'idea 2; audit3: verifica fatti e chiarezza della versione breve; audit4: test di chiarezza con lettori indipendenti e testo inviato al professore; audit5: verifica della bozza del 28 ottobre; final: versioni del documento lungo).
- `submission_22466/paper.txt`: testo estratto del PDF della submission 22466.
- `scripts/`: script usati per gli sweep (scansione liste di paper, controllo caratteri vietati).
