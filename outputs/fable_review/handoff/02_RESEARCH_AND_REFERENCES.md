# 02. Ricerca, risultati e fonti

Stato al 14 novembre 2026. Convenzioni: VERIFICATO, RIPORTATO (da un agente, controllato in parte), DEDOTTO, IPOTESI.

## 1. Il problema di ricerca

Una early exit network risparmia compute facendo uscire gli input facili agli exit intermedi. Fast yet Safe (Jazbec et
al., NeurIPS 2024) sceglie la soglia di uscita λ con una garanzia distribution-free e senza label: il consistency risk
(il disaccordo tra l'exit usato e l'exit finale) resta sotto ε con probabilità 1 − δ, sotto l'ipotesi i.i.d. Sotto
distribution shift:
- con la λ calibrata sul pulito la garanzia si rompe;
- con λ ricalibrata la garanzia regge ma i savings calano;
- una fase di adattamento (teste, score, soglie, BatchNorm) può recuperarne una parte, ma costa compute.

La domanda, nella formulazione del professore: **come cambiano i savings a livello computazionale sotto shift,
considerando una fase di adattamento.**

## 2. Domande e ipotesi di partenza, e il loro stato

Fissate nel documento di progetto prima degli esperimenti.

| | Domanda o ipotesi | Stato al 14 novembre |
|---|---|---|
| RQ1 | Quanto calano i savings sotto shift se il consistency risk deve restare sotto ε? Con soglia fissa e con la migliore sul dominio nuovo | Risposta offline solida (blocco 3 e Parte 1) |
| RQ2 | Senza label e online, quanti savings perde un controller con audit rispetto alla soglia migliore, e quanto ci mette a stabilizzarsi? | Non iniziata |
| RQ3 | Quanto recupera una fase di adattamento delle teste al netto di audit e training, e dopo quanti input si ripaga? | Solo offline: lordo su 3 seed (parziale), payback di M8b misurato |
| H1 | I savings calano con la severity soprattutto perché gli exit superficiali perdono traffico | Confermata su CIFAR10, in parte su CIFAR100 |
| H2 | Con la soglia fissa il consistency risk supera ε alle severity alte | Confermata |
| H3 | Il calo dipende da regime e score; l'agreement regge meglio della max softmax | Il regime conta su CIFAR100; lo score poco |
| H4 | L'adattamento recupera parte dei savings; il costo viene soprattutto dagli audit | Prima metà vera offline; seconda falsa su CIFAR100, dipende dalla contabilità su CIFAR10 |
| H5 | Allenando solo sugli input arrivati in fondo si recupera meno, o il rischio sale | Non testata |

## 3. Risultati sperimentali

### 3.1 Blocco 1: Fast yet Safe sui logit ImageNet (6 novembre)

VERIFICATO (README, `docs/findings.md`).
- Il porting è esatto (`tests/test_upstream_equivalence.py`).
- Con 100 split le garanzie reggono: CRC al massimo ε + 0,002 per il consistency risk; UCB violazioni al massimo 0,11
  per un ε, 0,047 in media. Il prediction-gt-gap con CRC supera ε di 0,011 a ε = 0,01: quella loss può essere negativa e
  non è monotona.
- CRC risparmia da 4 a 15 punti più di UCB.
- Learn then Test nel codice originale è off by one (restituisce la prima soglia non certificata); rimosso.

### 3.2 Blocco 2: le 24 reti (6–7 novembre)

VERIFICATO.
- Backbone (exit 9 di disjoint): 91,3% su CIFAR10, 65,9% su CIFAR100.
- All'exit 1 joint, mixed e joint_sd stanno da 12 a 16 punti sopra disjoint.
- Nei regimi che allenano tutta la rete l'exit 9 è al massimo accurato quanto l'exit 8.
- Su CIFAR-C l'exit 9 perde da 24 a 28 punti; gli exit anticipati perdono meno punti assoluti.
- Le unità di confidence fittate sulle feature hanno AUROC 0,58–0,74 contro 0,74–0,91 della max softmax.
- α < 0 in 46 teste su 192; dove i seed differiscono nel segno, quelle negative sono meno accurate (21 casi su 22,
  2,6 punti in media).

### 3.3 Blocco 3: savings a rischio fissato sotto shift (7–9 novembre)

VERIFICATO; tabella principale nel README. Con la revisione (sezione 3.5) alcune letture sono cambiate.
- H2: M1 sopra ε nel 58–90% dei casi su CIFAR10 e nel 29–55% su CIFAR100.
- M2 regge; a severity 3 resta il 59% dei savings puliti su CIFAR10 e il 53% su CIFAR100.
- Su CIFAR10 quasi tutto il calo è agli exit 1–3.
- Score: differenze al massimo 0,04.
- M8a perde savings (0,255 contro 0,272 su CIFAR10).
- M8b recupera in mediana dall'11 al 25% del calo (seed 0); il criterio della settimana 4 passa in 1 setting su 8.

### 3.4 Controlli preliminari (10 novembre)

Fatti con il seed 0, 2–4 reti, soglia scelta in modo empirico su tutte le righe (senza UCB): **non sono risultati**.
Script e output in `outputs/fable_review/pilots/00_controlli_preliminari/`.
- Alla λ che tiene il consistency risk a 0,05, l'accuracy persa rispetto all'exit 9 è circa zero (+0,012 su CIFAR10
  joint pulito, negativa su CIFAR100). Confermato con il protocollo completo nella Parte 1 (R01).
- Con l'accuracy persa (con label) tenuta a 0,05 i savings non calano sotto shift; con il consistency risk sì.
- L'oracolo per input (primo exit che concorda con l'exit 9) cala poco: da 0,78 a 0,65 su CIFAR10 joint.
- La BN adaptation (statistiche su 2000 input non etichettati) alza molto l'accuracy dell'exit 9 (da 0,47 a 0,64 su
  CIFAR10 joint a severity 5) ma quasi per niente i savings (da 0,260 a 0,292).
- La regola di patience non aiuta (meno di 0,001).
- Soglie per exit proporzionali al costo con la max softmax: da −0,06 a +0,04 rispetto alla λ condivisa, con il segno
  che dipende dal regime.

### 3.5 Parte 1: la revisione critica (11–13 novembre)

Deliverable `outputs/fable_review/01_revisione.md` (24 rilievi con evidenza, correzione ed effetto sui numeri); voci
del quaderno P1.01–P1.18. Tutti i rilievi sono stati approvati da Davide il 13 novembre ("Si applica tutto").

| ID | Area | Severità | Rilievo in una riga |
|---|---|---|---|
| R01 | Riferimento | Bloccante come decisione di progetto | L'exit 9 è un riferimento debole nei regimi congiunti; alla λ certificata su CIFAR100 ha ragione solo nel 28% dei disaccordi; un ensemble 7–9 dà più savings, metà per un effetto meccanico |
| R02 | Blocco 3 | Importante | I "valid savings" di M1 sono fragili; M1 e M2 sono due punti della stessa curva savings contro rischio |
| R03 | Blocco 3 | Importante | Il compute di M8b non è contato e lo domina il training; payback 80.000 e 1,7 milioni di input |
| R04 | Blocco 3 | Importante | joint_sd meglio di joint su CIFAR10 viene da un seed; il ruolo di α è un'ipotesi |
| R05 | Blocco 3 | Importante | M8b solo sul seed 0: va eseguito sui seed 1 e 2 (in corso) |
| R06 | Blocco 3 | Importante | Mancano intervalli e lettura per famiglia; il "passa" del criterio è nel rumore |
| R07 | Blocco 3 | Importante | M2 è un riferimento ottimistico (dominio noto, 3000 forward); dipende da n_cal |
| R08 | Blocco 3 | Importante | La scomposizione 60/40 dipende dall'ordine: 61–71% su CIFAR10 |
| R09 | Bound | Importante | Bound corretto ma largo; test troppo deboli (rafforzati) |
| R10 | Calibrazione | Importante | CRC va riportato accanto a UCB: con poche righe vale 0,13 di savings |
| R11 | Costi | Importante | Nessuna latenza reale misurata |
| R12 | Testi | Importante | Frasi del README e dei findings da riscrivere (elenco nel rilievo) |
| R13 | Dati | Importante | Le scelte del blocco 3 sono state fatte sugli stessi dati dei numeri: serve una conferma a metodo congelato |
| R14 | Calibrazione | Minore | Violazioni di M2 misurate con 100 split: 6,7% e 5,7% fuori dalla calibrazione |
| R15 | Calibrazione | Minore | Test UCB con righe non monotone (aggiunto) |
| R16 | Blocco 3 | Minore | Griglia di λ più fitta: +0,002–0,003, e violazioni fino al 9,8%: tenere la griglia attuale |
| R17 | Blocco 3 | Minore | δ da 0,01 a 0,2: ±0,005–0,01; M2 con 7000 righe: +0,004; del guadagno di M8b resta il 92% |
| R18 | Costi | Minore | Il pooling delle teste non è contato (0,52% invece di 0,12% su CIFAR10) |
| R19 | Training | Minore | Training ed export corretti; l'ultima epoca costa fino a 1 punto |
| R20 | Leakage | Minore | Nessun leakage |
| R21 | Riproducibilità | Minore | Test e rigenerazioni a posto; `reproduce_fys.py` senza `--out-dir` |
| R22 | Codice | Minore | Elenco di punti di qualità, senza riscrivere |
| R23 | Blocco 3 | Minore | La spiegazione di M8a è un'ipotesi sostenuta dalla distribuzione sugli exit |
| R24 | Training | Minore | Regimi e budget di Kubaty da controllare sul loro codice: fatto nella Parte 2 (joint_sd non c'è; loro usano early stopping) |

**Il revisore indipendente** (Opus 5.5, 13 novembre) ha ricalcolato 34 numeri con codice suo: coincidono. Le sue
obiezioni, verificate e integrate:
1. l'ensemble 7–9 come riferimento dà savings in più per metà in modo meccanico;
2. "H4 è falsa" dipende dalla contabilità degli audit;
3. il legame tra α e savings era presentato con troppa forza;
4. M1 non viola sempre: a severity 1 il 42% dei casi su CIFAR10 è sotto ε.

### 3.6 M8b su tre seed (14 novembre, parziale)

VERIFICATO su 19 reti su 24 (`outputs/block3_review/m8b_three_seeds_partial.log`; lo script
`outputs/fable_review/esecuzione/m8b_three_seeds.py` e il ramo a più seed di `savings_figures.py` danno numeri
identici).
- Guadagno di M8b su M2 a severity ≥ 3 (max softmax), sui 5 setting con tre seed: 0,058, 0,053, 0,053. Scarto tra seed
  0,005 (agreement: 0,039, 0,034, 0,033).
- Criterio della settimana 4: solo CIFAR10 joint_sd, al limite su ogni seed (0,533, 0,489, 0,600; in comune 73 casi su
  135, 0,54). Con lo score agreement nessun setting passa.
- Guadagno per severity su CIFAR10: da 0,022–0,024 a severity 1 a 0,073–0,088 a severity 5.
- M8b sopra ε sulle righe di eval: 0,107, 0,110, 0,121 (seed 0, 1, 2). Da ricontrollare fuori dalla calibrazione.
- Mancano CIFAR100 disjoint seed 2, joint_sd seed 1 e 2, mixed seed 1 e 2 (job in corso).

## 4. Letteratura

### 4.1 La mappa precedente

Non rifatta. È in `HANDOFF.md` §4 (alla radice) e nei report di `/Users/davide/Desktop/Tesi/research_bundle/reports/`,
soprattutto "Verifica stato dell'arte 2 novembre.md" (27 lavori di risk control su early exit) e "Ricerca estesa 31
ottobre - idea finale.md". Note per venue, alberi di citazione e verdetti di novità in
`/Users/davide/Desktop/Tesi/research_bundle/research_notes/`. Testo completo della submission 22466 in
`/Users/davide/Desktop/Tesi/research_bundle/submission_22466/paper.txt`.

### 4.2 L'aggiornamento della Parte 2 (13–14 novembre)

Sintesi in `outputs/fable_review/02_letteratura.md`; note complete in `outputs/fable_review/parte2/`:
- `L1_arxiv_recente.md`: 7.254 preprint arXiv dal 29 ottobre al 13 novembre (8 categorie), filtrati per parole chiave e
  autore, più 498 lavori che citano i lavori semi. Nessun lavoro a minaccia 3; nessuno a 2 sul nucleo.
- `L2_openreview_neurips.md`: UAT letto per intero; NeurIPS 2026 (9.276 accettati) filtrato sui titoli; review di ICLR
  2027 non lette (bloccate).
- `L3_direzione_A.md`: la direzione A, con le fonti più vicine lette per intero.
- `L4_fonti_citate.md`: 19 riferimenti del piano verificati; Kiyani et al., Gibbs et al., TinyTTA, Kaplan 2026, DAdEE,
  SAFE-KD letti per intero; codice di Kubaty et al. controllato.

Minaccia: 0 nessuna, 1 da citare, 2 vicino, 3 fa già la stessa cosa.

| Lavoro | Cosa fa | Direzione | Minaccia | Livello di lettura |
|---|---|---|---|---|
| Einbinder, Ringel, Romano 2024 (arXiv 2412.11174) | Risk control semi-supervised con prediction-powered inference; su un early exit di un LLM controlla l'accuracy gap "giusto il modello completo, sbagliato l'exit" con pseudo label e 300–4000 label; i.i.d. | A (poche label) | 3 | Testo completo (RIPORTATO) |
| Submission ICLR 2027 22466 (OpenReview wbXdPogQnU) | Audit del disaccordo dopo il deployment sotto cambi di distribuzione; il disaccordo limita l'errore in eccesso rispetto al riferimento senza assumere che sia corretto; la qualità del riferimento va valutata a parte; con pixelation il modello completo è meno accurato dell'uscita anticipata (errore da 0,477 a 0,496 passando al riferimento) | A, J | 2 | Testo completo in locale, passaggi letti (VERIFICATO) |
| Submission ICLR 2027 22325, "Support Before Sign" (NJsRndgACN) | Nelle cascate di LLM il disaccordo identifica il supporto del guadagno ma non il segno; intervallo stretto; label sui disaccordi | A (lemma) | 2 | Solo abstract |
| Fast yet Safe (Jazbec et al. 2024, arXiv 2405.20915) | Il Brier consistency risk usa una label campionata dal riferimento; nessun esperimento sotto shift (lavoro futuro); nessuna disuguaglianza tra i rischi | baseline | 1 | Testo completo (RIPORTATO) |
| Mehra et al. 2022 (arXiv 2212.01562) | Multi exit su CIFAR-10/100-C: l'oracolo sugli exit guadagna almeno 10 punti sull'ultimo exit, le regole pratiche circa 1; AugMix e BN adaptation riducono gli effetti | B, C, D, G | 1–2 | Testo completo (RIPORTATO) |
| UAT, Bajpai e Hanawal, NeurIPS 2025 (arXiv 2509.23666) | Soglia condivisa scelta online da un bandit; garanzia asintotica su un rischio rispetto alle label; sotto shift riporta qualità e speedup, non il rischio realizzato; solo modelli di linguaggio e BLIP-2 | J | 1–2 | Testo completo (RIPORTATO) |
| TinyTTA (NeurIPS 2024, OpenReview XIcBCBe6C3) | TTA delle sole teste di un early exit su microcontrollori; soglie di entropia fisse; niente risk control né bilancio dei savings | D | 2 | Testo completo (RIPORTATO) |
| Kaplan 2026 (arXiv 2609.14144) | Su un LLM: ricalibrare la soglia per deployment vale fino a 59 punti di exit rate, riadattare la testa circa 2; niente garanzie | D, E | 2 | Testo completo (RIPORTATO) |
| DAdEE (EMNLP 2024 Findings, arXiv 2410.04424) | Domain adaptation offline di un early exit; riusa la soglia della sorgente | D | 1 | Testo completo (RIPORTATO) |
| SAFE-KD (arXiv 2602.03043) | Soglie per exit con CRC, garanzia per exit; shift solo come tabella di accuracy | E | 1 | Testo completo (RIPORTATO) |
| CompCRC (arXiv 2611.11114) | CRC per cascate di LLM a più stadi, scomposizione per stadio, costo; offline, con label | E | 1–2 | Testo quasi completo (RIPORTATO) |
| Kiyani et al. 2026 (arXiv 2602.17633, ICML 2026) | SSV: due soglie aggiornate online solo nei round verificati, pesi inversi; Thm 5.1 ad alta probabilità | J (bound) | 1 | Testo completo (RIPORTATO) |
| Gibbs, Han, Tibshirani, Zrnic 2026 (arXiv 2610.31662) | Audit selettivo online con pesi inversi; Prop. 3 in aspettativa; audit tutti allo stesso costo | J (bound) | 1 | Testo completo (RIPORTATO) |
| Kubaty et al. 2025 (arXiv 2407.14320, ICML 2025) | Regimi joint, disjoint, mixed; AdamW e early stopping (epoche reali 523, 1610, 1166); nessun regime con self distillation | G | 1 | Codice (RIPORTATO) |
| Kaya et al. 2019, SDN (arXiv 1810.07052) | Overthinking distruttivo: metà degli errori anche senza shift | A | 1 | Abstract |
| Mitra 2026 (arXiv 2608.17711) | Cascate sotto corruzione: tre modi di rottura, tra cui "si degradano entrambi i modelli" | A, C | 1 | Abstract |
| Jiang et al. 2022 (arXiv 2106.13799), Kirsch e Gal 2022 (arXiv 2202.01851), Garg et al. 2022 (arXiv 2201.04234) | Disaccordo e accuracy: uguaglianza sotto calibrazione, che si rompe sotto shift; identificare l'accuracy senza label richiede ipotesi | A, F | 1 | Abstract |
| NeurIPS 2026 "Beyond FLOPs" (poster 152802), "Anytime-Valid Conformal Risk Control" (154243), "Auditing Conformal Prediction ... Inference-Time Model Shift" (138875) | Latenza ed energia reali di un multi exit; CRC valido in ogni momento; copertura che si rompe quando cambia il modello | I, B, J | 1 | Abstract |
| ADEx (IFIP Networking 2026) | Adattamento leggero degli early exit a un drift delle classi, con costi di sistema | D | 1 (da leggere) | Solo uno snippet |
| CARED (arXiv 2610.40252), CA-OCRC (arXiv 2611.17059), PROBE (arXiv 2611.11279) | Controllo online con audit o feedback parziale e pesi inversi, fuori dagli early exit | J | 1 | Testo o abstract |
| Bai e Ramdas 2026 (arXiv 2610.35821) | "Recertification" per CRC | B | 1 | Abstract |

Altri lavori verificati in L4 (esistenza e venue): Miller et al. 2021 (accuracy on the line), Baek et al. 2022
(agreement on the line), Schneider et al. 2020, Tent, EATA, RDumb, PABEE, AugMix, Ovadia et al. 2019, Pareto Testing
(Laufer-Goldshtein et al., ICLR 2023, arXiv 2210.07913), EENet (WACV 2024), JEI-DNN (ICLR 2024), T-TAMER (ICLR 2026),
MultiRisk (arXiv 2512.24587, che però non riguarda gli early exit). Dettagli in `parte2/L4_fonti_citate.md`.

### 4.3 Cosa è già fatto e cosa non ho trovato

**Già fatto da altri** (da citare, non rivendicare):
- il disaccordo con il riferimento non è qualità del task, e sotto shift il riferimento può essere peggiore (22466;
  SDN e Mehra et al. sul fenomeno);
- senza label il disaccordo è il massimo identificabile (22325);
- il controllo dell'accuracy gap con poche label e pseudo label in un early exit (Einbinder et al. 2024);
- il controllo online della soglia con audit e pesi inversi (Kiyani et al., Gibbs et al., CARED e altri);
- l'adattamento a test time delle teste di un early exit, senza garanzie (TinyTTA, DAdEE);
- l'allocazione degli audit con costi (Neyman, Angelopoulos et al. 2025, ReSight, OMVV).

**Frasi di novità aggiornate.** Da quello che è stato trovato al 14 novembre 2026, con le ricerche descritte in
`02_letteratura.md`:
- non è stato trovato un lavoro che misuri i compute savings di un early exit con risk control senza label sotto
  distribution shift, a garanzia mantenuta, né per training regime;
- non è stata trovata la scomposizione del consistency risk in "sbaglia l'exit", "sbaglia il riferimento", "sbagliano
  entrambi" misurata alla soglia certificata sotto shift, né il suo effetto sui savings;
- non è stato trovato un bilancio in compute di una fase di adattamento dentro un risk control (costo, payback, netto);
- non è stata trovata, per gli early exit, l'osservazione che distillare le teste verso un riferimento degradato
  recupera savings certificati copiandone gli errori (il fenomeno generale è noto in distillazione e TTA).

Sono risultati negativi di una ricerca, non prove di assenza. Vanno ricontrollati prima di ogni invio.

### 4.4 Cosa cambia nei testi esistenti

- **`docs/controller_bound.md`.** Attribuzione giusta nella sostanza ma incompleta: il termine deterministico coincide
  con la Proposizione 3 di Gibbs et al. (che vale in aspettativa); la versione ad alta probabilità segue la prova del
  Teorema 5.1 di Kiyani et al. (normalizzazione e costanti diverse); mancano i riferimenti completi; nel termine di
  ordine inferiore basterebbe 1/3 al posto di 2/3. RIPORTATO da L4, non riverificato.
- **MultiRisk** va tolto dalle "soglie per exit" nella mappa.
- **Kubaty et al.** La frase del README sul "same total budget" vale solo per i tetti nominali.

### 4.5 Punti ciechi della ricerca

- Review di ICLR 2027 (rilasciate il 5 novembre secondo il calendario della conferenza): non lette, OpenReview risponde
  con una bot challenge, non aggirata.
- Della 22325 solo l'abstract. ADEx solo uno snippet.
- NeurIPS 2026 filtrato sui titoli; ICML 2026 non cercato in modo sistematico.
- Il dump di arXiv copre 8 categorie.
- Semantic Scholar ha risposto spesso con errori di limite.

## 5. Le direzioni

Le direzioni della v2 del piano erano A–J. Dopo la Parte 1, la Parte 2 e la frase del professore:

| Direzione | Stato | Motivo |
|---|---|---|
| B. Bilancio computazionale dei savings sotto shift | **Scelta**, pilota B1 del piano v3 | Risponde direttamente al professore; non trovata in letteratura |
| D. Fase di adattamento che ripaga il suo costo | **Scelta**, pilota B2 | Il professore la chiede; M8b recupera ma non ripaga |
| C. Meccanismo (traffic contro reliability shift) | Scelta, pilota B3 | Spiega perché i savings calano |
| E. Soglie per exit | Scelta come adattamento minimo, pilota B4 | 8 numeri invece di 100.000 pesi; il controllo preliminare 3 dice che la versione semplice non basta |
| J. Parte online (RQ2, RQ3, H4, H5) | Fase C | È il resto del documento di progetto (settimane 5–7) |
| A. Quale rischio certificare | Ridotta a un'analisi dentro B1 e B2 | Le idee teoriche sono già in letteratura (22466, 22325, Einbinder et al.) |
| F. Prevedere i savings senza pagarli | Chiusa, resta una correlazione descrittiva | CRC con 100 righe dà già quasi tutti i savings |
| G. Training robusto (AugMix) | Rimandata, Fase D | Serve Colab; l'effetto è atteso |
| H. Scala e shift naturale | Rimandata, Fase D | Requisito per pubblicare; CIFAR-10.1 è il dato di conferma |
| I. Compute reale e device-server | Latenza in Fase D; device-server per la tesi | Senza latenza un revisore attaccherà i MAC |

**Due idee nuove, da mettere alla prova** (IPOTESI, da `outputs/fable_review/02b_piloti.md`):
1. **La copia degli errori.** Il consistency risk è almeno la massa degli input usciti su cui il riferimento sbaglia e
   l'exit non copia lo stesso errore. Distillare le teste verso il riferimento recupera savings certificati anche
   insegnando a copiare i suoi errori. Test: ipotesi D3 del pilota B2.
2. **Il consistency risk come miglior certificato senza label.** Fissate le predizioni, l'accuracy persa può valere
   qualunque cosa tra −R_c e +R_c al variare delle label. È un caso del risultato della 22325: da scrivere come
   osservazione con la citazione, non come risultato.

**La tesi candidata** (piano v3, IPOTESI): sotto shift i savings certificati senza label si dimezzano circa; il margine
perso non sta nella ricalibrazione ma nello score che non riconosce gli input che possono uscire, e in parte nel prezzo
di certificare l'accordo con un riferimento che si degrada; una fase di adattamento va giudicata per il suo compute
netto, e la distillazione delle teste recupera savings ma non ripaga il suo costo sugli orizzonti reali.

## 6. Risultati teorici

Candidati della v2 (sezione 8.5) e il loro stato:
- **Bound del controller** (`docs/controller_bound.md`): dimostrato, è un adattamento di Kiyani et al. e Gibbs et al.
  Verificato numericamente con avversari adattivi. Vale per aggiornamenti predicibili delle teste; non vale se cambia
  il riferimento (condizione 4), per esempio con una BN adaptation.
- **T1** (identità accuracy persa = P(exit ✗, final ✓) − P(exit ✓, final ✗) ≤ consistency risk): banale; la parte
  "miglior certificato" è nella 22325.
- **T2** (scomposizione esatta di ΔS e ΔR in traffic e reliability shift): pilota B3.
- **T3** (UCB con fixed sequence testing lungo un cammino di soglie fissato prima): standard (Pareto Testing); serve
  per B4.
- **T4** (garanzia sotto adattamento predicibile di teste, score e statistiche BN): le teste e gli score sì; la BN no.
- **T5** (limite sul calo dei savings in funzione dello spostamento degli score): ambizioso, non iniziato.

## 7. Collegamenti tra risultati e letteratura

- Il riferimento che sbaglia (R01) è coerente con l'osservazione della 22466 e con l'overthinking di SDN e Mehra et al.:
  il contributo possibile non è l'osservazione ma la misura in compute su 24 reti e 75 domini.
- L'oracolo per input che cala poco sotto shift (controlli preliminari) è coerente con Mehra et al. (oracolo +10 punti
  sotto corruzione).
- Il prezzo statistico piccolo della ricalibrazione (R07, R14) e il guadagno modesto dell'adattamento delle teste (R03)
  sono coerenti con Kaplan 2026 (la soglia conta molto più del readout).
- La BN adaptation che alza l'accuracy ma non i savings a consistency risk fissato (controlli preliminari) è coerente con
  Schneider et al. 2020 per l'accuracy; per i savings non è stata trovata una fonte.

