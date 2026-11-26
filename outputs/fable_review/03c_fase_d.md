# Fase D: requisiti per pubblicare

Stato al 15 novembre 2026, ore 23:05. Piano: `documenti/piano_fable_5_2.md` (v3), sezione 10. Ogni esperimento, con la
previsione scritta prima, è nel quaderno (voci da D.01 a D.04). Lavoro tutto in locale, con 1 worker (dopo i panic
del pomeriggio), cartella `outputs/fable_review/fase_d/`. Nulla committato, nessun file della repository fuori da
`outputs/fable_review/` toccato. Le voci che chiedono Colab sono un piano da approvare, in fondo.

## Cose da decidere

1. **ImageNet: la versione minima è fatta, e non basta.** Con teste lineari allenate su 22.500 immagini gli early
   exit di ResNet50 risparmiano il 3% dei MAC sul pulito e meno del 2% sotto shift (voce D.08). Opzioni: (a) rifare con
   teste allenate su una parte di ImageNet train (un blocco da circa 256.000 immagini, circa 29 GB scaricati dentro
   Colab; per te è lo stesso notebook di prima, 2–3 ore di attesa, qualche CU in più); (b) cercare un modello
   multi-exit già allenato per ImageNet (da verificare in letteratura); (c) chiudere ImageNet come limite dichiarato e
   passare alla Fase E. Proporrei (a), se vuoi la scala nel paper. La dimensione del blocco è da verificare, e teste
   lineari allenate su più immagini restano più deboli di exit allenati insieme alla rete: mi aspetterei savings
   intorno al 10%, non al livello di CIFAR. Training robusto e seconda architettura restano in attesa.
2. **Come presentare l'adattamento.** Su CIFAR-10.1 le mappe sui logit non recuperano nulla. Proporrei di dirlo così:
   la fase di adattamento recupera il 13–17% del calo sotto corruzioni sistematiche (confermato su corruzioni mai
   viste), e niente su uno shift naturale. Un secondo shift naturale più grande (CINIC-10, circa 700 MB da scaricare,
   da verificare) direbbe se dipende dalle poche righe o dal tipo di shift.
3. **La latenza.** Con ResNet20 a 32×32 e batch 1 i savings in MAC diventano tempo solo in parte su CPU e per niente
   su MPS. Proporrei: (a) provare in locale un modello compilato (torch.compile o CoreML), circa un'ora; (b) riportare
   i savings in MAC come metro principale con la latenza accanto, dichiarando il limite; (c) la scala ImageNet, dove il
   calcolo pesa di più.
4. **Fase E** (paper al contrario, roadmap, nota per il professore): non chiede compute. Proporrei di farla mentre
   decidi su Colab.
5. **Tracciamento.** Testi in `03c_tracciamento_proposto.md` e righe in `03c_righe_log_documento.txt` (controllo dei
   caratteri passato).

## In breve

- **I risultati della Fase C si confermano sulle 4 corruzioni mai usate** (24 reti, metodi congelati, previsioni
  scritte prima): il controller vale la soglia oracolo (scarti da −0,006 a +0,005), la fase con fit pesato recupera
  il 13–17% del calo con intervallo sopra zero in tutte le celle e quanto le mappe oracolo, la protezione riporta il
  rischio del ritorno al pulito a quello del controller da solo.
- **Lo shift naturale (CIFAR-10.1) conferma il problema e il controller, non l'adattamento.** La soglia pulita sale a
  rischio 0,066; i savings certificati scendono al 74%; il controller online batte di poco la soglia oracolo; le mappe
  sui logit non recuperano nulla.
- **La latenza reale smonta in parte i savings in MAC.** A batch 1 su CPU resta in tempo il 14–57% dei savings in MAC
  (negativo su CIFAR100 sotto shift); su MPS l'early exit è sempre più lento della rete intera.
- **Previsioni**: 11 su 14 giuste o giuste in media; una giusta a metà (la protezione: giusta sul rischio, ma su
  CIFAR10 il rilevatore scatta nel 79% dei casi invece di almeno il 90%); due sbagliate (scarto da M2 e transitorio su
  CIFAR-10.1, di poco; latenza su CPU, di molto).

## Conferma sulle 4 corruzioni extra

Speckle noise, gaussian blur, spatter e saturate: i logit erano esportati per le 24 reti e mai usati. Metodi congelati
(γ, π, ripartenza, fit pesato, soglie del rilevatore), stessi stream e lettori della Fase C (`fase_d/confirm_jump_read.log`,
`fase_d/confirm_return_read.log`).

| Ultimi 10.000 input | CIFAR10 sev. 3 | CIFAR10 sev. 5 | CIFAR100 sev. 3 | CIFAR100 sev. 5 |
|---|---|---|---|---|
| RQ2: M4 − M2 | −0,006 [−0,016, +0,002] | −0,001 | +0,004 [+0,000, +0,007] | +0,005 |
| M1: rischio | 0,078 | 0,140 | 0,052 | 0,087 |
| RQ3: M7w − M4 | +0,027 [0,011, 0,044] | +0,041 [0,015, 0,068] | +0,014 [0,010, 0,018] | +0,017 [0,012, 0,023] |
| Frazione recuperata | 16,7% | 16,4% | 16,3% | 13,1% |
| M8c − M7w | −0,002 | −0,001 | −0,004 | −0,002 |
| Ripagato entro lo stream | 35% | 54% | 65% | 69% |

- Transitorio di M4 su CIFAR10 a severity 3: 5.900 input (Fase C: 6.450). Garanzia: bound sempre rispettato.
- **Ritorno al pulito con la protezione** (reti del seed 0): rischio sull'intero stream M4 0,050, M7w 0,055, M7wg
  0,049 su CIFAR10; 0,049, 0,059, 0,048 su CIFAR100. Il rilevatore scatta nel 79% delle esecuzioni su CIFAR10 e nel
  100% su CIFAR100, in mediana 226 e 357 input dopo il ritorno, mai prima.

## Shift naturale: CIFAR-10.1

CIFAR-10.1 v6 (Recht et al. 2018): 2.000 immagini nuove raccolte come il test di CIFAR-10, scaricate dal repository
degli autori dopo il tuo ok (`fase_d/data/`). Logit delle 12 reti CIFAR10 calcolati con lo stesso `predict` dell'export
(`fase_d/export_cifar101.py`); la pipeline ridà i logit salvati del test di CIFAR-10 (scarto massimo 1,2e-4, stessa
classe a ogni exit su 2.000 immagini). L'accuracy dell'exit 9 scende in media di 9,5 punti (da 8,4 a 10,9 secondo la
rete).

**Offline** (`fase_d/cifar101_offline.log`; M2 con la UCB su metà delle immagini e valutato sull'altra metà, nei due
versi, contro M2 sul pulito con lo stesso numero di righe):

| Media sulle 12 reti | Pulito | CIFAR-10.1 |
|---|---|---|
| M1, soglia pulita: savings e rischio | 0,535 e 0,045 | 0,440 e **0,066** |
| M2 con 1.000 righe di calibrazione: savings e rischio | 0,506 e 0,036 | 0,376 e 0,037 |
| Soglia oracolo empirica | 0,541 | 0,405 |
| Guadagno delle mappe sui logit alla soglia oracolo | +0,001 | **+0,000** (da −0,010 a +0,016) |

- La soglia pulita su CIFAR-10.1 porta il rischio da 0,045 a 0,066: la garanzia si rompe anche con uno shift naturale.
- I savings certificati restano il 74% di quelli puliti (da 67% a 79% secondo la rete): per perdita di savings
  CIFAR-10.1 sta tra la severity 1 e la 3 di CIFAR-C.
- **Le mappe sui logit non recuperano nulla su CIFAR-10.1**, almeno con 1.000 righe di fit: su uno shift naturale
  non c'è uno spostamento sistematico per classe da correggere. È il risultato negativo più importante della Fase D.

**Online** (stream di 10.000 input puliti e 30.000 di CIFAR-10.1 pescati con ripetizione da 1.000 immagini; le altre
1.000 calibrano M2; `fase_d/cifar101_online_read.log`): M1 resta a rischio 0,063; M4 fa 0,389 negli ultimi 10.000
input contro 0,377 di M2 (+0,011, da −0,014 a +0,039 secondo la rete), con rischio 0,050 e un transitorio di 4.150
input. Il controller vale la soglia oracolo anche sullo shift naturale, e la batte quando M2 ha poche righe.

## Latenza reale

Batch 1, una rete per dataset (joint, seed 0), PyTorch in modalità eager, CPU con un thread e MPS
(`fase_d/latency.log`). Inferenza early exit: blocchi, teste incontrate, max softmax e confronto con la soglia sul
host. M0 è la ResNet20 senza teste.

| | CIFAR10 CPU | CIFAR100 CPU | CIFAR10 MPS | CIFAR100 MPS |
|---|---|---|---|---|
| M0 (ms per input) | 1,20 | 1,05 | 1,53 | 1,56 |
| Uscire all'exit 1: tempo / M0 (MAC / M0 0,13) | 0,28 | 0,19 | 0,30 | 0,29 |
| Uscire all'exit 3: tempo / M0 (MAC / M0 0,36) | 0,75 | 0,72 | 0,82 | 0,79 |
| Arrivare all'exit 9 con tutte le teste: tempo / M0 | 1,13 | 1,17 | 2,31 | 2,28 |
| Soglia pulita sul pulito: savings in MAC e in tempo | 0,575 e 0,326 | 0,245 e 0,034 | 0,575 e 0,001 | 0,245 e −0,715 |
| M2 a severity 3: savings in MAC e in tempo | 0,354 e 0,126 | 0,132 e −0,058 | 0,354 e −0,515 | 0,132 e −0,966 |

- **Il tempo non sta dove stanno i MAC.** I primi tre blocchi lavorano a 32×32 con 16 canali: sono limitati dalla
  memoria e costano tre quarti del tempo con un terzo dei MAC. Uscire presto fa risparmiare meno tempo che MAC.
- **Le teste e i controlli costano**: su CPU il 13–17% di M0 se si arriva in fondo; su MPS ogni exit costa circa
  0,4 ms tra lancio dei kernel e sincronizzazione, e l'early exit è sempre più lento della rete intera.
- Limiti della misura: una rete per dataset, nessuna compilazione, CPU con un thread. Rimedi nella decisione 3.

## ImageNet nella versione minima (scelta da Davide)

- **Disegno** (voce D.05): ResNet50 di torchvision `IMAGENET1K_V1` congelata, 8 teste lineari su global average
  pooling dopo i blocchi 2, 4, 6, 8, 10, 12, 14, 15, più la fc: 9 exit come su CIFAR. Teste allenate su metà di
  ImageNet val; dominio pulito l'altra metà; shift: ImageNet-V2, ImageNet-R e 4 corruzioni di ImageNet-C a severity
  3 (una per famiglia) generate con il codice ufficiale. Previsioni I1–I4 scritte prima del run.
- **Latenza di ResNet50 a batch 1** (pesi casuali, `fase_d/imagenet/latency_resnet50.log`):

  | Exit | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|
  | MAC / M0 | 0,139 | 0,283 | 0,390 | 0,535 | 0,642 | 0,749 | 0,894 | 0,948 | 1,002 |
  | Tempo / M0, CPU (M0 20,0 ms) | 0,309 | 0,454 | 0,521 | 0,639 | 0,725 | 0,810 | 0,929 | 0,993 | 1,046 |
  | Tempo / M0, MPS (M0 5,84 ms) | 0,194 | 0,371 | 0,496 | 0,668 | 0,800 | 0,930 | 1,210 | 1,366 | 1,506 |

  A 224 pixel su CPU il tempo segue i MAC molto meglio che su ResNet20 (scarto da +0,17 sui primi exit a +0,04 sugli
  ultimi, contro fino a +0,39). Su MPS le sincronizzazioni per il confronto sul host si sommano: all'exit 9 con tutte
  le teste 1,5 volte M0. I savings in tempo veri si calcolano quando arrivano i logit (le distribuzioni degli exit).
- **Pacchetto per Colab**: un solo notebook, `imagenet_minimo.ipynb` (dati scaricati su Colab, zip di pochi MB alla
  fine); provato in locale su immagini casuali.
- **Risultato del run di Davide** (voce D.08; 5 domini su 7, fermo a fog per un bug di `imagecorruptions` con NumPy 2,
  ora corretto; `fase_d/imagenet/imagenet_offline.log`):

  | | Accuracy fc | M1: savings e rischio | M2 | Oracolo | M2 in tempo, CPU | Mappe alla soglia oracolo |
  |---|---|---|---|---|---|---|
  | Pulito | 0,759 | 0,032 e 0,043 | 0,032 | 0,034 | −0,011 | |
  | ImageNet-V2 | 0,632 | 0,020 e 0,040 | 0,021 | 0,021 | −0,022 | +0,009 |
  | ImageNet-R | 0,362 | 0,004 e 0,022 | 0,007 | 0,009 | −0,036 | +0,010 |
  | Gaussian noise 3 | 0,313 | 0,001 e 0,008 | 0,006 | 0,006 | −0,038 | +0,011 |
  | Defocus blur 3 | 0,384 | 0,005 e 0,024 | 0,009 | 0,008 | −0,034 | +0,011 |

  **Con le teste della versione minima gli early exit non risparmiano quasi nulla** (3% dei MAC sul pulito, meno del
  2% sotto shift) e in tempo sono più lenti della rete intera. Le prime sei teste (accuracy 2–15%) non escono mai;
  escono solo gli exit dell'ultimo stadio, che pesa il 10% dei MAC. Sotto shift la soglia pulita non rompe la
  garanzia: le teste diventano meno sicure ed escono meno input. Per una scala ImageNet che dica qualcosa servono teste
  più forti (decisione 1).

## Piano per le voci che chiedono Colab (ImageNet minimo approvato, il resto in attesa)

Riferimenti per le stime (README, blocco 2): su un'A100 di Colab, con 3 training insieme, un'epoca costa circa 4 s per
training; le 24 ResNet20 (6.720 epoche) hanno richiesto circa 7,5 ore. Per la tariffa uso circa 12 CU l'ora di A100,
da verificare sul pannello di Colab: con 200 CU sono circa 16 ore. Tutte le cifre sotto sono stime, non misure.

| Voce | Opzione minima | Opzione completa |
|---|---|---|
| Training robusto (AugMix più consistenza tra exit) | 6 reti (joint, due dataset, 3 seed), 3–4 ore, 35–45 CU | 12 reti (joint e joint_sd), 6–8 ore, 65–90 CU |
| Seconda architettura (ResNet56 multi-exit, stesso codice, circa 3 volte i MAC) | 6 reti (joint, due dataset, 3 seed), circa 6 ore, circa 70 CU | 24 reti (i 4 regimi), oltre il budget |
| ImageNet (ResNet50 preallenata, teste disjoint su feature in cache) | teste fittate su metà di ImageNet val, valutate sull'altra metà, su ImageNet-V2, R e 4 corruzioni di ImageNet-C a severity 3; circa un'ora, circa 12 CU; circa 20 GB su Drive | teste fittate su ImageNet train (140 GB) e tutta ImageNet-C (66 GB); 2–4 ore, circa 210 GB |

- **Il criterio di stop del piano** per il training robusto resta: se la quota di savings rimasta a severity 3 migliora
  meno di 5 punti, si chiude.
- **Le tre opzioni minime insieme** costano circa 120 CU e stanno nel budget; le complete no.
- **ImageNet lo devi scaricare tu** (val, V2, R, il sottoinsieme di ImageNet-C). L'opzione minima evita i 140 GB di
  ImageNet train usando metà di val per fittare le teste: è una scelta da dichiarare.
- **Nei prossimi training** (piano, sezione 10): vincolare α delle pooled head a [0, 1] o fissarlo, e usare più seed.
- Ordine che proporrei, per valore: prima ImageNet minimo (la scala è il requisito più chiesto per una venue
  principale ed è il più economico), poi il training robusto minimo, poi la seconda architettura.

## Limiti

- CIFAR-10.1 ha 2.000 immagini: M2 e le mappe offline usano metà per calibrare o fittare e metà per valutare; online
  lo stream ripesca da 1.000 immagini, e per questo niente fase di adattamento online su CIFAR-10.1.
- CIFAR-10.1 esiste solo per CIFAR10.
- Il ritorno con la protezione sulle corruzioni extra è solo sulle 8 reti del seed 0.
- La latenza è misurata su una rete per dataset, in PyTorch eager, a batch 1; con modelli compilati o batch più
  grandi i numeri cambiano.
- Restano i limiti della Fase C (offline su logit in cache, numeri casuali comuni tra corruzioni).

## Comandi e file

Tutto in `outputs/fable_review/fase_d/` (lanci dalla radice della repository, ambiente `eesavings`).
- Dati: `data/` (CIFAR-10.1 v6, sha256 dei dati `2997188e…`, delle label `ae40beda…`); logit: `cifar101/`
  (`export_cifar101.py`, log accanto).
- Job: `run_d01.sh` (conferma sulle corruzioni extra: `confirm_jump/`, `confirm_return/`); stream di CIFAR-10.1:
  comando in `quaderno.md` (D.03), cartella `cifar101_online/`; registro `STATUS.txt`.
- Letture: `confirm_jump_read.log` (`c_read.py` con `ADAPT=M7w`), `confirm_return_read.log`, `cifar101_offline.py`
  e `.log`, `cifar101_online_read.log`.
- Latenza: `latency.py`, `latency.log`.
- Il runner `fase_c/run.py` ora accetta il dominio `cifar101` (metà calibrazione, metà pool dello stream).
