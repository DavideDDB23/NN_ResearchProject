# Checkpoint 3b: la parte online (Fase C)

Stato al 15 novembre 2026, ore 21:50 (aggiornato dopo la prova della protezione, voci C.21 e C.22). Piano di
riferimento: `documenti/piano_fable_5_2.md` (v3), sezione 9. Ogni esperimento, con la previsione scritta prima, è nel
quaderno (`outputs/fable_review/quaderno.md`, voci da C.01 a C.22). Ha orchestrato ed eseguito Opus 5.5, senza agenti; i job hanno girato in shell, al massimo 3 processi, e 1 dopo
i due blocchi del Mac del pomeriggio. Nulla è stato committato e nessun file della repository fuori da
`outputs/fable_review/` è stato toccato.

## Cose da decidere

1. **La fase di adattamento da usare.** Proporrei M7w, con la protezione della decisione 2 (insieme: M7wg): fase con
   audit al 25% e mappe fittate con ogni riga pesata 1/q (q la sua probabilità di essere osservata). Su entrambi i dataset recupera più di M7 (118% e 129%), quasi
   quanto il fallback completo, allo stesso costo e con lo stesso rischio, e si ripaga più spesso. L'adattamento
   continuo pesato (M7cw) è il migliore su CIFAR10 ma su CIFAR100 non è affidabile. Dettagli in "Il fit pesato".
2. **Una protezione dopo l'adattamento: provata, funziona** (voci C.21 e C.22). Quando il dominio cambia ancora i
   metodi adattati spendono più rischio del controller da solo, perché mappe e soglia ripartita sono sicure solo sul
   dominio su cui sono state fittate. Un rilevatore senza label sulle probabilità dell'exit 1 (M7wg), con la soglia
   scelta solo sul pulito, toglie le mappe entro poche centinaia di input dal cambio e riporta il rischio a quello del
   controller da solo, senza costi quando il dominio resta. Proporrei M7wg (M7w più il rilevatore) come metodo di
   riferimento della fase di adattamento. Resta aperto se riattivare la fase dopo l'allarme: conviene solo se ogni
   dominio dura più del payback. Lo lascerei alla Fase D o alla tesi.
3. **Il Mac.** Due panic del kernel oggi, sul bus I2C del codec audio e di un chip di autenticazione di Apple, che
   a ogni avvio dà errori di bus (voce C.19). Proporrei Apple Diagnostics e, se si ripete, l'assistenza con il report
   del panic; per i job pesanti un caricatore da almeno 70 W (ora è da 30 W). Fino ad allora i job girano con 1
   worker.
4. **γ.** La regola fissata prima ha scelto 2e-4. Con 5e-4 il transitorio si dimezza e il rischio di lungo periodo
   sotto shift scende (da 0,055 a 0,052 su CIFAR10), al prezzo di più rumore sul pulito. Proporrei di tenere 2e-4 come
   risultato principale e riportare il resto come sensibilità.
5. **Cosa del piano v3 (sezione 9) non ho fatto.** La rampa continua di severity (fatti i gradini), le 50 ripetizioni
   (10 nel job principale, 6 e 4 nelle varianti: gli intervalli sono già stretti), l'orizzonte lungo in stile CCC, lo
   score come control variate (stima AIPW) per stringere il bound, una selezione alla B-PAC, l'allocazione degli audit
   alla Neyman, un trigger che si riarma dopo ogni fase. Proporrei di lasciarli alla Fase D o alla tesi.
6. **Fase D** (serve il tuo ok, alcune voci chiedono Colab): CIFAR-10.1 come shift naturale, conferma sulle 4
   corruzioni extra, latenza reale, ImageNet.
7. **Tracciamento.** Testi per `docs/findings.md` in `03b_tracciamento_proposto.md`, righe per il log del documento di
   progetto in `03b_righe_log_documento.txt` (controllo dei caratteri passato). Non ho scritto nulla nella repository
   né nel documento online.
8. **Aperte da prima.** Se e come committare (nulla è committato); righe per il log e le sei frasi superate del
   documento online; review di ICLR 2027 dal tuo browser; i 12 GB di cache in `pilots/B2_adattamento/cache_bn/`; la
   regola di uscita consapevole del costo (B3); una ricerca in letteratura sulla novità delle mappe sui logit e del fit
   pesato, prima di presentarli come contributo.

## In breve

- **RQ2.** Online e senza label il controller arriva ai savings della soglia oracolo del dominio nuovo a meno di
  0,005, e su CIFAR100 la supera di 0,003–0,004. Il prezzo è il transitorio: su CIFAR10 circa 6.500 input con il
  rischio sopra ε + 0,01 (0,087 nei primi 1.000), su CIFAR100 circa 1.000.
- **RQ3.** La fase di adattamento (scala e bias per classe sui logit delle teste, audit al 25%) recupera a regime il
  12–16% del calo dei savings, quasi quanto le stesse mappe stimate con accesso al dominio nuovo. Su 30.000 input
  dopo lo shift è in pareggio a severity 3 e guadagna 0,010–0,014 a severity 5; si ripaga entro lo stream nel 39–73%
  dei casi.
- **H4 vera**: il costo sta negli audit in più, il training pesa lo 0,1% e il 6–7%. **H5 vera**: con i soli input
  arrivati in fondo si recupera il 45% e il 17% di quello che recupera la fase con gli audit.
- **Le tre versioni** (figura `fase_c/phases_sev3.png`): il fallback completo recupera di più (20% e 21%) ma si
  ripaga meno spesso; l'adattamento continuo non paga una fase ed è il migliore sul netto su CIFAR10, ma su CIFAR100
  è instabile; gli audit rinforzati stanno in mezzo.
- **Il fit pesato** (M7w, ogni riga pesata con l'inverso della probabilità di essere osservata) porta il guadagno
  della fase al 118% e al 129%, quasi quanto il fallback completo, allo stesso costo e rischio.
- **Il risultato che non mi aspettavo: dopo un adattamento il cambio di dominio successivo è pericoloso.** Al ritorno
  al pulito il controller da solo resta sotto ε, i metodi adattati salgono fino a 0,077 su CIFAR10 e 0,144 su
  CIFAR100 nei primi 1.000 input; con corruzioni miste spendono più rischio del controller da solo (su CIFAR100 0,075
  contro 0,053).
- **La protezione** (dopo il checkpoint): un rilevatore senza label che toglie le mappe quando il dominio cambia
  (M7wg) riporta il rischio a quello del controller da solo (0,050 e 0,048 sull'intero stream di ritorno, contro 0,057
  e 0,062 senza), con falsi allarmi nell'1,7% e nello 0% quando il dominio resta. Con shift ripetuti si perde però il
  guadagno della fase singola.
- **La garanzia** di lungo periodo regge sempre (bound 0,22–0,25, largo); il rischio realizzato su 40.000 input sta
  sotto ε + 0,01 nell'80–98% delle ripetizioni.

## Cosa è stato fatto

Il simulatore (`fase_c/sim.py`, `fase_c/run.py`, scritti da me) gioca uno stream di input su logit in cache: per ogni
input si esce al primo exit con max softmax sopra λ_t; se l'input è uscito prima, con probabilità π diventa un audit e
la rete va fino all'exit 9; il controller di `src/eesavings/controller.py` aggiorna λ_t; i savings netti contano
l'exit, gli audit e il training. Per ogni dominio le 10.000 immagini di test sono divise una volta in 5.000 di
calibrazione (solo per i riferimenti offline) e 5.000 da cui lo stream pesca. ε = 0,05, δ = 0,1, π = 0,05, γ = 2e-4
(scelto sugli stream puliti con una regola fissata prima, voce C.01).

La fase di adattamento è quella raccomandata al Checkpoint 3a: scala e bias per classe sui logit di ogni testa, a
backbone ed exit 9 fermi (la garanzia resta), fittati per distillazione verso l'exit 9 su 3.000 righe arrivate in fondo
durante la fase; un quarto delle righe osservate resta fuori dal training e serve a far ripartire la soglia, con un
test a due errori standard prima di ripartire (voce C.04).

| Job | Cosa | Casi | Tempo | Cartella |
|---|---|---|---|---|
| Scelta di γ | Stream solo puliti, 5 valori di γ | 8 reti, 10 ripetizioni | 11 s | `gamma_clean/` |
| Principale | M1, M2, M4, M7, M8, M8c, salto pulito → shiftato | 24 reti, 15 corruzioni a severity 3 e 5, 10 ripetizioni (720 coppie) | 46 min | `main/` |
| Principale corretto | M4 e M7 con la ripartenza corretta | gli stessi 720 | 42 min | `main_v2/` |
| Varianti | M3 (valido), più M5, M6, M7f, M7c, M7t nella versione difettosa | 8 reti del seed 0, 15 corruzioni a severity 3, 6 ripetizioni | 60 min | `variants/` |
| Varianti corrette | M4, M5, M6, M7, M7f, M7c, M7t | come sopra; su CIFAR100 4 ripetizioni e senza M6 | 25 min | `variants_v2/` |
| Fit pesato | M4, M7, M7w, M7cw | come le varianti corrette | 24 min | `variants_w/` |
| Ritorno al pulito | M4, M7, M7c, M8c; M7cw a parte, con M4 | 8 reti del seed 0, 15 corruzioni a severity 3, 6 ripetizioni | 49 min | `return/`, `return_w/` |
| Sensibilità | π = 0,02 e 0,1; γ = 1e-4, 5e-4, 1e-3 (M4 e M7) | 8 reti del seed 0, 15 corruzioni a severity 3, 6 ripetizioni | 35 min | `sens_pi002/`, `sens_pi01/`, `sens_gamma/` |
| Gradini e corruzioni miste | M2, M4, M7, M7t, M7c; M7cw a parte, con M4, solo su CIFAR10 | 8 reti del seed 0, 15 corruzioni; 6 ripetizioni su CIFAR10, 4 su CIFAR100 | circa 1 h 40 min (CIFAR100 con 1 worker, dopo due panic) | `steps/`, `mixed/`, `steps_w/`, `mixed_w/` |
| Protezione (dopo il checkpoint) | M4, M7w, M7wg su salto, ritorno, miste | 8 reti del seed 0, 15 corruzioni a severity 3; ripetizioni come sopra | 52 min, 1 worker | `guard_jump/`, `guard_return/`, `guard_mixed/` |

Verifiche con codice separato (`pilots/verifica_fable/c_check.py`): M1 e M2 simulati contro savings e rischio
calcolati direttamente sui logit (16 confronti, scarto massimo 2,1 errori standard); il percorso della soglia identico
a `controller_step` su 10.000 passi; le mappe identiche a quelle di B2.02; la ripartenza con pesi uniformi uguale
all'oracolo empirico. M4 di `main_v2` è identico a M4 di `main` (57.600 righe, differenza 0). md5 di
`outputs/savings_under_shift/results.csv` invariato.

## Esiti

### RQ2: il controller contro la soglia oracolo

Negli ultimi 10.000 input dello stream (24 reti, 15 corruzioni, 10 ripetizioni; `fase_c/main_v2_read.log`):

| | M4 netti | M2 | M4 − M2 [IC 95%] | M4 − M2 lordi | Rischio M4 / M2 |
|---|---|---|---|---|---|
| CIFAR10 severity 3 | 0,312 | 0,317 | −0,005 [−0,009, −0,001] | +0,012 | 0,050 / 0,044 |
| CIFAR10 severity 5 | 0,222 | 0,223 | −0,001 [−0,004, +0,002] | +0,011 | 0,050 / 0,044 |
| CIFAR100 severity 3 | 0,126 | 0,122 | +0,003 [+0,001, +0,005] | +0,011 | 0,050 / 0,042 |
| CIFAR100 severity 5 | 0,096 | 0,092 | +0,004 [+0,002, +0,006] | +0,010 | 0,050 / 0,042 |

- Online e senza label il controller arriva ai savings della soglia oracolo a meno di 0,005, e su CIFAR100 la supera:
  usa tutto il budget ε (M2, calibrata con la UCB, si ferma a 0,042–0,044) e paga 0,007–0,017 di audit.
- **Il prezzo è il transitorio.** Input dopo lo shift finché la finestra di 1.000 input torna sotto ε + 0,01 (mediana):
  6.450 e 7.150 su CIFAR10, 1.000 su CIFAR100. Nei primi 1.000 input il rischio di M4 è 0,087 su CIFAR10 a severity 3
  (finestra peggiore media 0,103 e 0,134 alle due severity).
- Senza controller (M1, la soglia pulita fissa) il rischio dopo lo shift resta 0,087 e 0,130 su CIFAR10, 0,052 e 0,066
  su CIFAR100: savings più alti di M2, ma con la garanzia rotta.

### RQ3: quanto recupera la fase di adattamento, al netto

S_0 savings netti di M4 negli ultimi 5.000 input puliti, S_1 di M4 e S_2 di M7 negli ultimi 10.000 input shiftati
(`fase_c/main_v2_read.log`; payback: pareggio del compute netto cumulato dallo shift contro M4, mediana sui casi
contando come infiniti quelli non ripagati entro i 30.000 input):

| | S_0 | S_1 | S_2 | S_2 − S_1 [IC 95%] | Frazione recuperata | M8c | Payback (ripagati) | Netto sui 30.000 |
|---|---|---|---|---|---|---|---|---|
| CIFAR10 sev. 3 | 0,515 | 0,312 | 0,343 | +0,032 [0,022, 0,042] | 15,5% [10,7, 21,0] | 0,348 | oltre 30.000 (39%) | −0,003 |
| CIFAR10 sev. 5 | 0,515 | 0,222 | 0,264 | +0,042 [0,024, 0,064] | 14,3% [9,1, 19,8] | 0,268 | 24.700 (58%) | +0,014 |
| CIFAR100 sev. 3 | 0,223 | 0,126 | 0,138 | +0,012 [0,009, 0,016] | 12,5% [8,6, 17,3] | 0,140 | 27.100 (56%) | +0,003 |
| CIFAR100 sev. 5 | 0,223 | 0,096 | 0,116 | +0,020 [0,012, 0,032] | 15,7% [10,0, 22,3] | 0,117 | 19.350 (73%) | +0,010 |

- A regime la fase recupera il 12–16% del calo dei savings, quasi quanto le mappe oracolo (M8c, fittate sulle righe
  di calibrazione del dominio nuovo: +0,002–0,005 sopra M7). Rischio di M7 0,049.
- **Sul conto completo è circa in pareggio.** La fase dura circa 9.000 input su CIFAR10 (audit al 25% finché ci sono
  4.000 righe osservate; reti del seed 0, `variants_v2`) e nei primi 5.000 input M7 risparmia circa 0,08 in meno di
  M4. Su 30.000 input dopo lo shift il netto va da −0,003 a +0,014. Ripaga in un orizzonte di qualche decina di
  migliaia di input, solo se il dominio resta.
- Accuracy negli ultimi 10.000 input: +0,8 e +0,2 punti su CIFAR10, −0,1 e −0,2 su CIFAR100.

### H4: da dove viene il costo

Nei 30.000 input dopo lo shift (`fase_c/h4_v2.log`), in forward complete: gli audit in più di M7 costano 715 e 512 su
CIFAR10, 184 e 158 su CIFAR100; il training delle mappe 0,4 e 11; il guadagno lordo vale 625 e 923, 286 e 484. Il
training pesa lo 0,1% e il 6–7% del costo: **H4 è vera** con le mappe sui logit (con le teste di M8b era falsa, voce
P1.14). Su CIFAR10 a severity 3 gli audit in più costano più di quanto la fase faccia guadagnare entro lo stream.

### La garanzia

Rischio medio sull'intero stream di 40.000 input, per ripetizione (`fase_c/main_v2_read.log`, sezione 5):

| | Media | Massimo | Sotto ε + 0,01 | Bound di Freedman (δ = 0,1) |
|---|---|---|---|---|
| CIFAR10 M4 | 0,0545 | 0,080 | 80% | 0,222, sempre rispettato |
| CIFAR10 M7 | 0,0532 | 0,074 | 88% | 0,241, sempre rispettato |
| CIFAR100 M4 | 0,0495 | 0,078 | 94% | 0,222, sempre rispettato |
| CIFAR100 M7 | 0,0483 | 0,068 | 98% | 0,248, sempre rispettato |

Il bound regge sempre ma è largo: con γ = 2e-4 e 40.000 input il solo termine deterministico vale 0,126, e ogni
ripartenza verso il basso della soglia aggiunge (λ prima − λ dopo)/(γT). Il rischio realizzato è il numero da
riportare; su CIFAR10 sopra ε + 0,01 nel 12–20% delle ripetizioni, per il transitorio dopo lo shift.

### Le tre versioni della fase, e H5

Figura: `fase_c/phases_sev3.png` (sopra i savings netti attorno allo shift con la frazione recuperata, sotto il
compute risparmiato su M4 dallo shift, cumulato: dove passa lo zero la fase si è ripagata). Le 8 reti del seed 0, 15
corruzioni a severity 3; 6 ripetizioni su CIFAR10, 4 su CIFAR100 (`fase_c/variants_v2_read.log`).

| | Guadagno a regime su M4 | Quota di M7 | Durata della fase | Netto sui 30.000 input | Ripagato entro lo stream |
|---|---|---|---|---|---|
| **CIFAR10** | | | | | |
| M7 audit rinforzati (25%) | +0,035 [0,021, 0,051] | 100% | 8.900 input | −0,001 | 40% |
| M7f fallback completo | +0,041 [0,030, 0,055] | 119% | 4.000 | −0,018 | 27% |
| M7c adattamento continuo | +0,021 [0,005, 0,040] | 62% | nessuna | +0,015 | 73% (mediana 11.400) |
| M5 solo input arrivati in fondo | +0,016 [−0,002, +0,038] | 45% | 14.600 | +0,009 | 60% |
| M6 solo audit | +0,003 | 8% | 22.600 | −0,044 | 0% |
| M7t trigger sulla soglia | +0,012 | 34% | scatta nel 63% | 0,000 | 30% |
| **CIFAR100** | | | | | |
| M7 | +0,013 [0,008, 0,018] | 100% | 5.600 | +0,003 | 57% (mediana 27.200) |
| M7f | +0,020 [0,016, 0,024] | 159% | 4.000 | −0,004 | 33% |
| M7c | +0,007 [0,003, 0,012] | 56% | nessuna | +0,001 | 57% |
| M5 | +0,002 [−0,003, +0,009] | 17% | 6.600 | −0,001 | 48% |
| M7t | +0,001 | 6% | scatta nel 20% | 0,000 | 8% |

- **Nessuna versione domina.** Il fallback completo (M7f) recupera di più a regime ma paga una fase carissima
  (savings netti a zero per 4.000 input) e si ripaga meno spesso. L'adattamento continuo (M7c) recupera meno ma non
  paga una fase, ed è il migliore sul netto dello stream su CIFAR10. Gli audit rinforzati (M7) stanno in mezzo.
- **Le mappe sono buone quanto il campione è rappresentativo.** M7f completa tutti gli input, M7 aggiunge audit al 25%
  agli input arrivati in fondo, M5 ha solo questi ultimi (i più difficili): l'ordine a regime è lo stesso.
- **H5 è vera**: con i soli input arrivati in fondo si recupera il 45% e il 17% di quello che recupera M7, senza
  rischio più alto. Solo con gli audit (M6) la fase dura 22.600 input e non si ripaga entro lo stream.
- **Un trigger realistico sulla soglia (M7t) recupera poco**: su CIFAR100 la soglia sotto shift quasi non si muove
  (0,849 prima, 0,851 dopo), quindi non è un buon segnale di shift; nessun falso allarme sul pulito.
- **M7c su CIFAR100 non è affidabile**: sul pulito è instabile (contro M4 −0,035 in un run e +0,002 in un altro, a
  seconda di quante volte la soglia riparte al primo rifit su righe quasi tutte difficili; voce C.17) e dopo lo shift
  ha rischio 0,055.

### M3: monitor con fallback alla rete completa

Dai dati di `variants` (M3 non è toccato dal difetto della ripartenza). Su CIFAR10 il monitor scatta nel 55% delle
ripetizioni, con mediana 4.373 input dopo lo shift; su CIFAR100 nel 18%, mediana 5.632. Dopo, savings a zero: netto
sullo stream −0,097 e −0,019 contro M4. È la scelta prudente della 22466: sicura (rischio medio 0,037 su CIFAR10), ma
butta i savings che il controller tiene a rischio ε.

### Ritorno al pulito: dopo l'adattamento il cambio successivo è pericoloso

Stream di 10.000 input puliti, 15.000 shiftati, 15.000 puliti; 8 reti del seed 0, 15 corruzioni a severity 3, 6
ripetizioni (`fase_c/return_read.log`, `fase_c/return_w_read.log`; M7cw da `return_w`, con lo stesso M4).

| | Rischio, primi 1.000 dopo il ritorno | Primi 5.000 | Tutti i 15.000 | Ultimi 5.000 | Savings ultimi 5.000 − M4 | Rischio su tutto lo stream |
|---|---|---|---|---|---|---|
| CIFAR10 M4 | 0,033 | 0,034 | 0,038 | 0,043 | 0 | 0,050 |
| CIFAR10 M7 | 0,075 | 0,068 | 0,059 | 0,054 | −0,020 | 0,057 |
| CIFAR10 M8c | 0,073 | 0,066 | 0,058 | 0,052 | −0,015 | 0,050 |
| CIFAR10 M7c | 0,069 | 0,064 | 0,055 | 0,050 | −0,014 | 0,057 |
| CIFAR10 M7cw | 0,077 | 0,069 | 0,056 | 0,049 | +0,010 | 0,057 |
| CIFAR100 M4 | 0,042 | 0,045 | 0,046 | 0,046 | 0 | 0,048 |
| CIFAR100 M7 | 0,125 | 0,108 | 0,084 | 0,065 | −0,014 | 0,061 |
| CIFAR100 M8c | 0,115 | 0,100 | 0,080 | 0,063 | −0,010 | 0,057 |
| CIFAR100 M7c | 0,128 | 0,110 | 0,073 | 0,054 | +0,005 | 0,059 |
| CIFAR100 M7cw | 0,144 | 0,122 | 0,080 | 0,057 | +0,010 | 0,060 |

- **M4 attraversa il ritorno senza rischio**: la sua soglia era solo salita. Su CIFAR10 torna ai savings di prima
  dello shift in 4.950 input (mediana), su CIFAR100 subito.
- **I metodi adattati hanno un picco di rischio al ritorno**, molto più forte su CIFAR100: fino a 0,144, quasi tre
  volte ε, e sopra ε + 0,01 per 6.000–10.500 input in mediana (per M7 nel 40% dei casi non rientra entro i 15.000).
  Subito dopo il ritorno hanno più savings di prima dello shift: presi a prestito dal rischio.
- **Il meccanismo** (verificato su curve ed eventi): la ripartenza dopo il fit abbassa la soglia (M7 da 0,92 a 0,79 su
  CIFAR10, da 0,85 a 0,68 su CIFAR100); mappe e soglia insieme sono sicure solo sul dominio su cui sono state stimate;
  il controller risale di γ(rischio − ε) per input. Lo stesso succede con le corruzioni miste (sotto). È la ragione
  della decisione 2.

### Sensibilità a π e γ

Stream con salto, 8 reti del seed 0, 15 corruzioni a severity 3, 6 ripetizioni; riferimento M4 e M7 di `variants_v2`
(`fase_c/sens_read_cifar10.log`; CIFAR100 in `fase_c/sens_read.log`, voce C.20). M4, ultimi 10.000 input:

| CIFAR10 | Netti | Rischio | Finestra peggiore | Rischio su tutto lo stream | Transitorio (mediana) | Guadagno di M7 |
|---|---|---|---|---|---|---|
| Riferimento (π 0,05, γ 2e-4) | 0,302 | 0,050 | 0,110 | 0,055 | 6.250 | +0,035 |
| π 0,02 | 0,313 | 0,051 | 0,110 | 0,054 | 4.950 | +0,035 |
| π 0,1 | 0,286 | 0,050 | 0,105 | 0,054 | 5.750 | +0,033 |
| γ 1e-4 | 0,307 | 0,052 | 0,110 | 0,058 | 9.700 | |
| γ 5e-4 | 0,299 | 0,050 | 0,108 | 0,052 | 2.900 | +0,037 |
| γ 1e-3 | 0,298 | 0,051 | 0,109 | 0,052 | 2.150 | |

- **π** sposta i savings netti di circa quanto costano gli audit (+0,011 con il 2%, −0,016 con il 10%) e non cambia
  il guadagno della fase, i cui audit sono al 25% comunque.
- **γ** decide la durata del transitorio, circa come 1/γ. Con 5e-4 il rischio di lungo periodo sotto shift scende
  da 0,055 a 0,052; la finestra peggiore non cambia, perché nei primi 1.000 input la soglia non fa in tempo a
  muoversi. Sul pulito un γ più grande alza la finestra peggiore (0,082 contro 0,073 con 5e-4, voce C.01): è un
  compromesso, e la regola fissata prima ha scelto 2e-4.
- **CIFAR100** (riferimento con 4 ripetizioni, quindi confronti più rumorosi): π 0,02 dà +0,002 e π 0,1 dà −0,011;
  il guadagno di M7 resta +0,013–0,015 con ogni π e γ. Il transitorio è già al minimo misurabile (1.000 input) per
  ogni γ, e la finestra peggiore sale con γ (da 0,079 a 0,094): senza un transitorio da accorciare resta solo il
  rumore.

### Gradini di severity e corruzioni miste

Figure: `fase_c/timeline_steps.png`, `fase_c/timeline_mixed.png`. Gradini: la stessa corruzione a severity da 1 a 5,
6.000 input ciascuna. Miste: tre corruzioni di famiglie diverse a severity 3, 10.000 input ciascuna. La fase di M7 e
di M7t è una sola per stream (`fase_c/steps_read.log`, `mixed_read.log`, `steps_w_read_cifar10.log`,
`mixed_w_read_cifar10.log`). Savings netti contro M4 sui 30.000 input shiftati, tra parentesi il rischio sugli stessi
30.000:

| | Gradini CIFAR10 | Gradini CIFAR100 | Miste CIFAR10 | Miste CIFAR100 |
|---|---|---|---|---|
| M4: rischio e finestra peggiore | 0,060 e 0,109 | 0,053 e 0,083 | 0,058 e 0,143 | 0,053 e 0,107 |
| M7 | −0,013 (0,059) | −0,005 (0,050) | −0,006 (0,069) | +0,004 (0,075) |
| M7t | −0,002 (0,059) | −0,001 (0,052) | −0,005 (0,066) | +0,001 (0,061) |
| M7c | +0,009 (0,056) | −0,005 (0,047) | +0,019 (0,068) | +0,013 (0,075) |
| M7cw | +0,020 (0,056) | non provato | +0,027 (0,068) | non provato |

- **Gradini.** Il controller resta sempre indietro di un gradino (rischio 0,053–0,060). Su CIFAR10 l'adattamento
  continuo segue la deriva con più savings e meno rischio di M4 (M7cw +0,020); su CIFAR100 nessun metodo adattivo
  batte M4.
- **Corruzioni miste.** Tutti i metodi adattati spendono più rischio del controller da solo, molto di più su
  CIFAR100 (0,075 contro 0,053; finestre peggiori 0,21–0,25 contro 0,11). Le mappe della prima corruzione e la soglia
  ripartita valgono solo per quella. I savings in più (M7c +0,013–0,019) sono in parte presi a prestito dal rischio.
- M7cw non l'ho provato su CIFAR100 negli stream a segmenti: la previsione C4 era condizionata al fatto che reggesse
  su CIFAR100, e non regge.

### Il fit pesato delle mappe

Nel fit di una fase (M7w) o dell'adattamento continuo (M7cw) ogni riga osservata pesa 1/q, con q la probabilità che
aveva di essere osservata: 1 per gli input arrivati in fondo da soli, 0,25 per gli audit della fase, 0,05 per quelli
del continuo. La loss di distillazione diventa una stima non distorta di quella sul dominio intero, il bersaglio che
M7f raggiunge pagando la rete completa per ogni input. Stesse reti e ripetizioni delle varianti
(`fase_c/variants_w_read.log`); M4 e M7 rifatti dopo la modifica del simulatore sono identici a quelli di
`variants_v2` su tutte le 6.000 righe.

| | Guadagno a regime su M4 | Quota di M7 | Rischio, ultimi 10.000 | Netto sui 30.000 | Ripagato entro lo stream |
|---|---|---|---|---|---|
| CIFAR10 M7 | +0,035 [0,021, 0,051] | 100% | 0,049 | −0,001 | 40% |
| CIFAR10 **M7w** | +0,041 [0,030, 0,054] | 118% | 0,049 | +0,004 | 52% (mediana 29.800) |
| CIFAR10 M7cw | +0,042 [0,030, 0,056] | 122% | 0,051 | +0,028 | 98% (mediana 1.050) |
| CIFAR100 M7 | +0,013 [0,008, 0,018] | 100% | 0,051 | +0,003 | 57% (mediana 27.200) |
| CIFAR100 **M7w** | +0,016 [0,013, 0,020] | 129% | 0,051 | +0,005 | 70% (mediana 21.300) |
| CIFAR100 M7cw | +0,013 [0,008, 0,017] | 100% | 0,058 | +0,006 | 70% (mediana 9.900) |

- **M7w**: stessa fase, stessi audit e stesso rischio di M7; recupera il 118% e il 129% di quello che recupera M7
  (il 99% e l'81% di M7f) e si ripaga più spesso. È la versione che proporrei (decisione 1).
- **M7cw** su CIFAR10 è il migliore di tutti sul netto dello stream e sui gradini di severity, ma con rischio un po'
  sopra ε (0,054 sul pulito). Su CIFAR100 non regge: perde 0,020–0,023 di savings sul pulito e ha rischio 0,058 dopo
  lo shift. Con 100 classi le mappe stimate da circa 400 righe effettive (audit pesati 20) sono troppo rumorose.
- Pesare con l'inverso della probabilità di osservazione è la correzione standard (inverse propensity weighting):
  non dichiaro novità. Se c'è qualcosa di nuovo è solo l'uso dentro la fase di adattamento di un early exit
  certificato, da verificare in letteratura (decisione 7).

### La protezione dopo l'adattamento (dopo il checkpoint, voci C.21 e C.22)

Davide ha detto di continuare: ho provato la decisione 2. M7wg è M7w più un rilevatore senza label: dopo il fit
confronta la media mobile delle probabilità dell'exit 1 (teste originali, calcolate per ogni input) con il suo valore
al momento del fit, e quando la distanza di variazione totale supera κ_d toglie le mappe e rimette la soglia di prima
della ripartenza (da lì il sistema è M4). Il segnale l'ho scelto prima delle simulazioni: riconosce l'83–88% dei ritorni
al pulito e il 100% dei cambi tra corruzioni, contro il 40–62% della confidenza media. κ_d l'ha scelto una regola
fissata prima, solo su stream puliti: 0,06 su CIFAR10, 0,10 su CIFAR100, nessun falso allarme su 40 esecuzioni.
Figure: `fase_c/timeline_guard_return.png`, `fase_c/timeline_guard_mixed.png`.

| | Rischio, primi 5.000 dopo il ritorno | Rischio su tutto lo stream di ritorno | Miste: rischio sui 30.000 | Miste: savings − M4 | Salto: guadagno a regime |
|---|---|---|---|---|---|
| CIFAR10 M4 | 0,034 | 0,050 | 0,058 | 0 | 0 |
| CIFAR10 M7w | 0,068 | 0,057 | 0,067 | −0,009 | +0,041 |
| CIFAR10 **M7wg** | 0,037 | 0,050 | 0,058 | −0,019 | +0,040 |
| CIFAR100 M4 | 0,045 | 0,048 | 0,053 | 0 | 0 |
| CIFAR100 M7w | 0,111 | 0,062 | 0,075 | +0,004 | +0,016 |
| CIFAR100 **M7wg** | 0,050 | 0,048 | 0,053 | −0,005 | +0,016 |

- **Funziona.** Al ritorno scatta nel 99–100% delle esecuzioni, in mediana 235 e 382 input dopo il cambio; con le
  corruzioni miste nel 100%, in mediana circa 200 input dopo. Il sistema adattato torna sicuro come il controller da
  solo; resta un picco breve prima dell'allarme (0,044 e 0,068 di rischio nei primi 1.000 input).
- **Quando il dominio resta non costa quasi nulla**: falsi allarmi dopo la fase nell'1,7% e nello 0% delle
  esecuzioni, guadagno a regime uguale a M7w entro 0,001.
- **Il limite che resta è la fase singola.** Con shift ripetuti la fase si paga nel primo segmento e il guadagno si
  perde al cambio successivo: sulle corruzioni miste M7wg sta sotto M4 di 0,019 su CIFAR10 e 0,005 su CIFAR100, con lo
  stesso rischio. Una fase che si riattiva dopo l'allarme si ripagherebbe solo se ogni dominio dura più del payback.

## Previsioni contro risultati

Tutte scritte nel quaderno prima dei job; il dettaglio, con il numero vero di ogni previsione, è nelle voci indicate.

| Voce | Previsione | Esito |
|---|---|---|
| C.01, 1 | M4 entro ±0,015 da M2 | giusta (da −0,005 a +0,004) |
| C.01, 2 | Rischio di M4 dopo lo shift | giusta sulle medie, sbagliata sulla finestra peggiore (0,103 contro 0,12–0,20) |
| C.01, 3 | Transitorio 2.000–6.000 input su CIFAR10 | di poco sbagliata (6.450) |
| C.01, 4 | M1 con rischio 0,08–0,15 e savings illusori | giusta |
| C.01, 5 | M3 scatta entro 500–3.000 input | sbagliata (55% delle ripetizioni, mediana 4.373) |
| C.01, 6 | Guadagno di M7 su M4 | giusta |
| C.01, 7 | Frazione recuperata 10–15% e 7–13% | con la versione corretta di poco sbagliata su CIFAR10 (15,5%), giusta su CIFAR100 (12,5%) |
| C.01, 8 | Payback 15.000–35.000 e 25.000–60.000 | giusta su CIFAR100 (27.100); su CIFAR10 a severity 3 la mediana va oltre lo stream |
| C.01, 9 | M8c − M7 tra 0 e 0,005 | giusta |
| C.01, 10 | H5: M5 recupera il 50–90% di M7 | sbagliata nei numeri (45% e 17%), H5 vera |
| C.01, 11 | M6 come M7 ma con una fase di 20.000 input | giusta sulla durata, sbagliata sul resto |
| C.01, 12 | M7f fase più breve e cara, payback simile | sbagliata sul payback (più lungo) |
| C.01, 13 | M7c recupera quanto M7, in ritardo | sbagliata (62% e 56%) |
| C.01, 14 | Garanzia sotto ε + 0,01 e sotto il bound | giusta sul bound, in parte sbagliata su ε + 0,01 |
| C.01, 15 | Accuracy entro un punto | giusta |
| C.04 | Con la ripartenza corretta il payback su CIFAR100 si allunga di 1.000–4.000 | sbagliata (+14.550) |
| C.06, R1–R2 | M4 al ritorno: rischio basso, recupero in 4.000–12.000 input | giuste |
| C.06, R3–R4 | Savings di M7, M8c, M7c negli ultimi 5.000 | giuste sui savings, sbagliate sul rischio; R4 sbagliata su CIFAR10 |
| C.06, R5 | Nessun picco di rischio al ritorno | **sbagliata** (fino a 0,077 e 0,144) |
| C.06, S1–S5 | Sensibilità a π e γ (CIFAR10) | giuste tranne la finestra peggiore con γ (non cambia) e due scarti di 0,001 |
| C.07, G1–G7 | Gradini e corruzioni miste (CIFAR10) | giusti gli ordinamenti; sbagliate le finestre peggiori (più alte) e il rischio dei metodi adattati con corruzioni miste |
| C.09, W1–W3 | Fit pesato: M4 e M7 identici; M7w vicino a M7f, payback più breve | giuste (salvo +0,002 invece di +0,003 sul netto su CIFAR100) |
| C.09, W4 | M7cw sul pulito di CIFAR100 entro 0,015 da M4 | sbagliata (−0,023) |
| C.13, C1–C3 | M7cw sugli stream con più cambi | giusta sui gradini; sbagliate sul picco al ritorno e sul rischio con corruzioni miste |

| C.07, G8 | Su CIFAR100 gli stessi ordinamenti, con differenze dimezzate | sbagliata sui gradini (nessun metodo adattivo batte M4), a metà sulle miste (M7c primo, ma rischio in eccesso doppio) |
| C.06, S1–S5 su CIFAR100 | Come su CIFAR10 | S3 e S5 giuste, S1 di poco e S2 sbagliate; il transitorio non si confronta (già al minimo); la finestra peggiore sale con γ |
| C.13, C4 | M7cw su CIFAR100, se regge | non provata: la condizione non vale |
| C.21, K1 | Soglia del rilevatore tra 0,04 e 0,08 (CIFAR10) e tra 0,08 e 0,15 (CIFAR100) | giusta (0,06 e 0,10) |
| C.21, K2 | Salto: falsi allarmi al più 5%, guadagno entro 0,003 da M7w | giusta (1,7% e 0%; −0,0006) |
| C.21, K3 | Ritorno: allarme entro 2.000 input nell'80%, rischio nei primi 5.000 al più 0,055 e 0,060 | giusta (99–100%; 0,037 e 0,050) |
| C.21, K4 | Miste: rischio al più M4 + 0,005, savings sotto M4 di 0,005–0,025 | giusta, salvo −0,0047 su CIFAR100 (appena fuori) |

Su circa 40 previsioni: una quindicina giuste, altrettante giuste solo in parte, una decina sbagliate. Gli errori
vanno soprattutto in una direzione: ho sottovalutato quanto rischio spende un sistema adattato quando il dominio
cambia ancora, e le finestre peggiori in generale.

## Limiti

- Tutto offline, su logit in cache: lo stream pesca con ripetizione dalle 5.000 immagini di test di ogni dominio
  tenute fuori dalla calibrazione. Il compute è contato in MACs, senza latenza reale.
- Il job principale è sulle 24 reti; varianti, ritorno al pulito, sensibilità, gradini e corruzioni miste solo sulle
  8 reti del seed 0, con 6 ripetizioni su CIFAR10 e 4 o 6 su CIFAR100.
- Solo corruzioni sintetiche (CIFAR-C). Uno shift naturale (CIFAR-10.1) è nella Fase D.
- M7 usa un trigger oracolo (sa quando arriva lo shift). Il trigger realistico sulla soglia (M7t) recupera poco, e
  quasi niente su CIFAR100, dove sotto shift la soglia quasi non si muove.
- La fase di adattamento è una sola per stream: dopo, il metodo non si riadatta (tranne M7c e M7cw).
- M7c e M7cw su CIFAR100 hanno rischio sopra ε negli ultimi 10.000 input (0,055 e 0,058); M7cw anche sul pulito di
  CIFAR10 (0,054).
- Il bound di Freedman regge sempre ma è largo (0,22–0,25): il numero da guardare è il rischio realizzato.
- La soglia di due errori standard del test prima della ripartenza e il quarto di righe tenuto da parte non sono stati
  variati; γ e π solo nella sensibilità.
- M7w e M7wg li ho provati sul ritorno al pulito e sulle corruzioni miste (voce C.22), non sui gradini di severity:
  lì il rilevatore scatterebbe a ogni gradino e M7wg tornerebbe M4.
- Gli intervalli sono bootstrap a due vie su reti e corruzioni, con le ripetizioni mediate dentro ogni cella. Ogni
  caso usa lo stesso seed per lo stream e per gli audit, quindi per una rete e una ripetizione il segmento pulito è
  lo stesso run per tutte le corruzioni (voce C.17). È un vantaggio per i confronti appaiati tra metodi, ma la parte
  casuale che dipende dal pulito è condivisa tra le corruzioni: gli intervalli di M7c e M7cw, il cui comportamento sul
  pulito dipende dal caso, sono troppo stretti, e sul pulito i run indipendenti sono solo 4 reti per le ripetizioni.
  Nei prossimi job conviene un seed per corruzione.

## Errori in questo giro

1. **La ripartenza della soglia era ottimista** (voce C.04). La soglia ripartiva da una stima del rischio calcolata
   sulle stesse righe su cui erano fittate le mappe. Trovato da me rileggendo il rischio delle varianti (M7 a 0,054 su
   CIFAR100 per 15.000 input, M7c sopra ε anche sul pulito). Corretto con un quarto delle righe tenuto da parte e un
   test prima di ripartire; rifatti M7 sulle 24 reti e tutte le varianti con adattamento. Il payback su CIFAR100 è
   passato da 12.550 a 27.100 input: avevo previsto un allungamento di 1.000–4.000.
2. **Velocità.** Il primo rilancio ricostruiva a ogni passo la lista delle righe di training: fermato, sostituito con
   un contatore, risultati identici su una prova.
3. **Un worker rimasto vivo** dopo lo stop ha scritto 3 risultati nella cartella nuova: rifatti da capo, identici.
4. **Il lettore delle varianti** avrebbe preso per M6 su CIFAR100 le curve difettose della cartella vecchia: corretto
   prima della lettura (ogni metodo, per dataset, da una sola cartella).
5. **`args.json`** viene riscritto a ogni lancio nella stessa cartella: quello di `variants_v2` è del lancio su
   CIFAR100. Da lì tengo una copia per dataset (`args_cifar10.json`, `args_cifar100.json`).
6. **Una stima di M7c presa per buona** (voce C.17). In C.08 avevo scritto che M7c perde 0,035 di savings sul pulito
   di CIFAR100; un secondo run, uguale salvo le estrazioni degli audit, dà +0,002. Il segmento pulito è lo stesso run
   per tutte le corruzioni, quindi quella stima veniva da 16 run indipendenti, non da 240. Corretta nel quaderno.
7. **Le previsioni sbagliate** sono nella sezione sopra, ciascuna con il numero vero.
8. **Un incidente, non un errore** (voci C.18 e C.19). Il Mac si è bloccato due volte durante i gradini su CIFAR100,
   alle 17:33 e alle 18:03, con un panic del kernel sul bus I2C condiviso dal codec audio del jack e da un chip di
   autenticazione di Apple, che a ogni avvio dopo il primo blocco dà errori di bus. È un guasto hardware o di driver;
   il carico dei job può aver fatto da innesco. Nessun dato perso: le sintesi si scrivono in modo atomico e il runner
   salta i casi fatti. Ripreso con 1 worker su decisione di Davide, finito senza altri errori.

## Comandi e file

Tutto in `outputs/fable_review/` (lanci dalla radice della repository, ambiente `eesavings`).

- Simulatore e runner: `fase_c/sim.py`, `fase_c/run.py` (stream `clean`, `jump`, `return`, `steps`, `mixed`; metodi
  M1–M8c, M7w, M7cw). Job: i comandi del job principale e delle varianti sono in `quaderno.md` (C.02–C.05);
  ritorno, sensibilità, segmenti e fit pesato in `fase_c/run_c06.sh`. Registro dei job con orari e batteria:
  `fase_c/STATUS.txt`.
- Letture: `pilots/verifica_fable/c_read.py` (job principale), `c_variants_read.py` (varianti e fit pesato),
  `c_return_sens_read.py` (ritorno, sensibilità, segmenti), `c_h4.py` (H4), `c_check.py` (verifiche del simulatore),
  `c_same_methods.py` (riproducibilità tra cartelle). I log sono accanto alle cartelle in `fase_c/`.
- Figure: `fase_c/figure_timeline.py` (timeline: del job principale `timeline_main_sev3.png` e `sev5`, del ritorno
  al pulito `timeline_return_sev3.png`, dei gradini `timeline_steps.png`, delle corruzioni miste `timeline_mixed.png`)
  e `fase_c/figure_phases.py` (le tre versioni della fase, `phases_sev3.png`, la figura chiesta dal piano).
- I job con `run_c06.sh` accettano `WORKERS=1` o `2` nell'ambiente (dopo i panic, 1 worker).
- Protezione: `fase_c/guard_calibrate.py` (soglia sul pulito, `guard_calibrate.log`), modo `guard` di `run_c06.sh`
  (con `GUARD_KAPPA`), lettura `pilots/verifica_fable/c_guard_read.py` (`fase_c/guard_read.log`), figure
  `timeline_guard_return.png` e `timeline_guard_mixed.png`.
- Cartelle dei risultati: `gamma_clean/`, `main/`, `main_v2/`, `variants/` (solo M3 è valido), `variants_v2/`,
  `variants_w/`, `return/`, `return_w/`, `sens_pi002/`, `sens_pi01/`, `sens_gamma/`, `steps/`, `steps_w/`, `mixed/`,
  `mixed_w/`. Non usate nelle letture: `return_restart_in_sample_STOPPED/` (ritorno con la ripartenza difettosa,
  fermato), `main_v2_lingering/` (i 3 casi scritti dal worker rimasto vivo), `main_v2_redo/` (gli stessi 3 casi
  rifatti, identici, già copiati in `main_v2/`), `main_cifar10_first_run/` (sintesi CIFAR10 di un lancio precedente
  del job principale).
