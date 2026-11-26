# Quaderno di laboratorio

> **Nota della ricostruzione (26 novembre 2026).** Il quaderno originale è andato perso con il Mac. Questa
> versione è ricostruita dalle chiamate della sessione Claude del 14–16 novembre. Le righe 1–195 (intestazione
> e voci da P1.01 a P1.13 della revisione dell'11–13 novembre) non sono mai state stampate in quella sessione e
> non si possono recuperare: i loro esiti sono in `docs/findings.md` (sezione "Review of blocks 1 to 3").
> I titoli noti di quelle voci, con il numero di riga originale:
>
>     11:## Parte 1: revisione critica
>     13:### P1.01 (11 nov) Riproducibilità: i test passano tutti?
>     20:### P1.02 (11 nov) Riproducibilità: la riproduzione di Fast yet Safe con 5 split è identica?
>     33:### P1.03 (11 nov) Calibrazione UCB: equivalenze e copertura
>     47:### P1.04 (11 nov) Riproducibilità: rigenerare righe di results.csv
>     60:### P1.05 (11 nov) Cosa dice results.csv rispettando la dipendenza tra i casi
>     74:### P1.06 (11 nov) Il tasso di violazione di M2 misurato senza il rumore delle 3000 righe di eval
>     84:### P1.07 (11 nov) La scomposizione 60/40 nei due ordini
>     99:### P1.08 (11 nov) Il riferimento: exit 9 contro exit 8 e contro un ensemble
>     131:### P1.09 (11 nov) Sensibilità a n_cal, δ, ε e CRC al posto di UCB
>     141:### P1.10 (11 nov) Costi, training ed export
>     159:### P1.11 (11 nov) Il bound del controller contro avversari adattivi
>     175:### P1.12 (11 nov) I numeri di README e findings
>     185:### P1.13 (11 nov) La conclusione sugli score dipende dalla griglia di λ?

  dà +0,015 su UCB con 3000 righe e +0,127 con 100, con rischio medio 0,049 e il 39–47% delle calibrazioni sopra ε.
- **Convinzioni.** Giuste le previsioni su n_cal = 250, su δ e su CRC. Sbagliata quella su n_cal = 100: prevedevo meno di
  0,25 su CIFAR10, è 0,40. Il bound WSR è più stretto di quanto pensassi quando le loss sono rare. La quota "53%" di
  CIFAR100 dipende da n_cal; CRC va riportato accanto a UCB, perché è nel regime di poche righe (quello realistico sotto
  shift) che la scelta della garanzia pesa.

### P1.14 (11 nov) Quanto costa l'adattamento delle teste (M8b) e quando si ripaga

- **Domanda.** Quanti MAC costa il training di M8b (L-BFGS, 5 valori di L2, 8 teste) rispetto a una forward completa, e
  dopo quanti input dello stesso dominio il guadagno su M2 ripaga il costo?
- **Previsione.** Limite superiore a mano: 125 valutazioni per 2·N·D·C MAC, cioè circa 4000 forward complete su
  CIFAR10 e 40000 su CIFAR100. Payback di decine di migliaia di input su CIFAR10 e di milioni su CIFAR100.
- **Comando.** `$PY m8b_cost.py` da `outputs/fable_review/parte1/` (log in `m8b_cost.log`).
- **Risultato.** In media 84 (CIFAR10) e 97 (CIFAR100) valutazioni per testa e per L2: il training vale 2682 e 31215
  forward complete. Con le 4000 forward di adattamento, il payback a severity 3 è 83.000 input su CIFAR10 e 1,7 milioni
  su CIFAR100 (40.000 e 766.000 a severity 5); senza il costo del training sarebbe 26.000–29.000. La sola ricalibrazione
  di M2 su uno stream di 10.000 input porta i savings netti da 0,301 a 0,210 (CIFAR10) e da 0,127 a 0,086 (CIFAR100).
- **Convinzioni.** Previsione giusta nell'ordine di grandezza. La seconda metà di H4 ("il costo viene soprattutto dagli
  audit") è falsa per M8b così com'è: domina il training delle teste, soprattutto con 100 classi. Un adattamento che
  ripaga deve costare 10–100 volte meno (un solo L2, poche iterazioni, o solo le teste iniziali).

### P1.15 (11 nov) M8b contro M2 a pari dati

- **Domanda.** M8b usa 7000 righe del dominio, M2 solo 3000. Contro M2 calibrato su 7000 righe quanto guadagno resta?
- **Previsione.** Quasi tutto: le 4000 righe in più danno a M2 meno di 0,005.
- **Comando.** `$PY m2_equal_data.py` (log in `m2_equal_data.log`).
- **Risultato.** M2 con 7000 righe guadagna 0,004. Il guadagno di M8b a severity ≥ 3 passa da 0,047 a 0,043 (resta il
  92%).
- **Convinzioni.** Previsione giusta. Il guadagno di M8b non è un artefatto del budget di dati.
- **Risultato P1.13.** La griglia 5 volte più fitta e quella ai quantili danno +0,002–0,003 di savings (massimo 0,006).
  A severity 3 lo score migliore è lo stesso in 8 setting su 8; sul pulito cambia in 1 su 8, per meno di 0,003. La
  differenza agreement − max softmax si sposta di al massimo 0,002.
- **Convinzioni.** Previsione giusta: la conclusione su H3 non dipende dalla griglia.

### P1.16 (12 nov) La griglia fine resta valida?

- **Domanda.** Con la griglia fine i casi sopra ε sulle righe di eval salgono dal 10% al 14%. È ancora rumore?
- **Previsione.** Sì: il fixed sequence testing vale per ogni griglia fissata prima. Mi aspetto che sulle 7000 righe
  fuori dalla calibrazione il tasso resti sotto δ = 0,1, ma più vicino di prima.
- **Comando.** `$PY grid_fine_validity.py` (seed 0, severity 3, 40 split; log in `grid_fine_validity.log`).
- **Risultato.** Griglia originale: 9,0% sulle righe di eval, 6,9% sulle 7000 righe, 3,2% sulla popolazione. Griglia
  fine: 12,5%, 9,8% e 5,2%.
- **Convinzioni.** Previsione giusta. La griglia fine usa quasi tutto il budget δ per guadagnare 0,002: non conviene
  cambiarla.

### P1.17 (13 nov) Controllo incrociato dello script sul riferimento e revisore indipendente

- **Domanda.** `reference_exit9.py`, con il riferimento "exit9", ridà esattamente le righe clean e M2 di `results.csv`?
- **Previsione.** Sì, al bit: usa gli stessi split e una UCB già verificata equivalente.
- **Comando.** Frammento Python sui CSV `reference_alternatives.csv` e `reference_disagreements.csv` contro `results.csv`.
- **Risultato.** 3720 righe confrontate: differenza 0 su λ, rischio e savings. La somma delle tre quote dei disaccordi
  è il rischio, e vale l'identità accuracy persa = P(sbaglia solo l'exit) − P(sbaglia solo il final).
- **Convinzioni.** Previsione giusta. I riferimenti alternativi passano per lo stesso codice, cambiando solo i logit
  dell'ultimo exit.
- **Revisore indipendente.** Due tentativi (12–13 nov) si sono fermati sul limite d'uso del modello senza conclusioni;
  i loro script parziali sono in `parte1/revisore/`. Terzo tentativo il 13 nov su un altro modello, con un perimetro
  più stretto (8 controlli mirati).

### P1.18 (13 nov) Verifica delle obiezioni del revisore indipendente

- **Domanda.** Il revisore (Opus 5.5, 34 numeri ricalcolati, tutti coincidenti) obietta che i savings in più con
  l'ensemble sono in buona parte meccanici, perché gli exit 7 e 8 votano nel riferimento. È vero, e quanto resta senza
  sovrapposizione?
- **Previsione.** È vero su CIFAR100, dove il 20% degli input esce agli exit 7–8. Con i soli exit 1–6 usabili mi aspetto
  che resti circa metà del guadagno.
- **Comando.** `$PY reference_mechanical.py` da `outputs/fable_review/parte1/` (log in `reference_mechanical.log`).
- **Risultato.** Alla λ certificata, su CIFAR100 joint e mixed il 57–61% del calo del rischio viene dagli exit 7–8 (il
  revisore aveva 58%). Con i soli exit 1–6 il guadagno dell'ensemble è +0,032 su CIFAR100 (era +0,061) e +0,016 su
  CIFAR10 (era +0,023). Confermate anche le altre obiezioni: H4 dipende dalla contabilità degli audit; α spiega al più un
  quarto dello scarto del seed 2; a severity 1 il 42% dei casi di M1 su CIFAR10 sta sotto ε.
- **Convinzioni.** Previsione giusta. Avevo presentato l'ensemble come "riferimento migliore" senza controllare la
  sovrapposizione: un errore di confronto non equo, lo stesso tipo di errore che cercavo nel lavoro degli altri. Il
  rilievo R01 resta, ma motivato dalla composizione dei disaccordi e non dall'aumento dei savings.

---

## Applicazione delle correzioni (dopo l'ok di Davide del 13 novembre)

### X.01 (14 nov) Gli script di revisione portati nella repository ridanno i numeri dei piloti?

- **Domanda.** Gli script di `scripts/6_block3_review/` (scritti da un esecutore Opus a partire dai miei piloti di
  `parte1/`) producono gli stessi CSV?
- **Previsione.** Sì, identici: la logica non doveva cambiare.
- **Comando.** `zsh outputs/fable_review/esecuzione/run_all.sh <scratch>`, poi un confronto riga per riga.
- **Risultato.** Differenza massima 0 su `m2_validity.csv` (182.400 righe), `calibration_budget.csv` (149.040),
  `decomposition_orders.csv` (5.400) e sui tre CSV del riferimento (744, 3.720 e 11.160 righe). 136 test passano.
  Il costo dell'adattamento, ora misurato su 3 corruzioni per dataset, è 2.604 forward complete su CIFAR10 e 30.860 su
  CIFAR100 (da 29.177 a 32.188): il valore misurato prima su una sola corruzione (31.215) era rappresentativo.
- **Convinzioni.** Previsione giusta. I numeri della revisione ora si rigenerano dalla repository.

### X.02 (14 nov) M8b sui seed 1 e 2

- **Domanda.** Il guadagno di M8b misurato sul seed 0 si trasferisce ai seed 1 e 2? Il criterio della settimana 4
  passa in qualche setting?
- **Previsione**, scritta mentre il job gira e prima di vedere i risultati. Guadagno medio a severity ≥ 3 vicino a
  quello del seed 0 (0,047 con max softmax, tutti i dataset insieme), con scarti tra seed sotto 0,015. Il criterio non
  passa in almeno 7 setting su 8 per ogni seed. Su CIFAR10 joint_sd il seed 2, quello con α positivo agli exit 1–2,
  ha guadagno più basso degli altri due, perché le sue teste concordano già di più con l'exit 9.
- **Comando.** Dentro `run_all.sh`: `savings_under_shift.py --results outputs/savings_under_shift_v2/results.csv
  --only-adaptation --recompute-features --workers 4 --runs <16 reti>`. Partito alle 08:42.
- **Interruzione.** Il job è stato fermato alle 11:12 del 14 novembre dal watchdog della batteria (exit 143 dopo 9.001 s,
  con 4 worker il caricatore da 30 W non teneva: batteria da 80% a meno di 30% in 2,5 ore). Finite 11 reti su 16:
  tutte le 8 di CIFAR10 e 3 di CIFAR100 (disjoint seed 1, joint seed 1 e 2). `outputs/savings_under_shift/` non è
  stato toccato (md5 0de4091121255c7f4a1b89920d763b2a prima e dopo).
- **Ripresa.** 14 novembre alle 11:38, staccato dalla sessione, con 2 worker:
  `zsh outputs/fable_review/esecuzione/run_m8b_resume.sh 2`. Stato in `outputs/savings_under_shift_v2/STATUS.txt`.
  Mancano CIFAR100 disjoint seed 2, joint_sd seed 1 e 2, mixed seed 1 e 2 (circa 70–80 minuti per rete e worker).
- **Risultato parziale** (19 reti su 24, `python outputs/fable_review/esecuzione/m8b_three_seeds.py`, log in
  `outputs/block3_review/m8b_three_seeds_partial.log`). Max softmax, severity ≥ 3:
  - sui 5 setting con tutti e tre i seed il guadagno medio di M8b su M2 è 0,058 (seed 0), 0,053 (seed 1) e 0,053
    (seed 2); deviazione tra seed 0,005. Il seed 0 era leggermente fortunato;
  - criterio della settimana 4: CIFAR10 joint_sd 0,533 / 0,489 / 0,600 sui tre seed (in comune 73 casi su 135, 0,54);
    tutti gli altri setting falliscono su ogni seed disponibile. Con lo score agreement non passa nulla;
  - guadagno per severity su CIFAR10: da 0,022–0,024 (severity 1) a 0,073–0,088 (severity 5);
  - M8b sopra ε sulle righe di eval: 0,107 / 0,110 / 0,121 (max softmax, seed 0 / 1 / 2). Il seed 2 è un po' alto:
    da ricontrollare sulle righe fuori dalla calibrazione quando il job finisce.
- **Convinzioni.** Previsione giusta sul trasferimento del guadagno (scarto tra seed 0,005, sotto 0,015) e sul
  criterio (1 setting su 8, sempre lo stesso, al limite su ogni seed). Previsione su CIFAR10 joint_sd seed 2 vera solo
  di misura (0,066 contro 0,067 del seed 1): non sostiene la spiegazione con α. Da completare con le 5 reti mancanti.
- **Completato il 14 novembre alle 15:18:** il risultato sulle 24 reti è nella voce X.04.

### X.03 (14 nov) Pauser della batteria per il job M8b

- **Motivo.** Con 2 worker la batteria scendeva ancora (39% alle 11:38, 37% alle 11:56, in carica): il watchdog al 30%
  avrebbe ucciso il job perdendo le reti in corso.
- **Cosa.** `outputs/fable_review/esecuzione/battery_pauser.sh 62798`, avviato alle 11:57 e staccato dalla sessione:
  sotto il 33% sospende il processo python e i worker (SIGSTOP), al 45% li riprende (SIGCONT). Il watchdog al 30% resta
  come ultima sicurezza. Righe `PAUSED` e `RESUMED` in `outputs/savings_under_shift_v2/STATUS.txt`.
- **Correzione a una stima.** Una rete CIFAR100 richiede circa 90 minuti con 2 worker (3 corruzioni su 15 in circa 18
  minuti), non 70–80: fine prevista tra le 16:00 e le 17:00, più le pause.
- **Correzione a un numero di `01_revisione.md`.** CRC sopra ε nel 38–47% delle calibrazioni (era scritto 39–47%; il
  minimo è 0,377 su CIFAR100 con 250 righe).

## Fase A del piano v3 (chat nuova, dal 14 novembre)

### X.04 (14 nov) M8b sulle 24 reti e violazioni di M8b

- **Contesto.** Sessione nuova (Fable 5.2), ripresa dal pacchetto `handoff/`. Alle 12:12 il job M8b è vivo: 12
  corruzioni su 75, CIFAR100 disjoint seed 2 e mixed seed 1 a 6 su 15 ciascuna, batteria al 33%. Ritmo misurato: circa
  78 minuti per rete e worker. Fine stimata tra le 16:30 e le 18:00, con le pause della batteria.
- **Domanda 1.** Le 5 reti mancanti di CIFAR100 cambiano il quadro del parziale a 19 reti (voce X.02)?
- **Previsione 1**, scritta prima che il job finisca e prima di vedere una sola riga nuova.
  - Sulle 5 reti nuove il guadagno di M8b su M2 a severity ≥ 3 (max softmax) sta entro ±0,010 dal valore del seed 0
    dello stesso setting: 0,013 per disjoint, 0,040 per joint_sd, 0,030 per mixed. Nessuna passa il criterio della
    settimana 4.
  - Sulle 24 reti lo scarto tra seed del guadagno (media sui setting) resta ≤ 0,008. Il criterio passa solo per
    CIFAR10 joint_sd, al limite.
- **Domanda 2.** M8b sopra ε sulle righe di eval nel 10,7%, 11,0% e 12,1% dei casi (seed 0, 1, 2, max softmax, 19
  reti): è un problema di M8b o il rumore delle 3.000 righe di eval?
- **Nota di metodo.** Il pacchetto chiede di misurarlo "fuori dalla calibrazione", come per M2 (7.000 righe). Per M8b
  non si può: le 4.000 righe di adapt allenano le teste (3.000) e scelgono L2 e score (1.000), quindi su quelle righe
  il rischio è distorto verso il basso. Le sole righe pulite sono le 3.000 di eval, e `results.csv` salva solo il
  rischio su eval. La garanzia UCB vale comunque condizionatamente alle teste adattate, perché le righe di cal non
  entrano nell'adattamento (DEDOTTO dal codice di `savings_under_shift.py`, da ricontrollare leggendo il ramo M8b).
- **Previsione 2.** Sugli stessi casi (rete, corruzione, severity, split 0–2, ε = 0,05, max softmax) il tasso di M8b
  supera quello di M2 di meno di 0,03 per ogni seed. I casi sopra ε di più di 2 errori standard binomiali sono ≤ 2% per
  ogni seed. L'intervallo al 95% del tasso del seed 2, con bootstrap a cluster su (rete, corruzione), contiene 0,10.
- **Soglia di allarme**, fissata prima. Differenza M8b − M2 ≥ 0,03 su un seed, oppure più del 3% dei casi sopra ε di
  più di 2 errori standard. Se scatta, propongo a Davide un job dedicato (M8b con molti split su pochi domini,
  salvando le loss per riga) e non lo lancio da solo.
- **Comando.** Dopo `END exit=0` in `STATUS.txt`: `python outputs/fable_review/esecuzione/m8b_three_seeds.py`,
  `savings_figures.py --results outputs/savings_under_shift_v2/results.csv --out-dir outputs/block3_review/figures_v2`
  e `python outputs/fable_review/fase_a/m8b_violations.py`.
- **Verifica della nota di metodo** (VERIFICATO, `savings_under_shift.py` righe 134–147): `adapt_heads` riceve solo
  righe di adapt (3.000 per le teste, 1.000 tenute da parte per L2 e score); λ si sceglie con `ucb_index` sulle righe
  di cal; la misura è sulle righe di eval. Le righe di cal non entrano nell'adattamento.
- **Risultato parziale sulla domanda 2** (19 reti su 24, job in pausa per la batteria dalle 12:19;
  `python outputs/fable_review/fase_a/m8b_violations.py`, log `fase_a/m8b_violations_partial19.log`). Max softmax,
  4.275 casi appaiati, intervalli al 95% con bootstrap a due vie su reti e corruzioni:
  - M8b sopra ε: 0,107 / 0,110 / 0,121 (seed 0, 1, 2). M2 sugli stessi casi: 0,107 / 0,120 / 0,117. Differenza
    +0,000 / −0,010 / +0,004; su tutti i seed −0,002 [−0,022, +0,018];
  - sopra ε + 2 errori standard (0,058): M8b 1,1% / 1,3% / 1,0%, M2 1,6% / 1,3% / 1,0%;
  - rischio medio su eval 0,043 per entrambi; rischio peggiore 0,077 (M8b) e 0,085 (M2);
  - con lo score agreement la differenza sul seed 2 è +0,021 [−0,011, +0,056], dentro il rumore e sotto la soglia.
- **Convinzioni (parziali).** Previsione 2 giusta sulle 19 reti: nessun allarme. Lo 0,121 del seed 2 non è di M8b:
  M2 sugli stessi casi e sulle stesse righe di eval è a 0,117. È il rumore delle 3.000 righe di eval, che per M2 la
  Parte 1 aveva già misurato (8,9% su eval contro 6,7% fuori dalla calibrazione). Nota: sugli split 0–2 e su queste
  reti M2 è all'11%, un po' sopra l'8–11% del README (5 split, 24 reti).
- **Fine del job.** Ripartito alle 13:01 (batteria al 45%), finito alle 15:18: 75 corruzioni su 75, ultima riga del
  log `saved outputs/savings_under_shift_v2/results.csv 237240 rows`. `STATUS.txt` riporta `END exit=145
  original_results_unchanged=yes`, non `exit=0`: 145 è 128 + 17, cioè il SIGSTOP del pauser delle 12:19, che `wait` di
  zsh restituisce al posto del codice di uscita vero (DEDOTTO). Il job è completo: lo dicono i controlli sotto, non
  il codice di uscita.
- **Controlli sul file** (`python outputs/fable_review/fase_a/check_a1.py`, log accanto): 237.240 righe = 230.040 +
  7.200; ogni riga dell'originale è nel file nuovo con gli stessi valori (differenza massima 0); le righe nuove sono
  solo M8b dei seed 1 e 2, 450 per rete, nessun valore mancante; L2 sempre nella griglia, al bordo nel 3,1% / 2,7% /
  2,4% delle teste (seed 0 / 1 / 2). md5 di `outputs/savings_under_shift/results.csv` invariato.
- **Risultato finale, domanda 1** (24 reti; `outputs/block3_review/m8b_three_seeds.log`, max softmax, severity ≥ 3).
  - Guadagno di M8b su M2 per seed, media sugli 8 setting: 0,047 / 0,043 / 0,044. Scarto tra seed, media sui setting:
    0,003. CIFAR10: 0,065 / 0,056 / 0,057. CIFAR100: 0,029 / 0,031 / 0,031. Con lo score agreement: 0,029 / 0,026 /
    0,026.
  - Le 5 reti nuove: CIFAR100 disjoint seed 2 0,0136 (seed 0: 0,0133); joint_sd seed 1 e 2 0,0423 e 0,0420 (seed 0:
    0,0395); mixed seed 1 e 2 0,0289 e 0,0302 (seed 0: 0,0300). Nessuna passa il criterio (quote 0,04, 0,42, 0,31,
    0,33, 0,29).
  - Criterio della settimana 4: passa solo CIFAR10 joint_sd sui seed 0 e 2 (0,533 / 0,489 / 0,600; insieme 0,54).
  - Il mio ricalcolo indipendente del guadagno (join da `results.csv`) coincide con `m8b_three_seeds_week4.csv` a
    1e-17, e questo con `figures_v2/adaptation_test.csv` (48 righe, stessi verdetti).
- **Risultato finale, domanda 2** (24 reti, 5.400 casi appaiati; `fase_a/m8b_violations.log`). Max softmax:
  - M8b sopra ε: 0,107 / 0,092 / 0,108. M2 sugli stessi casi: 0,107 / 0,107 / 0,109. Differenza +0,000 / −0,015 /
    −0,001; su tutti i seed 0,102 contro 0,107, differenza −0,005 [−0,022, +0,013].
  - Sopra ε + 2 errori standard: M8b 1,1% / 1,0% / 0,9%; M2 1,6% / 1,0% / 0,9%. Rischio medio 0,0425 e 0,0428.
  - Score agreement: differenza sul seed 2 +0,016 [−0,009, +0,041]; su tutti i seed +0,006 [−0,010, +0,021].
- **Convinzioni.** Tutte e due le previsioni giuste, nessun allarme. Il guadagno di M8b è una proprietà stabile del
  metodo (scarto tra seed 0,003 su un guadagno di 0,045), non del seed 0, che era un po' fortunato su CIFAR10 (0,065
  contro 0,056). Lo 0,121 del seed 2 nel parziale veniva dalle poche reti: sulle 24 è 0,108, come M2. DUB1 e DUB8 del
  pacchetto di handoff sono chiusi. Resta vero che M8b non ripaga il suo compute (voce P1.14).

### X.05 (14 nov) Piccole chiusure (A4 del piano v3)

- `outputs/fable_review/esecuzione/compare_b4.py` corretto (le chiavi con valori mancanti diventavano float e
  `"|".join` si fermava) e rieseguito: differenza 0 su tutti e sei i confronti (`esecuzione/compare_b4_rerun.log`).
  Chiude BUG1 del pacchetto di handoff.
- Lo stesso log spiega uno scarto tra i testi approvati il 13 novembre e gli output della repository: il pilota
  misurava il costo del training su un solo dominio (frost: 2.682 e 31.215 forward complete, payback 83.077 e
  1.687.912), lo script della repository fa la media di 3 domini (2.604 e 30.860, payback 81.403 e 1.669.064).
  Nei testi vanno i numeri della repository.
- Riga `.gstack/` in `.gitignore`: tolta alle 12:15 su decisione di Davide, rimessa dallo strumento gstack verso le
  12:30 (voce X.06). Davide ha poi deciso di lasciarla. Stato finale: la riga c'è, come all'inizio della sessione.
- Intestazioni "Stato" delle note `parte2/L2` e `parte2/L3` portate a "completa".

### X.06 (14 nov) La frase del README sul budget di Kubaty et al., verificata sulla fonte

- **Domanda.** La nota `parte2/L4_fonti_citate.md` (RIPORTATA da un agente) dice che in Kubaty et al. 2025 i regimi hanno
  lo stesso tetto nominale di epoche ma l'early stopping decide la durata reale (523, 1610, 1166 epoche). Regge sulla
  fonte primaria? Serve per la frase del README (riga 178, "same total budget").
- **Previsione.** Sì, la nota è corretta.
- **Come.** Skill `/browse` di gstack, 14 novembre 2026, due pagine lette da me:
  `https://raw.githubusercontent.com/kamadforge/early-exit-benchmark/main/scripts/regimes_resnet.py` (file intero) e
  `https://arxiv.org/html/2407.14320v2` (paragrafi con early stopping, patience e le epoche).
- **Risultato** (VERIFICATO).
  - Codice: `early_stopping = True`, patience 50, metrica accuracy; ottimizzatore `adam` con lr 0,001 e weight decay 0,
    cosine con warm restart, mixup e cutmix; ResNet-50 su TinyImageNet. Epoche per regime: disjoint 1000 (backbone) +
    1000 (teste), joint 2000, mixed 1000 + 1000. Il tetto nominale è quindi uguale tra i regimi.
  - Paper: l'early stopping è usato in ogni esperimento per una convergenza equa tra i regimi; per ResNet-50 su
    TinyImageNet riporta in media 523 (± 156) epoche per disjoint, 1610 (± 395) per joint e 1166 (± 136) per mixed.
- **Convinzioni.** Previsione giusta. Nel README la frase può portare anche i tre numeri, con dataset e architettura.
- **Effetto collaterale.** Lo strumento gstack, lanciato dalla radice della repository, ha rimesso la riga `.gstack/`
  in `.gitignore` (che Davide aveva chiesto di togliere). Lanciato da una cartella fuori dalla repository non la
  tocca. Davide, informato, ha deciso di lasciarla.

### X.07 (14 nov) Con M8a il traffico si sposta dagli exit 1–4 agli exit 5–8?

- **Domanda.** Il rilievo R23 e il testo approvato per il README dicono che con M8a il traffico passa dagli exit 1–4
  agli exit 5–8. L'esecutore di A2 non ha trovato un log che lo mostri e ha omesso la frase. È vero?
- **Previsione.** Sì (è l'affermazione della revisione): meno traffico agli exit 1–4 e più agli exit 5–8.
- **Comando.** `python outputs/fable_review/fase_a/m8a_traffic.py` (colonne `pi_1..pi_9` di `results.csv`; M8a contro
  M2 con lo stesso score agreement, ε = 0,05, severity 3–5, 24 reti, 5.400 casi appaiati per metodo).
- **Risultato.** CIFAR10: exit 1–4 da 0,241 a 0,172, exit 5–8 da 0,463 a 0,596, exit 9 da 0,296 a 0,233; savings da
  0,272 a 0,255. CIFAR100: da 0,085 a 0,054, da 0,294 a 0,361, da 0,621 a 0,585; savings da 0,110 a 0,098. Vale in 24
  reti su 24.
- **Convinzioni.** Previsione giusta, con un dettaglio in più: con M8a arrivano in fondo meno input, non di più. I
  savings calano perché si svuotano gli exit più economici. La frase è nel README con i numeri.

### X.08 (14 nov) Esecuzione di A2 (parte senza M8b) e verifica

- **Come.** Specifica scritta da Fable; esecutore Opus 5.5 a effort xhigh (un agente, 61 chiamate, circa 11 minuti,
  224.000 token); verifica di Fable sul diff e sui log. Un primo esecutore a effort high era stato fermato dopo
  pochi secondi, su indicazione di Davide ("usa opus xhigh"), senza aver toccato nulla.
- **Esito.** 23 modifiche applicate (R1–R18 nel README, F1–F5 in `docs/findings.md`). L'esecutore ha trovato e
  corretto imprecisioni della specifica e del testo approvato: l'exit 9 sta 0,5–1,5 punti sotto l'exit 8 su CIFAR100
  (non 0,8–1,5); le quote dei disaccordi sono 46–50% e 28–31% in joint, mixed e disjoint, 39% e 38% in joint_sd; il
  41–95% tra le corruzioni vale solo per CIFAR10 (su CIFAR100 va da 0,23 a 0,91); il payback di 81.000 input conta
  anche le righe di adattamento; tre avversari su cinque sono adattivi.
- **Verifica.** Tutti i numeri che ho riletto sui log coincidono. Dettaglio in
  `outputs/fable_review/fase_a/A2_scostamenti_e_fonti.md`, sezione 6.
- **Righe per il log del documento** (A3): bozza in `outputs/fable_review/A3_righe_log_documento.txt`, controllo dei
  caratteri vietati passato (restano due falsi positivi sui nomi di file con il punto). Per la riga sulla
  letteratura ho riletto due fonti: la submission 22466 in locale (righe 122–124 e 426–428 di `paper.txt`) e
  l'abstract di Einbinder et al. 2024 su arXiv. **Precisazione:** l'abstract parla di "early time series
  classification"; l'applicazione è uno stop anticipato su una sequenza letta da un LLM (nota L3, sezione VII-B del
  paper, RIPORTATA), non un early exit tra i layer come dice in breve il pacchetto di handoff.

---

## Fase B del piano v3: piloti offline (dal 14 novembre sera)

Checkpoint 2b chiuso il 14 novembre alle 21:30: Davide ha risposto "Sì, con le varianti" alla domanda sui piloti B1
e B2. Le varianti approvate per B2 sono due: il numero di righe di adattamento (500, 1.000, 2.000, 4.000) e le sole
righe gratis, cioè gli input che arrivano comunque all'exit 9. L'ipotesi D1 resta scritta com'è nel piano.
Modo di lavoro: Fable scrive disegno, previsioni e verifiche; un esecutore Opus a effort xhigh scrive il codice e fa
la prova su pochi casi; il job completo gira in shell in background; un solo agente alla volta.

### B1.01 (14 nov) Il bilancio computazionale dei savings sotto shift

- **Domanda.** Dove si perde il margine tra ciò che la rete potrebbe risparmiare sotto shift e ciò che M2 certifica?
- **Cosa so già, prima di lanciare** (misure esistenti, non previsioni).
  - Controllo preliminare del 10 novembre (seed 0, joint e joint_sd, 8.000 righe, senza UCB): l'oracolo per input a
    disaccordo zero dà 0,78 sul pulito, 0,70 a severity 3 e 0,65 a severity 5 su CIFAR10; 0,60–0,63, 0,50–0,54 e
    0,47–0,51 su CIFAR100 (`pilots/00_controlli_preliminari/output_probe_plan_premises.txt`).
  - L'oracolo di soglia condivisa sta 0,009–0,015 sopra M2 (`outputs/block3_review/m2_validity_oracles.csv`).
  - M2 con meno righe di calibrazione e CRC: `outputs/block3_review/calibration_budget_crc_against_ucb.csv` (a
    severity 3 su CIFAR10 UCB dà 0,262 con 250 righe e 0,311 con 3.000; CRC 0,32 con 100 righe).
  - Famiglie fisse di soglie proporzionali al costo: da −0,06 a +0,04 sulla soglia condivisa (controllo 3).
- **Livelli misurati**, tutti sulle stesse 3.000 righe di eval dei 5 split di `results.csv`, max softmax, ε = 0,05,
  δ = 0,1, 24 reti, pulito e severity 1, 3, 5 delle 15 corruzioni standard:
  1. M2 con UCB a 3.000, 1.000 e 250 righe, e CRC a 3.000, 250 e 100 righe;
  2. oracolo di soglia condivisa (la λ della griglia con più savings e rischio ≤ ε sulle righe di fit);
  3. oracolo con 8 soglie per exit (la migliore regola trovata con rischio ≤ ε sulle righe di fit);
  4. oracolo per input a disaccordo zero (primo exit che concorda con l'exit 9) e con budget ε (il 5% degli input
     dove uscire all'exit 1 fa risparmiare di più);
  5. accanto, con le label solo per valutare: soglia condivisa a parità di accuracy persa, assoluta (0,05) e
     relativa (5% dell'accuracy dell'exit 9); la frontiera accuracy contro compute lungo la griglia di λ;
  6. savings netti di M2 su orizzonti di 10.000, 100.000 e 1 milione di input, per ogni n_cal.
- **Scostamenti dal piano, dichiarati prima.** Gli oracoli si scelgono sulle 7.000 righe fuori da eval (adapt più
  cal) e si misurano sulle 3.000 di eval, non metà e metà: così ogni livello è misurato sulle stesse righe di M2 e i
  confronti sono appaiati. L'oracolo per input non ha nulla da scegliere. Il riferimento senza sovrapposizione non
  si ricalcola (è in `outputs/block3_review/reference_mechanical.csv`). L'oracolo con 8 soglie è un limite inferiore
  dell'ottimo della classe (ottimizzazione euristica): basta per dire che una regola a 8 soglie ci arriva.
- **Ipotesi del piano.** Il salto tra oracolo per input e oracolo di soglia condivisa vale almeno il 60% del margine
  totale a severity 3. Nota: con i numeri già noti è quasi certa, quindi come ipotesi è debole. Le previsioni sotto
  sono quelle che possono sbagliare.
- **Previsioni**, scritte prima di avere il codice.
  1. Margine totale = oracolo per input con budget ε − M2 (3.000 righe). La quota tra oracolo per input e oracolo di
     soglia condivisa è il 93–97% a severity 3 su entrambi i dataset. Il prezzo statistico (soglia condivisa − M2)
     pesa il 2–4%.
  2. Oracolo per input con budget ε, media sulle 12 reti del dataset. CIFAR10: 0,80–0,84 pulito, 0,72–0,77 a
     severity 3, 0,67–0,72 a severity 5. CIFAR100: 0,62–0,68, 0,52–0,59, 0,49–0,56. Il budget ε aggiunge 0,040–0,044
     al disaccordo zero. Su CIFAR10 a severity 5 il salto verso la soglia condivisa è 0,42–0,50 (il piano diceva
     0,30–0,40 a disaccordo zero: mi aspetto 0,38–0,45). Il regime disjoint ha l'oracolo per input più basso dei
     quattro su entrambi i dataset.
  3. Le 8 soglie per exit, misurate su eval, guadagnano sulla soglia condivisa 0,020–0,045 su CIFAR10 e 0,015–0,040
     su CIFAR100 a severity 3, e almeno 0,02 in 5–7 setting su 8 sia sul pulito sia a severity 3. Chiudono solo il
     5–10% del salto verso l'oracolo per input. È la previsione più incerta: se il guadagno sta sotto 0,02 nella
     maggior parte dei setting, il pilota B4 (soglie per exit certificate) parte già morto.
  4. Savings netti di M2 con UCB: su 10.000 input conviene calibrare con al più 500 righe; da 100.000 in su con
     2.000–3.000. CRC con 100 righe batte ogni UCB fino a 100.000 input, ma garantisce solo il rischio in media.
     (Si calcola già dai numeri esistenti: il pilota lo mette in tabella.)
  5. A parità di accuracy persa assoluta (0,05) i savings a severity 3 sono 0,55–0,70 su CIFAR10 e 0,65–0,80 su
     CIFAR100, cioè non calano rispetto al pulito. A severity 5 su CIFAR100 la λ scelta è il minimo della griglia
     (tutto esce all'exit 1) in almeno il 30% dei domini. Con l'accuracy persa relativa la quota di savings rimasta
     a severity 3 è 0,80–0,95 su CIFAR10 e almeno 0,95 su CIFAR100.
  6. Dentro una rete, la correlazione di rango tra savings di M2 e accuracy dell'exit 9 sui 45 domini shiftati ha
     mediana tra 0,80 e 0,95.
- **Kill.** Nessuno: è una misura.
- **Comando e cartella.** `outputs/fable_review/pilots/B1_bilancio/` (script `b1_levels.py` e `b1_summary.py`,
  scritti dall'esecutore su mia specifica). Solo logit e label esistenti, CPU, 3 processi.
- **Esecuzione del codice** (22:01–22:10). Esecutore Opus 5.5 a effort xhigh, un agente, 25 chiamate, circa 190.000
  token. Si era fermato subito sul limite d'uso alle 21:45 senza scrivere nulla ed è ripartito alle 22:01. Prova su 2
  reti e 3 corruzioni: 30 test passati, controlli (a), (b), (d) con differenza 0. Ho riletto per intero
  `b1_levels.py` e `b1_summary.py`: seguono la specifica. Scelte dell'esecutore in `NOTE_ESECUTORE.md` (rischio in
  float64, passata greedy finale, conteggio di "tutto esce all'exit 1").
- **Difetto del mio disegno trovato sulla prova, prima del job completo** (verso le 22:11). Nella prova il livello delle 8
  soglie tiene su eval un rischio di 0,054–0,056, la soglia condivisa di 0,047–0,049. Le 8 soglie riempiono il
  budget sulle righe di fit e su righe nuove lo sforano: il confronto che avevo specificato non è a pari rischio.
  Con circa 3 punti di savings per punto di rischio, quello scarto vale da solo circa 0,02, cioè quanto il guadagno
  che volevo misurare. Errore dello stesso tipo dell'ensemble della voce P1.18 (confronto non equo).
- **Correzione, scritta prima di vedere i risultati completi.** Aggiungo io due livelli e tengo anche quello vecchio:
  - `shared_cal`: la soglia condivisa con più savings e rischio ≤ ε sulle sole righe di cal;
  - `exit_path_cal`: le 8 soglie seguono un cammino imparato sulle sole righe di adapt (le soluzioni lagrangiane
    lungo μ) e sulle righe di cal si sceglie solo dove fermarsi. Su cal si sceglie un numero, come per la soglia
    condivisa, quindi il rischio su righe nuove non è gonfiato dagli 8 parametri. Nessuna passata greedy.
  Il guadagno equo delle soglie per exit è `exit_path_cal − shared_cal` su eval, con i due rischi su eval accanto.
  È anche il limite superiore di E2 del pilota B4 (stesso schema, con il rischio empirico al posto della UCB).
  La previsione 3 si giudica su questo numero; riporto anche quello vecchio.
- **Due ritocchi alla correzione, ancora sulla sola prova** (2 reti, 3 corruzioni; tra le 22:11 e le 22:16).
  - Prima versione: il cammino si fermava a un rischio su eval di 0,045 contro 0,049 della soglia condivisa, perché
    le soluzioni saltano da un μ all'altro e lasciano budget inutilizzato. Ora cerco anche lungo il segmento tra
    l'ultima soluzione ammissibile su cal e la successiva (200 punti): resta una sola posizione lungo un solo cammino.
  - Così però il cammino riempiva il budget su cal (0,0500) e la soglia condivisa della griglia no (0,0486). Ho
    aggiunto `shared_cal_fine`, la soglia condivisa cercata con la stessa finezza tra due λ della griglia.
  - Sulla prova i due termini ora coincidono nel rischio: 0,0500 su cal per entrambi, 0,0508 e 0,0506 su eval
    (CIFAR10), 0,0496 e 0,0501 (CIFAR100). Il guadagno equo è `exit_path_cal − shared_cal_fine`.
  - 36 test passati (6 nuovi per il cammino). Job completo lanciato alle 22:16:46 con 3 processi, batteria all'80%
    (orari letti da `date`, non stimati).
- **Job completo.** Finito alle 22:28:26, 698 secondi, exit 0, batteria all'80%. 82.800 righe (24 reti, 46 domini, 5
  split, 15 livelli). md5 di `outputs/savings_under_shift/results.csv` invariato, `git status` identico all'inizio.
- **Verifiche** (VERIFICATO).
  - Dello script (`b1_summary.log`): M2 coincide con `results.csv` su 5.520 righe (differenza 0); i cinque livelli
    di calibrazione coincidono con `calibration_budget.csv` (5.520 righe ciascuno, differenza 0); il prezzo
    statistico coincide con `m2_validity_oracles.csv` entro 0,0005; invarianti rispettati su 5.520 casi.
  - Mie, con codice separato (`pilots/verifica_fable/b1_check.py`, log accanto): su 16 casi estratti a caso ho
    ricalcolato 13 quantità per caso dai logit, con la UCB lenta di riferimento al posto di quella veloce: 0
    differenze sopra 1e-9. Le soglie salvate, riapplicate, ridanno i valori del file.
  - Un secondo ottimizzatore delle 8 soglie scritto da me (cammino greedy su quantili) trova sulle righe di fit in
    media 0,010 di savings in meno di quello del pilota, ma in un caso su 16 ne trova 0,005 in più: quello del
    pilota è un limite inferiore, non l'ottimo. Nel confronto equo il mio cammino greedy non batte la soglia
    condivisa (−0,005 sui 16 casi, contro +0,019 del cammino del pilota): il guadagno delle soglie per exit dipende
    da quanto è buono il cammino.
- **Risultato** (`b1_summary.log`, `verifica_fable/b1_extra.log`; savings medi su eval; CIFAR10 e poi CIFAR100, a
  severity 0 / 1 / 3 / 5; intervalli al 95% con bootstrap a due vie su reti e corruzioni).
  - Oracolo per input a disaccordo zero: 0,763 / 0,725 / 0,677 / 0,617 e 0,576 / 0,520 / 0,466 / 0,432. Con budget
    ε: 0,800 / 0,765 / 0,718 / 0,659 e 0,620 / 0,564 / 0,510 / 0,476.
  - 8 soglie per exit, confronto equo: 0,557 / 0,452 / 0,349 / 0,265 e 0,269 / 0,203 / 0,151 / 0,120. Soglia
    condivisa scelta allo stesso modo: 0,545 / 0,435 / 0,329 / 0,236 e 0,247 / 0,184 / 0,134 / 0,102.
  - M2 con 3.000 righe: 0,526 / 0,415 / 0,311 / 0,219 e 0,231 / 0,170 / 0,123 / 0,091. Con 250 righe: 0,460 /
    0,361 / 0,262 / 0,180 e 0,200 / 0,138 / 0,096 / 0,067.
  - **Margine a severity 3.** Totale (oracolo per input con budget ε meno M2): 0,407 e 0,388. Il prezzo statistico è
    0,0145 e 0,0091, cioè il 3,6% e il 2,3%. Il 96,4% e il 97,7% sta tra l'oracolo per input e la soglia condivisa.
  - **Soglie per exit, guadagno equo** (stesso rischio su eval, 0,050–0,052 per entrambi): CIFAR10 +0,012 [0,009,
    0,015] sul pulito, +0,017, +0,019 [0,012, 0,027] a severity 3, +0,029 [0,019, 0,040] a severity 5. CIFAR100
    +0,022 [0,017, 0,027], +0,019, +0,017 [0,014, 0,020], +0,018 [0,015, 0,021]. Positivo nel 90–98% dei casi.
    Almeno 0,02 in 3 setting su 8 sul pulito e in 3 su 8 a severity 3 (4 su 8 a severity 1 e 5). Chiude il 5,0% e il
    4,5% del salto verso l'oracolo per input a severity 3. Il massimo è CIFAR10 joint_sd a severity 5 (+0,051).
  - **Il confronto non equo avrebbe detto altro**: +0,029 e +0,025 a severity 3, con 8 setting su 8 sopra 0,02.
  - **Capacità contro riconoscimento.** Dal pulito a severity 3 l'oracolo per input perde 0,086 [0,067, 0,106] su
    CIFAR10 e 0,110 [0,083, 0,141] su CIFAR100; M2 perde 0,215 [0,173, 0,256] e 0,109 [0,086, 0,133]. All'oracolo
    resta l'89% e l'81% dei savings puliti, a M2 il 59% e il 53%.
  - **A parità di accuracy persa** (label solo per valutare): con il vincolo assoluto 0,644 / 0,602 / 0,603 / 0,691
    e 0,499 / 0,523 / 0,606 / 0,731; con quello relativo 0,631 / 0,568 / 0,536 / 0,610 e 0,452 / 0,443 / 0,473 /
    0,568. Quota rimasta a severity 3: 0,94 e 1,22 (assoluto), 0,85 e 1,05 (relativo). A severity 5 tutto esce
    all'exit 1 nel 35% e nel 53% dei casi con il vincolo assoluto, nel 24% e nel 23% con quello relativo.
  - Alla λ di M2 il sistema perde 0,015 di accuracy rispetto all'exit 9 su CIFAR10 pulito, 0,005 a severity 3, e ne
    guadagna 0,002 a severity 5; su CIFAR100 ne guadagna 0,006–0,007 a ogni severity.
  - Savings netti di M2 (da `calibration_budget.csv`): su 10.000 input il miglior n_cal con UCB è 500 su CIFAR10
    (0,271 a severity 3 contro 0,217 con 3.000) e 1.000 su CIFAR100 sotto shift; da 100.000 in su è 3.000. CRC con
    100 righe dà 0,317 su 10.000 input e batte ogni UCB fino a 100.000.
  - Correlazione di rango tra savings di M2 e accuracy dell'exit 9, dentro una rete: mediana 0,949 (da 0,823 a
    0,976). Errore di una retta lasciando fuori una corruzione: 0,010–0,020 su CIFAR100, 0,031–0,069 su CIFAR10.
  - Per regime, l'oracolo per input più basso è disjoint (0,712 contro 0,775–0,783 su CIFAR10 pulito; 0,490
    contro 0,591–0,619 su CIFAR100).
  - Su CIFAR100 l'exit 8 concorda con l'exit 9 solo sul 79,5% degli input puliti e sul 65% a severity 5, pur
    essendo un po' più accurato (0,648 contro 0,645; 0,243 contro 0,236).
- **Convinzioni.**
  - Previsione 1 giusta su CIFAR10 (96,4%), di poco fuori su CIFAR100 (97,7% contro 93–97%). L'ipotesi del piano
    regge, ma era debole.
  - Previsione 2 troppo alta di circa 0,01 sotto shift (0,718 contro 0,72–0,77; 0,659 contro 0,67–0,72; 0,510
    contro 0,52–0,59; 0,476 contro 0,49–0,56): avevo estrapolato dai due regimi migliori. Giusti il salto su
    CIFAR10 a severity 5 (0,427) e il regime disjoint ultimo.
  - **Previsione 3 sbagliata**: mi aspettavo almeno 0,02 in 5–7 setting su 8, sono 3 su 8. Il guadagno equo è
    reale (intervalli lontani da zero, scarto tra seed 0,004) ma piccolo: 0,017–0,019 a severity 3, sotto la soglia
    di 0,02 del piano. È il limite superiore di E2 del pilota B4, che quindi parte sotto il suo criterio di kill
    nella maggior parte dei setting. Proposta per il Checkpoint 3a: ridurre B4 a questa misura.
  - Previsione 4 giusta, tranne CIFAR100 sotto shift (1.000 righe e non 500).
  - Previsione 5 giusta in 4 punti su 5: su CIFAR100 a severity 3 il vincolo assoluto dà 0,606 e non 0,65–0,80.
  - Previsione 6 giusta (0,949).
  - **Cosa cambia.** Il margine non sta nel prezzo statistico (2–4%) né nella forma della regola a soglie (5%):
    sta nel riconoscere quali input concordano. E sotto shift la rete perde poca capacità di uscire presto (resta
    l'81–89%) mentre i savings certificati si dimezzano. Due strade restano aperte: cambiare lo score (pilota B3,
    classificatore dell'accordo) e cambiare le teste (pilota B2). Che i savings a parità di accuracy persa non
    calino sotto shift era atteso dalla letteratura (22466); qui è misurato su 24 reti.
  - **Errore mio da non ripetere.** Un oracolo con più parametri scelto sulle stesse righe ha su righe nuove un
    rischio più alto: i confronti tra regole vanno fatti a pari rischio su righe nuove. Il difetto è venuto fuori
    dalla nota dell'esecutore, non da me.

### B2.01 (14 nov) Una fase di adattamento che ripaga il suo costo

Scritta tra le 21:45 e le 22:00 del 14 novembre, prima che esista una riga di codice di B1 o di B2 (l'esecutore di
B1 si è fermato subito sul limite d'uso del modello, senza scrivere nulla, ed è ripartito alle 22:01).

- **Domanda.** Quale adattamento recupera savings al netto del suo compute, e lo fa migliorando l'accuracy o copiando
  gli errori del riferimento?
- **Perimetro.** Seed 0 (ha le feature di CIFAR-C salvate), regimi joint e joint_sd, CIFAR10 e CIFAR100: 4 reti. Le 15
  corruzioni standard a severity 3 e 5: 120 domini. Split 0 e 1 di `results.csv`. Max softmax, ε = 0,05, δ = 0,1, λ
  con UCB sulle 3.000 righe di cal, misura sulle 3.000 di eval. Split 0: tutte le configurazioni. Split 1: solo M8b
  com'è e le varianti economiche.
- **Cosa so già** (misure esistenti sulle stesse 4 reti, split 0–2, `outputs/savings_under_shift_v2/results.csv`).
  - Guadagno di M8b su M2: CIFAR10 +0,050 a severity 3 e +0,096 a severity 5; CIFAR100 +0,024 e +0,052.
  - Sotto M2 arriva all'exit 9 il 28–56% degli input su CIFAR10 e il 62–71% su CIFAR100.
  - M8b sceglie L2 = 0,01 nel 35–81% delle teste, poi 0,1.
  - Controllo preliminare del 10 novembre (2.000 righe, senza UCB, solo joint): la BN adaptation alza l'accuracy
    dell'exit 9 di 7–17 punti ma i savings a consistency risk fissato solo di 0,00–0,03.
- **Le configurazioni.** Un nome dice tre cose: cosa si adatta, come, su quali righe.
  - Cosa: `head` (il layer lineare della testa, come M8b) oppure `logit` (una mappa affine C × C sui logit della
    testa, candidato (d) del piano).
  - Come: `grid` (5 valori di L2 scelti su righe tenute da parte, 100 iterazioni, come M8b) oppure `k5`, `k10`, `k20`
    (un solo L2 = 0,01 e 5, 10, 20 iterazioni, candidato (e)).
  - Righe: `n4000`, `n2000`, `n1000`, `n500` (le prime righe di adapt; tre quarti per il fit e un quarto tenuto da
    parte, che le varianti `k` non usano e non pagano); `free` (le righe di adapt che sotto la λ di M2 arrivano
    comunque all'exit 9: costo zero); `match` (lo stesso numero di righe, ma casuali: è il controllo di `free`).
  - BN: `bn_n500` … `bn_n4000` (statistiche ristimate sulle prime righe di adapt, candidato (a)); `bn_traffic`
    (statistiche di ogni blocco stimate solo sugli input che sotto M2 arrivano a quel blocco: costo zero, è il sotto
    pilota opzionale del piano e la versione "righe gratis" della BN); `c_…` (BN con 4.000 righe e poi teste o mappa
    affine verso il riferimento adattato, candidato (c)).
  - `head_grid_n4000` è M8b com'è, candidato (b): deve ridare le righe M8b di `results.csv`.
  - (g), M8a, si legge da `results.csv`. **(f), le soglie per exit riapprese, è rimandato al pilota B4**, che ne è il
    metodo: lo dico qui perché nel piano è tra i candidati di B2.
- **Contabilità**, in forward complete della ResNet20 senza teste.
  - Una riga di adattamento pagata costa c_9 − (1 − S_M2): l'input sarebbe uscito prima e invece arriva in fondo
    (conto incrementale). Accanto, il conto pieno (c_9 per riga).
  - Training: valutazioni di L-BFGS per 2·N·D·C MAC (testa) o 2·N·C² (mappa affine), come in
    `scripts/6_block3_review/adaptation_cost.py`.
  - Payback = (righe + training) / guadagno. Savings netti su 10.000, 100.000 e 1 milione di input dello stesso
    dominio, con le 3.000 righe di calibrazione pagate da tutti. Per `free` e `bn_traffic` i primi 4.000 input
    girano ancora con M2: il ritardo è contato.
  - **Lettura a pari audit**: per le configurazioni con al più 2.000 righe pagate, λ calibrata anche con 3.000 meno
    quelle righe, così il totale degli audit è lo stesso di M2. È una lettura contabile in più sulle stesse teste,
    non un fattore nuovo. La aggiungo io: non era tra le due varianti dette a Davide, glielo segnalo.
- **Cautela sulla garanzia.** Con `free` le teste dipendono dalle righe di cal attraverso la sola λ di M2, che
  sceglie quali righe arrivano in fondo. È una dipendenza debole ma c'è: confronto il tasso sopra ε con quello di M2
  sugli stessi casi e lo segnalo come punto da chiudere se la variante vince.
- **Ipotesi del piano, invariate.** D1: una variante economica tiene almeno il 70% del guadagno di M8b con al più il
  10% del suo compute e porta il payback a severity 3 sotto 30.000 input su entrambi i dataset. D2: (c) supera (b)
  di almeno 0,02. D3: con M8b l'accordo con l'exit 9 sale molto più dell'accuracy, e meno di metà del guadagno
  sopravvive se si certifica l'accuracy persa. Kill: nessuna variante tiene metà del guadagno con al più il 20% del
  costo; (c) non supera (b) di 0,02.
- **Previsioni.**
  1. **Iterazioni.** Con le 3.000 righe di M8b, `head_k5`, `head_k10` e `head_k20` tengono il 40–60%, il 60–80% e
     l'80–95% del guadagno a severity 3, con meno del 6% del training di M8b. Il kill sul costo non scatta.
  2. **Righe.** `head_grid` con 2.000, 1.000 e 500 righe tiene l'80–90%, il 60–75% e il 40–60% del guadagno su
     CIFAR10; il 65–80%, il 40–60% e il 20–40% su CIFAR100.
  3. **Mappa affine.** `logit_grid_n4000` tiene il 50–70% del guadagno di M8b su CIFAR10 e il 60–85% su CIFAR100. Con
     500 righe su CIFAR10 la mappa affine guadagna almeno quanto la testa.
  4. **Righe gratis.** `free` dà il 70–110% del guadagno di `match` (stima puntuale 85% su CIFAR10, 95% su CIFAR100):
     le righe che arrivano in fondo sono distorte verso gli input difficili, ma sono quelle che devono imparare a
     uscire prima. Previsione incerta nei due sensi.
  5. **BN.** `bn_n4000` alza l'accuracy dell'exit 9 di 6–9 punti a severity 3 e di 13–18 a severity 5 su CIFAR10, di
     8–10 e 12–15 su CIFAR100; i savings certificati salgono solo di 0,00–0,03 a severity 3 e 0,02–0,05 a severity 5
     su CIFAR10, e stanno tra −0,01 e +0,02 su CIFAR100. `bn_n500` dà almeno il 90% del guadagno di accuracy.
     `bn_traffic` perde 1–5 punti di accuracy dell'exit 9 su CIFAR10 e 0–3 su CIFAR100 rispetto a `bn_n4000`, con
     savings entro ±0,01.
  6. **D2 falsa.** (c) supera (b) di meno di 0,02 a severity 3 su entrambi i dataset, e di 0,00–0,02 a severity 5 su
     CIFAR10. Ma sulla frontiera (c) sta 5–15 punti di accuracy sopra (b) a pari compute.
  7. **D3 vera a metà.** L'accordo delle teste sale di 2–4 volte l'accuracy su CIFAR100, di 1,5–2,5 volte su CIFAR10 a
     severity 3. Tra i nuovi accordi, il riferimento sbaglia nel 35–50% dei casi su CIFAR10 a severity 3 e nel
     60–75% su CIFAR100 (70–85% sotto rumore). Ma l'accuracy persa si misura contro lo stesso riferimento, quindi
     premia anche lei la copia: mi aspetto che su CIFAR10 più di metà del guadagno sopravviva, e che su CIFAR100 a
     0,05 ne sopravviva meno di metà solo perché lì tutto esce già all'exit 1. La misura più pulita è la frontiera:
     con M8b sale di 1–3 punti di accuracy a compute medio su CIFAR10 a severity 3, di al più 1 su CIFAR100 sotto
     rumore.
  8. **D1 com'è scritta non passa**: su CIFAR100 con 3.000 righe il payback resta sopra 30.000 anche con tutto il
     guadagno (aritmetica della voce X.08 e di `adaptation_cost_payback.csv`). **Con le varianti** passa su CIFAR10
     con margine. Su CIFAR100 la migliore configurazione porta il payback sotto 30.000 solo con la mappa affine o
     con le righe gratis, e tiene il 50–70% del guadagno: il vincolo del 70% è quello che decide. Do il 50% che
     passi su entrambi.
  9. **Savings netti.** Su 10.000 input nessuna configurazione con righe pagate in più batte M2; ci riescono solo
     `free` e la lettura a pari audit, di 0,005–0,02 su CIFAR10 e di al più 0,01 su CIFAR100. Su 1 milione di input
     l'ordine è quello del guadagno lordo, con M8b e (c) in testa.
- **Comando e cartella.** `outputs/fable_review/pilots/B2_adattamento/`: un job su MPS per la BN (un processo) e poi
  un job su CPU per le teste (3 processi), ciascuno stimato sotto un'ora; se la prova dell'esecutore proietta di più,
  taglio lo split 1 invece di sforare. Cache temporanea di circa 13 GB per le feature con BN adattata, da cancellare
  dopo la verifica.
- **Esecuzione del codice** (22:20–22:42 del 14 novembre). Esecutore Opus 5.5 a effort xhigh, un agente, 79 chiamate,
  circa 279.000 token, 24 minuti. Prova su frost (CIFAR10 e CIFAR100 joint): 7 test passati; `m2` uguale alle righe M2
  di `results.csv` e `head_grid_n4000` uguale alle righe M8b, L2 scelti compresi (differenza 0); valutazioni di L-BFGS
  uguali a `adaptation_cost.csv` (3.356 e 3.872). Ho riletto per intero `b2_common.py`, `b2_bn.py`, `b2_heads.py` e
  `b2_summary.py`: seguono la specifica.
- **Tre rilievi dell'esecutore sul mio disegno, tutti fondati**, e cosa ne ho fatto prima di guardare un risultato.
  1. *`bn_traffic` non è gratis.* Le statistiche di un blocco vanno stimate su input normalizzati con le statistiche
     nuove dei blocchi precedenti, che nel traffico reale non sono mai stati calcolati: gli input vanno salvati e
     fatti ripassare fino al punto in cui si erano fermati. E dopo un cambio della BN cambia tutto il backbone,
     quindi le 3.000 righe di calibrazione di M2 non si possono riusare: serve una seconda calibrazione (3.000 audit).
     Ho corretto la contabilità: `bn_traffic` paga il ripasso e la seconda calibrazione. Le `bn_n…` restano
     contate nello scenario "si adatta e poi si calibra una volta sola".
  2. *Le righe gratis delle teste costano un po'*: le teste nuove vanno rivalutate sulle feature salvate delle 3.000
     righe di calibrazione (3 forward complete su CIFAR10, 32 su CIFAR100). Aggiunto al conto.
  3. *La mappa affine sui logit non aggiunge costo a inferenza*, al contrario di quanto temeva l'esecutore: A·(W f +
     b) + a è un solo layer lineare con pesi A·W. Nessuna correzione.
  Conseguenza che non avevo previsto e che vale più del pilota: **un adattamento che lascia fermo il backbone (teste,
  mappe sui logit, soglie, score) può riusare le righe di audit già pagate per ricalibrarsi; uno che lo cambia (BN,
  qualunque fine tuning) deve ripagare la calibrazione.**
- **Tempi.** La prova proietta 13 minuti per il job BN su uno split e 55 per il job delle teste sullo split 0 (70–80
  con i due split). Come scritto sopra, taglio: split 0 per tutto, in due lanci (CIFAR10 e poi CIFAR100), e lo split
  1 solo dopo, se serve. Job BN: dalle 22:45:01 alle 22:57:02, exit 0, batteria al 79%, logit su MPS identici a
  quelli salvati (differenza 0 sulle 4 reti), cache di 12 GB. Job delle teste su CIFAR10 lanciato alle 03:01 del 15
  novembre (la sessione era rimasta ferma tra le 23 e le 3).
- **Job delle teste, split 0.** CIFAR10 dalle 03:01:40 alle 03:15:40 (825 s), CIFAR100 dalle 03:16:10 alle 04:14:11
  (58 minuti, appena sotto l'ora: la stima era 50), exit 0 entrambi, batteria dal 79% al 67%. 120 casi, 6.600 righe.
  md5 di `outputs/savings_under_shift/results.csv` invariato.
- **Verifiche** (VERIFICATO).
  - Dello script (`b2_summary.log`), sui 120 casi: `m2` uguale alle righe M2 di `results.csv` (differenza 0);
    `head_grid_n4000` uguale alle righe M8b, L2 scelti compresi (differenza 0 su 120 righe); valutazioni di L-BFGS
    uguali a `adaptation_cost.csv` sui 6 domini; BN rimisurata dalla cache uguale al job BN; `source` uguale a `m2`.
  - Mie, con codice separato (`verifica_fable/b2_check.py`, due log): su 6 casi estratti a caso ho ricostruito
    cinque configurazioni da logit, feature, pesi e immagini, con una loss scritta in altro modo, la UCB lenta di
    riferimento e la BN stimata da PyTorch invece che dalla forward scritta a mano. Scarto massimo 0,0006 di savings;
    righe scelte e conteggi identici.
  - Gli stessi guadagni ricalcolati da me senza lo script di sintesi (`verifica_fable/b2_extra.log`) coincidono.
  - Garanzia: sopra ε su eval l'8,3% dei casi per M2, tra il 5% e il 15% per le configurazioni principali (120 casi
    ciascuna, errore standard 0,026); le `free` insieme 7,3% su 960 casi, le BN 5,2% su 600. Nessun segno che la
    dipendenza delle righe gratis dalla λ di M2 gonfi le violazioni.
  - Due errori trovati e corretti da me nello script di sintesi dopo il job: i netti a pari audit delle
    configurazioni `match` confrontavano un sottoinsieme di casi con l'M2 di tutti; un test fissava la vecchia
    contabilità di `bn_traffic`.
- **Risultato** (split 0, 30 casi per dataset e severity; intervalli al 95% con bootstrap sulle 15 corruzioni; M2 a
  0,342 e 0,246 su CIFAR10 a severity 3 e 5, a 0,150 e 0,117 su CIFAR100).
  - **M8b com'è**: +0,053 [0,043, 0,063] e +0,094 [0,055, 0,150] su CIFAR10; +0,022 [0,015, 0,032] e +0,053 [0,034,
    0,075] su CIFAR100. Training 2.569 e 30.262 forward complete. Payback 74.300 e 1.387.000 input a severity 3.
  - **Meno iterazioni** (stesse 3.000 righe, un solo L2): 5, 10, 20 iterazioni tengono il 62%, 84%, 88% del
    guadagno su CIFAR10 a severity 3 e il 28%, 38%, 54% su CIFAR100, con l'1,5%, 2,8%, 5,5% del training.
  - **Mappa affine sui logit** con L2 scelto: tiene il 91% e l'86% su CIFAR10, il 114% e il 103% su CIFAR100, con
    l'1,5% del training di M8b su CIFAR10 (38 forward) e il 19% su CIFAR100 (5.679). Con un solo L2 e 20 iterazioni:
    88% e 86% su CIFAR10 con 2,6 forward, 58% e 67% su CIFAR100.
  - **Meno righe**, `head_grid` con 2.000 / 1.000 / 500 righe: 83% / 75% / 67% su CIFAR10 a severity 3, 69% / 57% /
    29% su CIFAR100. Mappa affine con L2 scelto: 88% / 74% / 72% e 93% / 46% / 22%. Su CIFAR100 con poche righe e
    L2 fisso la mappa affine fa danno (−0,016 con 375 righe e 20 iterazioni).
  - **Payback più corti con almeno il 70% del guadagno**, a severity 3: su CIFAR10 6.065 input (`logit_k20_n1000`,
    80%, 750 righe, 0,6 forward di training) e 4.663 (`logit_grid_n500`, 72%); su CIFAR100 152.000
    (`logit_grid_n2000`, 93%). Su CIFAR100 nessuna configurazione di B2 scende sotto 30.000 con più di un terzo del
    guadagno (con B2.02 il solo bias con 750 righe arriva a 15.600 con il 35%).
  - **Righe gratis.** Rendono meno delle stesse righe prese a caso in 22 confronti su 24 (sei varianti per quattro
    celle, intervallo fuori dallo zero). Con L2 scelto: per le teste l'82% e il 95% su CIFAR10, il 67% e l'80% su
    CIFAR100; per la mappa affine il 55% e il 78%, il 63% e l'80%. Con L2 fisso il rapporto va dal 10% (mappa
    affine, 20 iterazioni, CIFAR100 a severity 3) a più del 100% con 5 iterazioni, dove i due guadagni sono piccoli
    (precisazione aggiunta il 15 novembre ricontrollando i testi). Costano però quasi nulla: su CIFAR10
    `logit_k5_free` dà +0,023 e +0,043 con 0,3 forward (payback 14 e 10 input dopo i 4.000 di raccolta).
  - **Savings netti contro M2.** Su 10.000 input: su CIFAR10 +0,007 con 500 righe pagate in più (`logit_grid_n500`)
    e +0,027 a pari audit (`logit_k20_n1000`: 750 righe per adattare, 2.250 per calibrare); a severity 5 +0,035 e
    +0,052. Su CIFAR100 a severity 3 nulla batte M2 su 10.000 input, e su 100.000 il meglio tra le configurazioni di
    B2 è +0,005 (`logit_k20_n4000`; con B2.02 scala e bias arrivano a +0,009); a severity 5 +0,004 e +0,011 su
    10.000. Su 1 milione di input: M8b +0,049 e +0,090 su CIFAR10, ma −0,009 su CIFAR100 a severity 3 (non si
    ripaga nemmeno lì); la mappa affine con L2 scelto +0,046, +0,080, +0,019, +0,048. (Frase su CIFAR100 a 100.000
    input corretta il 15 novembre: prima diceva "nulla batte M2", ed era falso.)
  - **BN** con 4.000 righe: accuracy dell'exit 9 +7,3 e +16,2 punti su CIFAR10, +8,0 e +12,2 su CIFAR100; savings
    certificati −0,005 [−0,039, +0,031], +0,018, +0,003, −0,004. Per famiglia su CIFAR10 a severity 3: blur +0,046,
    digitale +0,016, meteo −0,028, rumore −0,068 (dove l'accuracy sale di 17 punti). Con 500 righe il 93–98% del
    guadagno di accuracy. Dal solo traffico: al più 0,7 punti in meno.
  - **BN e poi teste** contro teste da sole, nei savings certificati: −0,017 [−0,046, +0,013], −0,036, −0,012,
    −0,047 [−0,070, −0,026]; sotto rumore −0,077, −0,155, −0,041, −0,111. Ma l'accuracy del sistema è 0,723 contro
    0,645, 0,635 contro 0,467, 0,432 contro 0,352, 0,346 contro 0,220.
  - **Frontiera** (accuracy a pari compute di M2, rispetto a M2): M8b +0,8 punti e 0,0 su CIFAR10, −0,2 e −0,3 su
    CIFAR100; BN +7,7 e +16,3, +8,0 e +12,5; BN e poi teste +8,4 e +16,6, +7,8 e +12,3.
  - **Copia degli errori** (M8b, media sulle 8 teste, righe di eval): su CIFAR10 a severity 3 l'accuracy sale di
    2,8 punti e l'accordo di 3,4; a severity 5 di 1,3 e 8,2. Su CIFAR100 l'accuracy non sale (+0,0 e −0,6 punti) e
    l'accordo sale di 4,9 e 9,8. Tra i nuovi accordi il riferimento sbaglia nel 43% e 69% dei casi su CIFAR10, nel
    78% e 89% su CIFAR100; sotto rumore su CIFAR100 nel 92% e 97%.
  - **Savings certificati sull'accuracy persa** (soglia 0,05, label sulle righe di cal): con M8b +0,125 [0,087,
    0,164] e +0,095 su CIFAR10; +0,014 [−0,006, +0,036] e −0,020 [−0,044, +0,003] su CIFAR100. Con soglia 0,02:
    +0,123, +0,082, −0,008, −0,054 [−0,099, −0,012].
- **Ipotesi del piano.**
  - **D1 non passa**, né com'è scritta né con le varianti: su CIFAR100 nessuna configurazione rispetta le tre
    condizioni. Su CIFAR10 da solo le rispettano 16 configurazioni a severity 3 (7 già con le righe di M8b).
  - **Kill non scatta**: in ogni dataset e severity ci sono varianti con almeno metà del guadagno e al più il 20%
    del costo (su CIFAR100 a severity 3: `logit_grid_n2000` 93% con il 10%, `logit_k20_n4000` 58% con il 2%).
  - **D2 falsa, con il segno opposto.**
  - **D3 vera su CIFAR100 e sotto rumore, falsa su CIFAR10 a severity 3.**
- **Convinzioni** (previsioni contro risultati).
  1. Iterazioni: giusta su CIFAR10 (un po' meglio del previsto), sbagliata su CIFAR100 (28–54% contro 40–95%).
  2. Righe: giusta su CIFAR100; su CIFAR10 con 500 righe tiene il 67% e non il 40–60%.
  3. Mappa affine: **sbagliata per difetto**. Prevedevo 50–70% e 60–85%, sono 91% e 114%. Una correzione lineare
     dei logit di ogni testa spiega quasi tutto il guadagno di M8b. Giusta la parte sulle 500 righe su CIFAR10.
  4. Righe gratis: giusta per le teste su CIFAR10, sbagliata per la mappa affine (55–80%). L'idea che le righe
     difficili fossero quelle giuste da cui imparare è smentita: rendono meno.
  5. BN: giusta sull'accuracy; i savings sono un po' sotto il previsto (−0,005 contro 0,00–0,03). Il traffico
     distorce meno di quanto pensassi (0,7 punti contro 1–5).
  6. D2: giusta nel verdetto, ma non avevo previsto il segno negativo.
  7. D3: giusta l'idea che l'accuracy persa premi anche la copia su CIFAR10; sbagliati i numeri (su CIFAR100
     l'accuracy delle teste non sale affatto, e sotto rumore la copia è al 92–97%, non al 70–85%).
  8. D1: giusta com'è scritta; sbagliata con le varianti su CIFAR100 (nessuna sotto 30.000).
  9. Savings netti: sbagliata in meglio su CIFAR10 (500 righe pagate bastano a battere M2 su 10.000 input; a pari
     audit +0,027 e non 0,005–0,02), giusta su CIFAR100.
  - **Cosa cambia.** (i) Sui savings certificati l'adattamento giusto è il più piccolo: una mappa affine sui logit
    di ogni testa, non le teste intere. (ii) I due metri danno classifiche opposte: per i savings a consistency
    risk fissato vincono le teste e la BN è inutile; per l'accuracy a pari compute vince la BN e le teste sono
    inutili o dannose. (iii) Un riferimento degradato è più facile da imitare: per questo migliorare il riferimento
    con la BN abbassa i savings certificati sotto rumore, e per questo su CIFAR100 il guadagno delle teste è copia
    di errori. (iv) Su CIFAR100 le varianti che tengono almeno metà del guadagno si ripagano dopo almeno 33.700
    input (scala e bias, voce B2.02) e M8b dopo 1,4 milioni. (Corretto il 15 novembre: prima diceva che nessuna fase
    di adattamento si ripaga sotto i 100.000 input, ed era troppo forte.)
  - **Limiti.** Un solo seed, due regimi, uno split (il secondo è in corso per M8b e le varianti economiche). La
    scelta di L2 = 0,01 per le varianti a L2 fisso era scritta prima; su CIFAR100 con poche righe non regge.
- **Secondo split** (job dalle 04:17:51 alle 04:39:51 del 15 novembre, 1.220 secondi, 2 processi, exit 0, batteria
  al 68%; M8b e le 36 varianti con L2 fisso). M2 e M8b uguali a `results.csv` anche sullo split 1 (240 righe in
  tutto, differenza 0). Sui 148 confronti (37 configurazioni per 4 celle) il guadagno cambia tra i due split in media
  di 0,003 e al più di 0,011 (`logit_k20_n1000` su CIFAR10 a severity 3: 0,043 e 0,032); la correlazione di rango tra i
  guadagni dei due split va da 0,93 a 0,99. M8b: 0,053 e 0,046 su CIFAR10 a severity 3, 0,022 e 0,026 su CIFAR100.
  Le conclusioni non cambiano. La mappa affine con L2 fisso e 750 righe fa danno su CIFAR100 a severity 3 su tutti e
  due gli split (−0,003 e −0,007).
- **Cache.** La tengo fino alla decisione del Checkpoint 3a (12 GB, si rifà in 11 minuti): serve alla voce B2.03.
  Dal 15 novembre mattina l'orchestratore è Opus 5.5 (scelta di Davide): da qui lavoro direttamente, senza esecutori.

### B2.02 (15 nov) Cosa fa la mappa affine sui logit: calibrazione, bias di classe o confusioni?

Scritta alle 04:30 del 15 novembre, prima di scrivere il codice. È un seguito breve di B2 (candidato (d)), sugli
stessi 120 casi dello split 0, solo logit salvati, un processo, meno di 20 minuti. Lo aggiungo io: serve a dire cosa
adattare online, che è la raccomandazione chiesta al Checkpoint 3a.

- **Domanda.** La mappa affine tiene il 91% e il 114% del guadagno di M8b. Quanta parte viene da (i) una sola
  temperatura per testa, che cambia la confidence ma non le predizioni, (ii) un bias per classe, (iii) scala e bias
  per classe, (iv) il resto della matrice, cioè la correzione delle confusioni tra classi?
- **Configurazioni**, tutte fittate per distillazione verso l'exit 9 con L-BFGS, senza L2 (pochi parametri):
  `temp` (1 parametro per testa), `bias` (C), `diag` (2C), su 3.000 righe (`n4000`) e su 750 (`n1000`). Il termine di
  confronto è `logit_grid_n4000` di B2 (C² + C parametri). Poi M2 sulle righe di cal, come sempre.
- **Perché interessa anche B4.** `temp` più λ condivisa certificata è una ricalibrazione per exit: se desse quanto le
  8 soglie di B1 (+0,012 e +0,035 nel confronto equo su CIFAR10 joint e joint_sd a severity 3) sarebbe una versione
  certificata e quasi gratuita di quell'idea.
- **Previsioni.**
  1. `temp` da sola dà poco: tra 0,000 e +0,010 su CIFAR10 a severity 3 (al più il 20% del guadagno della mappa
     affine) e al più +0,005 su CIFAR100. La temperatura ottimizza la verosimiglianza, non il compromesso tra rischio
     e compute, e non cambia le predizioni.
  2. `bias` dà il 40–60% del guadagno della mappa affine su CIFAR10 e il 30–50% su CIFAR100: sotto corruzione le
     teste collassano su poche classi, e un bias per classe ne corregge una buona parte.
  3. `diag` dà il 55–75% su CIFAR10 e il 40–60% su CIFAR100.
  4. Il resto (mappa piena meno `diag`) vale almeno il 25% su CIFAR10 e almeno il 40% su CIFAR100.
  5. Con 750 righe `bias` e `diag` perdono meno del 15% del loro guadagno con 3.000.
  Sono incerto su tutte: non ho misure precedenti su questo.
- **Comando e cartella.** `python outputs/fable_review/pilots/B2_adattamento/b2_mech.py` (scritto da me, riusa
  `measure_config`), output in `runs_mech/`, lettura in `verifica_fable/b2_mech_read.py`.
- **Esecuzione.** Dalle 04:20 alle 04:24:35 del 15 novembre, 280 secondi, un processo, exit 0. 720 righe (120 casi
  per 6 configurazioni). Lettura in `verifica_fable/b2_mech_read.log`.
- **Risultato** (guadagno su M2, split 0, intervalli sulle corruzioni; tra parentesi la quota del guadagno della
  mappa affine piena con 3.000 righe).
  - CIFAR10 a severity 3 (mappa piena +0,048): `temp` +0,019 [0,013, 0,025] (39%), `bias` +0,031 (64%), `diag`
    +0,045 [0,039, 0,052] (94%). A severity 5 (mappa piena +0,082): 37%, 56%, 80%.
  - CIFAR100 a severity 3 (mappa piena +0,025): `temp` +0,002 (9%), `bias` +0,010 (39%), `diag` +0,015 [0,013,
    0,017] (57%). A severity 5 (+0,055): 6%, 39%, 54%.
  - Con 750 righe: su CIFAR10 `diag` dà +0,046 e +0,065 (come con 3.000), con 0,12 forward di training; su CIFAR100
    +0,007 e +0,022 (`bias` +0,008 e +0,020).
  - `temp` per regime su CIFAR10 a severity 3: +0,007 (joint) e +0,030 (joint_sd). Le 8 soglie per exit di B1, nel
    confronto equo, davano +0,012 e +0,035 sugli stessi regimi (tre seed).
  - Savings certificati sull'accuracy persa (soglia 0,05) su CIFAR100: con `bias` +0,028 a severity 3 e +0,002 a
    severity 5; con `diag` +0,002 e −0,020; con la mappa piena +0,005 e −0,024.
  - Payback a severity 3 con le righe pagate: `logit_diag_n1000` su CIFAR10 circa 5.600 input (85% del guadagno di
    M8b); su CIFAR100 `logit_bias_n1000` circa 15.600 (35%) e `logit_diag_n4000` circa 33.700 (65%).
- **Convinzioni.**
  - Previsione 1 sbagliata su CIFAR10: la sola temperatura dà il 39% e non al più il 20%. Giusta su CIFAR100.
  - Previsioni 2 e 3 sbagliate per difetto su CIFAR10 (64% e 94% contro 40–60% e 55–75%), giuste su CIFAR100.
  - Previsione 4 sbagliata su CIFAR10 (il resto della matrice vale il 6%), giusta su CIFAR100 (43%).
  - Previsione 5 giusta su CIFAR10, sbagliata su CIFAR100 per `diag` (perde metà).
  - **Cosa cambia.** Su CIFAR10 il guadagno dell'adattamento delle teste è quasi tutto una ricalibrazione per
    classe: scala e bias sui logit, 20 numeri per testa, tengono l'85% del guadagno di M8b con un training ventimila
    volte più piccolo e senza perdere con 750 righe. Su CIFAR100 metà del guadagno viene dalla correzione delle
    confusioni tra classi, che ha bisogno di più righe e copia di più gli errori del riferimento.
  - ~~La temperatura per testa con λ condivisa è una versione certificata e quasi gratuita delle soglie per exit su
    CIFAR10 (+0,019 contro +0,019 di B1 su tutti i regimi); su CIFAR100 non le riproduce (+0,002 contro +0,017).~~
    **Sbagliato** (voce B2.03): confrontavo joint e joint_sd del seed 0 con la media di B1 su tutti i regimi. Sulle
    24 reti la temperatura dà +0,004 su CIFAR10 e aiuta solo in joint_sd.
  - D1 resta non superata: su CIFAR100 la configurazione più vicina è `logit_diag_n4000` (65% del guadagno, payback
    33.700), sotto entrambe le soglie di poco.

### B2.03 (15 nov) Le mappe sui logit tengono su 24 reti? E dopo la BN?

Scritta verso le 10:31 del 15 novembre, prima di scrivere il codice (ora ricavata dai file: lo script è delle
10:32:17; avevo scritto "08:10", una stima sbagliata). Orchestratore Opus 5.5, che lavora direttamente.
È l'ultima aggiunta a B2 prima del Checkpoint 3a: la raccomandazione per la parte online poggia su `logit_diag`,
misurata su 4 reti e un seed (voce B2.02), e il piano dice che un effetto su un seed solo non basta (sezione 6.2).

- **Perché si può fare a costo quasi zero.** Le mappe sui logit hanno bisogno solo dei logit, salvati per tutte le
  24 reti; le teste no (feature salvate solo per il seed 0). M2 e M8b delle 24 reti sono già in
  `outputs/savings_under_shift_v2/results.csv` (split 0, 1, 2), misurati come `measure_config`: il confronto è
  appaiato.
- **Parte 1.** `temp`, `bias`, `diag` (voce B2.02) con le righe di fit di `n4000` (adapt[:3000], le righe su cui M8b
  allena le teste) e di `n1000` (adapt[:750]), sulle 24 reti, 15 corruzioni a severity 3 e 5, split 0, 1, 2: 2.160
  casi. Una riga `m2` per caso, da confrontare con le righe M2 di `results.csv` (attesa differenza 0).
- **Parte 2.** `bias` e `diag` fittati sui logit della rete con la BN stimata sulle 4.000 righe di adapt (cache di
  B2, seed 0, joint e joint_sd, split 0), con insegnante l'exit 9 adattato: `c_logit_bias_n4000`,
  `c_logit_diag_n4000`. Da confrontare con `bn_n4000` e `c_logit_grid_n4000` di B2. È il "secondo braccio" proposto
  per la parte online, che non avevo misurato nella sua forma economica.
- **Previsioni** (quote = rapporto tra i guadagni medi su M2, stessi casi).
  1. `logit_diag_n4000` tiene del guadagno di M8b: su CIFAR10 il 75–95% a severity 3 e il 60–80% a severity 5; su
     CIFAR100 il 50–75% e il 45–65%. (Seed 0, joint e joint_sd: 85%, 69%, 65%, 56%.)
  2. Seed: la deviazione standard tra seed del guadagno di `diag` (per setting, media sugli 8) è al più 0,005; la
     quota tenuta da ogni seed sta entro 10 punti da quella complessiva.
  3. Regimi: la quota tenuta da `diag` sta entro 15 punti dalla media del dataset in ogni regime. Il più incerto è
     disjoint, dove M8b guadagna poco (0,013 su CIFAR100 al seed 0).
  4. Con 750 righe `diag` tiene almeno il 90% del suo guadagno con 3.000 su CIFAR10 in ogni regime, tra il 40% e il
     75% su CIFAR100.
  5. `temp` tiene il 25–45% del guadagno di M8b su CIFAR10 e al più il 15% su CIFAR100; `bias` il 50–65% e il
     35–50%.
  6. Garanzia: la quota di casi sopra ε di `diag` sta entro 2 punti da quella di M2 sugli stessi casi.
  7. Parte 2: `c_logit_diag_n4000` guadagna su M2 +0,025–0,040 su CIFAR10 e +0,005–0,012 su CIFAR100 a severity 3
     (tra la BN da sola e `diag` da sola), con l'accuracy del sistema della BN (7–9 punti sopra M2).
  8. D1 con `diag` a severity 3 sulle 24 reti: passa su CIFAR10 (quota almeno 70%, training sotto il 10% di M8b,
     payback sotto 30.000), non passa su CIFAR100.
- **Comando e cartella.** `python outputs/fable_review/pilots/B2_adattamento/b2_mech24.py --workers 3` (scritto da
  me, riusa `b2_mech.py` e `measure_config`), output in `runs_mech24/`, lettura con codice separato in
  `verifica_fable/b2_mech24_read.py`. Solo CPU, 3 processi, stima 30 minuti.
- **Esecuzione.** Prova su due reti e una corruzione alle 10:32 (righe identiche a `results.csv` e a B2.02). Job dalle
  10:33:09 alle 10:59:09 (1.507 secondi), exit 0, batteria all'80%. 15.360 righe, 360 coppie (rete, corruzione).
  md5 di `outputs/savings_under_shift/results.csv` invariato.
- **Verifiche** (VERIFICATO). Le 2.160 righe `m2` coincidono con le righe M2 di `results.csv` (differenza 0 su λ,
  rischio e savings). Lettura e controlli con codice separato: `verifica_fable/b2_mech24_read.log` e
  `b2_mech24_extra.log`.
- **Risultato, parte 1** (24 reti, split 0–2, 540 casi per cella; guadagno su M2 e quota del guadagno di M8b, con
  intervallo al 95% a due vie su reti e corruzioni; a severity 3 e 5).
  - M8b: +0,045 e +0,078 su CIFAR10, +0,019 e +0,043 su CIFAR100.
  - `logit_diag_n4000`: +0,034 e +0,046 su CIFAR10, cioè il 77% [64%, 88%] e il 59% [48%, 72%]; +0,014 e +0,022
    su CIFAR100, il 76% [53%, 111%] e il 51% [40%, 64%]. Positivo nel 92–97% dei casi.
  - `logit_diag_n1000` (750 righe): 70% e 56% su CIFAR10, 29% e 37% su CIFAR100.
  - `logit_bias_n4000`: 63%, 50%, 43%, 38%. `logit_temp_n4000`: 9%, 13%, 14%, 1%.
  - Per seed, quota di `diag` a severity 3: 81%, 75%, 74% su CIFAR10; 72%, 78%, 77% su CIFAR100. Deviazione standard
    tra seed del guadagno di `diag` (per setting, media sui regimi): 0,004, 0,009, 0,002, 0,002.
  - Per regime, guadagno di `diag` a severity 3: su CIFAR10 da 0,028 (mixed) a 0,048 (joint_sd); su CIFAR100 da
    0,012 (mixed) a 0,018 (joint_sd). In disjoint su CIFAR100 `diag` guadagna più di M8b (0,014 contro 0,009).
  - La temperatura aiuta solo in joint_sd (+0,020 su CIFAR10 a severity 3); negli altri regimi da −0,005 a +0,002.
    Le 8 soglie per exit di B1 nel confronto equo davano +0,019 e +0,017 a severity 3; la temperatura +0,004 e +0,003,
    scala e bias +0,034 e +0,014 (baseline diverse: la UCB per le mappe, il rischio empirico su cal per B1).
  - Garanzia: sopra ε `diag` 11,5%, 11,5%, 6,7%, 5,6% contro M2 13,3%, 8,5%, 11,1%, 8,5% sugli stessi casi;
    differenze con M2 da −0,044 a +0,030, tutti gli intervalli contengono lo zero. Sopra ε + 2 errori standard al più
    l'1,1% (M2 dallo 0,6% all'1,9%). Rischio medio mai sopra quello di M2.
  - Accuracy del sistema al costo di M2: `diag` +0,75 e +0,15 punti su CIFAR10, −0,14 e −0,21 su CIFAR100.
  - Payback a severity 3 (training di M8b da `adaptation_cost.csv`): M8b circa 86.000 e 1.658.000 input; `diag` con
    3.000 righe 27.300 e 28.900; con 750 righe 7.500 (CIFAR10, 70%) e 18.700 (CIFAR100, 29%).
  - A pari audit, `diag` con 750 righe per adattare e 2.250 per calibrare: +0,028 e +0,041 su CIFAR10, +0,004 e
    +0,014 su CIFAR100; su 10.000 input netti +0,020 e +0,029, +0,003 e +0,010.
- **Risultato, parte 2** (seed 0, joint e joint_sd, split 0; guadagno su M2 e accuracy del sistema rispetto a M2).
  - BN e poi scala e bias: +0,038 e +0,059 su CIFAR10 con +8,4 e +17,0 punti; +0,012 e +0,004 su CIFAR100 con +8,0
    e +12,5 punti. Quasi uguale a BN e poi mappa affine piena (+0,039, +0,060, +0,010, +0,007).
  - Rispetto a scala e bias da soli tiene l'85% e il 90% del guadagno su CIFAR10, l'81% e il 14% su CIFAR100.
- **Previsioni.**
  1. Giusta nella sostanza: 77% e 51% dentro gli intervalli, 59% e 76% a un punto dal bordo.
  2. Giusta sulle quote per seed (tutte entro 10 punti); sbagliata sulla deviazione standard su CIFAR10 a
     severity 5 (0,009 contro al più 0,005).
  3. Sbagliata in 4 celle su 16: disjoint e joint_sd su CIFAR10 a severity 5 (38% e 76% contro una media del 59%),
     disjoint e joint su CIFAR100 a severity 3 (in disjoint M8b guadagna così poco che la quota non ha senso).
  4. Quasi giusta: su CIFAR10 con 750 righe `diag` tiene dall'87% al 97% del suo guadagno con 3.000 (due celle sotto
     il 90%); su CIFAR100 dal 31% al 79% (due celle fuori).
  5. **Sbagliata sulla temperatura su CIFAR10** (9% e 13% contro 25–45%), giusta per il resto.
  6. Formalmente sbagliata in 3 celle su 4, ma in due `diag` sta sotto M2; l'unica sopra (+0,030 su CIFAR10 a
     severity 5) è dentro il rumore. Nessun allarme.
  7. Giusta.
  8. **Sbagliata su CIFAR100**: prevedevo che D1 non passasse, e con `diag` passa al limite.
- **Convinzioni.**
  - La raccomandazione regge su 24 reti e 3 seed: scala e bias per classe, con 3.000 righe, tengono tre quarti del
    guadagno di M8b a severity 3 su entrambi i dataset, con lo 0,02–0,03% del suo training e un payback di circa
    28.000 input invece di 86.000 e 1,7 milioni.
  - Con 750 righe regge su CIFAR10 (70%), non su CIFAR100 (29%): lì servono le 3.000 righe.
  - **D1.** Con `diag` le tre condizioni sono rispettate su entrambi i dataset, ma al limite (payback appena sotto
    30.000, intervalli delle quote che scendono al 64% e al 53%). `diag` l'ho scelta dopo aver visto B2: questa è una
    replica su 20 reti nuove di una scelta esplorativa, sempre sulle 15 corruzioni standard. Non la chiamerei una
    conferma. D1, scritta per le 4 reti di B2, lì resta falsa.
  - **Errore mio in B2.02.** Avevo scritto che la temperatura per testa è una versione certificata e quasi gratuita
    delle soglie per exit su CIFAR10. Confrontavo joint e joint_sd del seed 0 con la media di B1 su tutti i regimi:
    lo stesso tipo di confronto non equo della voce B1.01. Sulle 24 reti è falso. Corretto nella voce B2.02.
  - Scala e bias battono le 8 soglie per exit su CIFAR10 (0,034 contro 0,019) e le eguagliano su CIFAR100: per il
    pilota B4 è un motivo in più per ridurlo.
  - BN e poi scala e bias è un buon secondo braccio su CIFAR10. Su CIFAR100 a severity 5 la BN si mangia quasi tutto
    il guadagno di savings, come già con le teste.
- **Cache.** Usata qui per l'ultima volta; resta da decidere se cancellarla (12 GB).

---

## Dopo il Checkpoint 3a

Il 15 novembre alle 11:50 circa Davide ha risposto "Procedi con 5.3 e poi fase C". Leggo "5.3" come B3, la prima delle
due opzioni che gli avevo proposto ("B3, poi Fase C"); glielo scrivo, perché mi fermi se intendeva altro. Le altre
decisioni del checkpoint (testi per il tracciamento, cache, commit, documento online) restano aperte: non scrivo nulla
nei file della repository né nel documento online, e la cache resta dov'è. Orchestratore Opus 5.5, che lavora
direttamente.

### B3.01 (15 nov) Il meccanismo: traffico contro affidabilità, e uno score che usa tutti gli exit

Scritta alle 12:00 del 15 novembre (ora letta da `date`: 11:58), prima di scrivere il codice.

- **Domanda.** Sotto shift i savings certificati calano perché meno input superano la soglia (traffico) o perché a
  pari score gli input concordano meno con l'exit 9 (affidabilità)? E uno score che usa le statistiche di tutti gli
  exit già calcolati riconosce meglio gli input che concordano?
- **Cosa so già.** Nella scomposizione della Parte 1 ("shift a λ ferma" contro "cambio di λ") lo shift a λ ferma
  vale il 61–71% del calo su CIFAR10 e l'85–88% su CIFAR100 (`outputs/block3_review/decomposition_orders.log`). M1, a
  λ pulita sui dati shiftati, ha un rischio di 0,065–0,127 su CIFAR10 a severity 3: a pari soglia gli input usciti
  concordano meno. M8a (score dell'accordo rifittato sulle righe di adapt, solo statistiche dell'exit corrente) perde
  contro M2 perché sposta il traffico dagli exit 1–4 agli exit 5–8 (voce X.07). B1: circa il 90% del margine sta nel
  riconoscere gli input che concordano. B2.03: scala e bias per classe sui logit recuperano +0,034 su CIFAR10 e +0,014
  su CIFAR100 a severity 3.
- **Parte 1, la scomposizione (C1).** Per ogni exit l e dominio D un modello g_l^D = P(disaccordo con l'exit 9 |
  storia degli score) fino a l: regressione logistica su z(s_l) e z(m_l), i loro quadrati e il prodotto, con s_l la
  max softmax dell'exit l, m_l il massimo degli score degli exit precedenti (che decide se l'input arriva a l) e
  z(s) = −log(1 − s). Fit incrociato su due metà delle 10.000 righe; il modello di un dominio applicato all'altro non
  vede mai le sue righe. Il rischio "con gli score di A e l'affidabilità di B" alla soglia λ è la media sulle righe
  di A di g^B all'exit usato. Per ogni coppia (pulito P, dominio shiftato Q) la soglia condivisa con più savings e
  rischio ≤ ε, e i savings che ne seguono: S_PP, S_QQ, S_QP (score di Q, affidabilità di P), S_PQ. Calo = S_PP − S_QQ.
  Ordine 1: traffico S_PP − S_QP, affidabilità S_QP − S_QQ. Ordine 2: affidabilità S_PP − S_PQ, traffico S_PQ −
  S_QQ. Controllo: il rischio del modello sullo stesso dominio contro quello empirico lungo la griglia di λ. 24 reti,
  pulito e 15 corruzioni a severity 1, 3, 5.
- **Parte 2, gli score appresi (C2).** Per ogni exit l, una regressione logistica (standardizzazione, C = 1)
  che prevede l'accordo con l'exit 9:
  - `cur`: le tre statistiche dell'exit l (log della max softmax, margine tra le prime due probabilità, entropia),
    come M8a;
  - `hist`: le stesse statistiche di tutti gli exit da 1 a l, più gli indicatori "l'exit l concorda con l'exit j"
    per j < l.
  Allenata sulle 4.000 righe di adapt del dominio shiftato (`_adapt`, costa 4.000 audit come M8b) oppure sulle 5.000
  righe di val pulite (`_clean`, costo zero). Con la λ condivisa sullo score, oppure con la regola consapevole del
  costo (`_cost`): score'_l = 1 − (1 − p_l)·(c_9 − c_1)/(c_9 − c_l), cioè si esce all'exit l se P(disaccordo) <
  μ·(c_9 − c_l), certificata con la UCB lungo la stessa griglia. Il moltiplicatore va da 1 all'exit 1 a 7,5 all'exit 8.
  È la versione E1 del pilota B4 applicata a questi score: la includo perché senza di essa C2 ripeterebbe il
  difetto noto di M8a. Anche `ms_cost`: la max softmax con la stessa trasformazione, senza alcun training. Poi M2
  sulle righe di cal, misura su eval. 24 reti, severity 3 e 5, split 0. AUROC dell'accordo per exit, su eval.
- **Ipotesi del piano.** C1: l'affidabilità spiega almeno metà del calo a severity 3 nei regimi congiunti su
  CIFAR10 (joint e joint_sd); il piano prevedeva 50–65%. C2: il classificatore con più exit recupera almeno 0,03 di
  savings sulla max softmax sotto shift. Kill: i due ordini danno risposte opposte e nessuna misura distingue le due
  cause.
- **Previsioni mie.**
  1. C1 falsa: l'affidabilità spiega il 35–55% del calo a severity 3 su CIFAR10 nei regimi congiunti (più bassa del
     piano: nella scomposizione della Parte 1 il cambio di λ valeva il 29–39%), e il 15–35% su CIFAR100. I due ordini
     differiscono di al più 15 punti. Sotto rumore su CIFAR100 la quota di affidabilità è la più bassa (sotto il 15%,
     forse negativa: a pari score gli exit concordano di più su un riferimento che sbaglia).
  2. Controllo del modello: scarto massimo tra rischio del modello ed empirico lungo la griglia sotto 0,01, savings
     alla soglia scelta entro 0,01 da quelli empirici.
  3. AUROC della max softmax per l'accordo, media sugli exit 1–8: 0,85–0,90 sul pulito, 0,80–0,87 a severity 3.
     `hist_adapt` sopra la max softmax di 0,01–0,03.
  4. Con la λ condivisa `cur_adapt` e `hist_adapt` perdono contro M2 da 0,00 a 0,03 su CIFAR10 (`cur_adapt` deve
     riprodurre più o meno M8a).
  5. Con la regola consapevole del costo `hist_adapt_cost` guadagna su M2 tra 0,000 e 0,020 su CIFAR10 e tra −0,005
     e +0,010 su CIFAR100 a severity 3. C2 (almeno 0,03) falsa. Meno di scala e bias (+0,034).
  6. `ms_cost` tra −0,010 e +0,010; `hist_clean_cost` tra −0,010 e +0,010.
  7. La garanzia regge per tutti: la quota di casi sopra ε resta entro il rumore di M2.
- **Comando e cartella.** `outputs/fable_review/pilots/B3_meccanismo/` (`b3_decomp.py`, `b3_scores.py`, scritti da
  me), solo logit, CPU, 3 processi, stima 20 minuti in tutto.
- **Esecuzione.** Prove alle 12:00 su una e due reti. Job dalle 12:02:02 alle 12:06:20 (scomposizione 70 secondi,
  score 3 minuti), exit 0, batteria all'80%. md5 di `outputs/savings_under_shift/results.csv` invariato. Lettura con
  `verifica_fable/b3_read.py` (log `b3_read_part1.log` e `b3_read_part2.log`).
- **Verifiche** (VERIFICATO). Le 720 righe `m2` coincidono con M2 di `results.csv` (differenza 0). `cur_adapt` riproduce
  M8a: −0,015 contro −0,015 su CIFAR10 a severity 3, −0,009 contro −0,010 su CIFAR100.
- **Risultato, parte 1** (24 reti; quote come rapporto di medie; intervalli a due vie su reti e corruzioni).
  - Controllo del modello: scarto tra rischio del modello ed empirico lungo la griglia in media 0,012 (massimo 0,027;
    sopra 0,01 nella metà dei domini); i savings dell'oracolo del modello stanno 0,011 (pulito) e 0,008 (shiftato)
    sopra quelli empirici. Il modello è un po' ottimista, ma allo stesso modo sui due domini.
  - **C1, CIFAR10, joint e joint_sd, severity 3**: l'affidabilità spiega il 37% [25%, 48%] del calo nell'ordine 1 e il
    39% [30%, 47%] nell'ordine 2. Per seed: 44%, 35%, 32% e 43%, 39%, 34%.
  - Per dataset e severity (ordine 1 e 2): CIFAR10 41% e 38% a severity 1, 41% e 44% a 3, 49% e 58% a 5; CIFAR100
    27% e 37%, 31% e 47%, 38% e 67%. La quota dell'affidabilità cresce con la severity.
  - Per regime a severity 3 su CIFAR10: disjoint 50% e 56%, joint 33% e 37%, joint_sd 41% e 41%, mixed 41% e 43%.
  - Per famiglia a severity 3: il rumore ha la quota più alta (66% e 58% su CIFAR10, 51% e 71% su CIFAR100); il
    digitale la più bassa (24% e 33%, 12% e 23%).
  - **Alla λ pulita**, sui dati shiftati, il rischio è 0,101 su CIFAR10 a severity 3 (empirico; 0,097 dal modello).
    Con gli score shiftati ma l'affidabilità pulita sarebbe 0,047, cioè a ε. A λ ferma il traffico non alza il
    rischio: abbassa i savings (da 0,547 a 0,413). Il rischio sale per l'affidabilità.
  - AUROC della max softmax per l'accordo, media sugli exit 1–8: 0,874, 0,838, 0,798, 0,750 su CIFAR10 (pulito,
    severity 1, 3, 5); 0,819, 0,790, 0,758, 0,729 su CIFAR100.
- **Risultato, parte 2** (24 reti, split 0, 180 casi per cella; guadagno su M2 e intervallo a due vie; a severity
  3 e 5).
  - Con la λ condivisa gli score appresi perdono o pareggiano: `cur_adapt` −0,015 e −0,010 su CIFAR10, `hist_adapt`
    −0,010 e −0,002; su CIFAR100 −0,009/−0,010 e −0,004/−0,004.
  - **Con la regola consapevole del costo guadagnano**: `hist_adapt_cost` +0,019 [0,011, 0,026] e +0,029 [0,018,
    0,040] su CIFAR10, +0,016 [0,012, 0,020] e +0,016 [0,012, 0,021] su CIFAR100; `cur_adapt_cost` +0,016, +0,026,
    +0,014, +0,014. Positivi nell'83–94% dei casi.
  - **Senza costo di adattamento**, allenati solo sui dati puliti di val: `cur_clean_cost` +0,009 [0,002, 0,018],
    +0,010, +0,012 [0,008, 0,015], +0,009; `hist_clean_cost` +0,008 [−0,001, +0,018], +0,010, +0,014 [0,010,
    0,018], +0,012.
  - La max softmax con la stessa trasformazione (`ms_cost`): −0,008 e −0,008 su CIFAR10, +0,007 e +0,004 su CIFAR100.
  - Scala e bias sui logit (B2.03, stessi casi): +0,034, +0,045, +0,015, +0,022.
  - Le statistiche degli exit precedenti alzano poco l'AUROC: media sugli exit da 0,800 a 0,805 su CIFAR10 e da 0,755
    a 0,761 su CIFAR100 a severity 3; di 0,024 e 0,014 a severity 5.
  - Payback a severity 3 di `hist_adapt_cost` (4.000 audit, training trascurabile): 68.900 input su CIFAR10, 33.100 su
    CIFAR100. Scala e bias: 27.300 e 28.900.
  - Garanzia: quota di casi sopra ε tra il 6% e il 15%, contro il 9–13% di M2 sugli stessi casi.
- **Previsioni.**
  1. Giusta su CIFAR10 (37% e 39%, dentro 35–55%); su CIFAR100 giusta nell'ordine 1 (31%) e sbagliata nell'ordine 2
     (47% contro 15–35%); i due ordini differiscono di 2 punti su CIFAR10 e di 15 su CIFAR100. **Sbagliata sul
     rumore**: è la famiglia dove l'affidabilità pesa di più, non di meno.
  2. Sbagliata: lo scarto del modello è in media 0,012, non sotto 0,01, e i savings dell'oracolo del modello stanno
     0,008–0,011 sopra quelli empirici.
  3. Sbagliata in parte: AUROC sul pulito 0,874 e 0,819 (la seconda sotto 0,85), a severity 3 0,798 e 0,758 (sotto
     0,80); il guadagno di `hist` è 0,005–0,006 a severity 3, non 0,01–0,03.
  4. Giusta.
  5. Giusta su CIFAR10 (+0,019), sbagliata su CIFAR100 (+0,016 contro al più +0,010) e sul confronto con scala e bias
     su CIFAR100 (lo eguaglia invece di restare sotto).
  6. Giusta per `ms_cost`; sbagliata per `hist_clean_cost` su CIFAR100 (+0,014 e +0,012).
  7. Giusta.
- **Ipotesi del piano.** **C1 falsa**: l'affidabilità spiega il 37–39% del calo, non almeno metà; il piano prevedeva
  50–65%. **C2 falsa**: lo score con più exit recupera +0,019 su CIFAR10 e +0,016 su CIFAR100, non 0,03, e solo con
  la regola consapevole del costo. Il kill non scatta: i due ordini dicono la stessa cosa.
- **Convinzioni.**
  - Sotto shift i savings certificati calano per due ragioni di peso simile: gli exit perdono confidenza (traffico,
    circa 60%) e a pari confidenza concordano meno con l'exit 9 (affidabilità, circa 40%). Il secondo effetto è
    quello che fa salire il rischio a λ ferma e costringe a ricalibrare; il primo una ricalibrazione non lo recupera.
  - Gli score appresi non riconoscono molto meglio gli input che concordano (AUROC quasi uguale): il loro guadagno
    viene dalla regola di uscita. Una P(accordo) calibrata con una λ condivisa sposta il traffico verso gli exit
    tardivi (la spiegazione di M8a nel README); con la soglia proporzionale al costo residuo la stessa
    probabilità guadagna 0,01–0,03.
  - **Un risultato che non cercavo**: la regola consapevole del costo con uno score allenato solo sui dati puliti
    guadagna +0,009–0,014 sotto shift senza alcun costo. È la versione E1 del pilota B4, che avevo proposto di
    ridurre: va riconsiderata, e va misurata anche sul pulito (qui solo severity 3 e 5).
  - Per la Fase C non cambio la scelta: l'adattamento resta scala e bias sui logit, che a parità di casi guadagna di
    più (+0,034 contro +0,019 su CIFAR10). La regola consapevole del costo è una variante dello score, ortogonale
    all'adattamento: la tengo come estensione, da combinare dopo.

---

## Fase C del piano v3: la parte online

### C.01 (15 nov) Il simulatore, la scelta di γ e le previsioni

Scritta alle 12:20 del 15 novembre (ora letta da `date`: 12:17), dopo due prove del simulatore su una e due reti
(gaussian noise a severity 3), prima di ogni job della Fase C.

- **Domande** (documento di progetto). RQ2: senza label e online, quanti savings perde un controller della soglia
  alimentato dagli audit rispetto alla soglia migliore, e quanto ci mette a stabilizzarsi dopo lo shift? RQ3: quanto
  dei savings persi recupera una fase di adattamento, al netto degli audit in più e del training, e dopo quanti input
  si ripaga? H4 (da dove viene il costo), H5 (con i soli input arrivati in fondo si recupera meno).
- **Il simulatore** (`outputs/fable_review/fase_c/sim.py` e `run.py`, scritti da me). Per ogni dominio le 10.000
  immagini di test sono divise una volta: 5.000 di calibrazione (solo per i riferimenti offline) e 5.000 da cui gli
  stream pescano a caso, con ripetizione. Per ogni input: si esce al primo exit con max softmax sopra λ_t; se l'input
  è uscito prima, con probabilità π diventa un audit e la rete va fino in fondo; il controller di
  `src/eesavings/controller.py` aggiorna λ_t; i savings netti contano l'exit, gli audit e il training.
- **Metodi.** M1 soglia pulita fissa (UCB su 5.000 righe pulite), senza audit. M2 soglia di ogni dominio calibrata
  sulle sue 5.000 righe (oracolo), senza audit. M3 M1 più un monitor che passa alla rete completa quando un
  e-detector sugli audit scatta (soglia 10⁴, quattro puntate). M4 controller. M7 controller più una fase di
  adattamento che parte allo shift (trigger oracolo): audit al 25% finché ci sono 3.000 righe arrivate all'exit 9
  (audit e no), poi scala e bias per classe sui logit di ogni testa, poi il controller riparte dalla soglia stimata
  sulle righe della fase con pesi inversi alle probabilità di essere osservate. M5 e M6: come M7 con i soli input
  arrivati in fondo da soli (senza audit in più) o con i soli audit. M7f: tutti gli input completati durante la fase.
  M7c: niente fase, le mappe rifittate ogni 5.000 input sulle ultime 3.000 righe osservate. M7t: M7 con un trigger
  sulla soglia (media mobile su 1.000 input sopra la soglia iniziale più 0,03). M8: mappe oracolo (3.000 righe di
  calibrazione del dominio nuovo) e soglia UCB sulle altre 2.000, senza audit. M8c: le stesse mappe al momento dello
  shift, con il controller di M7.
- **Scelte fatte sulle prove, prima dei job.** (1) Senza la ripartenza della soglia M7 crolla dopo il cambio delle
  teste (savings da 0,25 a 0,14 per migliaia di input: la soglia tarata sulle teste vecchie è troppo alta per le
  nuove, e il controller scende solo di γε per input). La ripartenza è parte della fase di adattamento. Effetto sul
  bound: ogni ripartenza verso il basso aggiunge (λ prima − λ dopo)/(γT) al termine deterministico; lo riporto.
  (2) M1, M2 e M8 senza audit. (3) M8c aggiunto perché M8, calibrato con la UCB, è più prudente del controller.
- **Stream.** Salto: 10.000 input puliti, poi 30.000 del dominio shiftato. Ritorno al pulito: 10.000 puliti, 15.000
  shiftati, 15.000 puliti. ε = 0,05, π = 0,05.
- **Scelta di γ, regola fissata ora.** Stream solo puliti di 40.000 input, le 8 reti del seed 0, 10 ripetizioni, γ
  in {1e-4, 2e-4, 5e-4, 1e-3, 2e-3}. Scelgo il γ più grande per cui, in media sulle reti, il rischio medio sullo
  stream resta sotto ε + 0,005 e la finestra di 1.000 input peggiore resta sotto 0,075 (con una soglia fissa a rischio
  0,043 la finestra peggiore attesa è circa 0,06). Nessun dato shiftato entra nella scelta. Gli altri γ li riporto
  come sensibilità.
- **Job principale.** Le 24 reti, le 15 corruzioni a severity 3 e 5, stream con salto, 10 ripetizioni, metodi M1, M2,
  M4, M7, M8, M8c. Due job, CIFAR10 e CIFAR100, ciascuno sotto l'ora (stima dalle prove: 15 e 50 minuti). Poi le
  varianti (M3, M5, M6, M7f, M7c, M7t, ritorno al pulito, sensibilità a π) sulle 8 reti del seed 0.
- **Previsioni** (S_0 savings netti negli ultimi 5.000 input puliti; S_1 di M4 e S_2 di M7 negli ultimi 10.000 input
  shiftati; a severity 3 se non detto).
  1. **RQ2.** I savings netti stabilizzati di M4 stanno entro ±0,015 da quelli lordi di M2: il controller usa tutto
     il budget ε (+0,01–0,02 sulla UCB, come CRC in B1) e paga circa π volte i savings in audit.
  2. Rischio di M4 dopo lo shift: medio sui 30.000 input 0,055–0,070 su CIFAR10; negli ultimi 10.000 0,048–0,053;
     finestra peggiore di 1.000 input 0,12–0,20.
  3. Il transitorio (input dopo lo shift finché la finestra di 1.000 input scende sotto ε + 0,01) dura 2.000–6.000
     input su CIFAR10.
  4. M1 dopo lo shift: rischio 0,08–0,15 su CIFAR10 e savings più alti di M2 (illusori).
  5. M3 scatta entro 500–3.000 input su CIFAR10; su CIFAR100 più tardi o in alcune ripetizioni mai; dopo, savings
     circa zero.
  6. **RQ3.** S_2 − S_1: +0,025–0,040 su CIFAR10, +0,008–0,016 su CIFAR100.
  7. Frazione recuperata (S_2 − S_1)/(S_0 − S_1): 10–15% su CIFAR10, 7–13% su CIFAR100.
  8. Payback di M7 contro M4 (savings netti cumulati dallo shift): 15.000–35.000 input su CIFAR10; su CIFAR100
     25.000–60.000, spesso oltre la fine dello stream.
  9. M8c − M7 negli ultimi 10.000 input: 0,000–0,005.
  10. H5: M5 recupera il 50–90% di quello che recupera M7, senza rischio più alto.
  11. M6 recupera quanto M7 ma la sua fase dura molto di più (circa 20.000 input su CIFAR10).
  12. M7f ha la fase più breve e più cara; payback simile a M7 o più breve.
  13. M7c alla fine recupera quanto M7, senza costo di fase ma in ritardo.
  14. Garanzia: il rischio medio sui 40.000 input di ogni metodo con controller resta sotto ε + 0,01, e sotto il bound
      di Freedman (con il termine delle ripartenze) in almeno il 90% delle ripetizioni.
  15. L'accuracy del sistema cambia di meno di un punto con l'adattamento.
- **Cosa non faccio ora.** La BN adaptation online (cambia il riferimento: andrebbe trattata come ripartenza del
  certificato) e la regola consapevole del costo di B3: estensioni, dopo il job principale.
- **Scelta di γ** (12:18, 11 secondi di job; `fase_c/gamma_clean/choice.log`). Sugli stream puliti, media su 8 reti e
  10 ripetizioni:

  | γ | rischio medio | finestra peggiore di 1.000 | savings lordi | savings netti |
  |---|---|---|---|---|
  | M1 (soglia fissa, senza audit) | 0,0427 | 0,0616 | 0,382 | 0,382 |
  | 1e-4 | 0,0467 | 0,0689 | 0,391 | 0,371 |
  | 2e-4 | 0,0479 | 0,0732 | 0,393 | 0,373 |
  | 5e-4 | 0,0491 | 0,0818 | 0,394 | 0,374 |
  | 1e-3 | 0,0497 | 0,0911 | 0,394 | 0,374 |
  | 2e-3 | 0,0504 | 0,1060 | 0,391 | 0,371 |

  La regola sceglie **γ = 2e-4**. Sul pulito il controller ha savings lordi più alti di M1 (usa tutto il budget ε) e
  netti più bassi: gli audit costano circa 0,02, più di quanto il controller guadagni sulla UCB.

### C.02 (15 nov) Il job principale: M1, M2, M4, M7, M8, M8c su 24 reti

- **Esecuzione.** CIFAR10 dalle 12:18:49 alle 12:28:49, CIFAR100 dalle 12:29:01 alle 13:05:02 (`fase_c/STATUS.txt`),
  exit 0, batteria dall'80% al 76%. 720 coppie (rete, dominio), 10 ripetizioni, stream di 10.000 input puliti e
  30.000 shiftati, π = 0,05, γ = 2e-4. md5 di `outputs/savings_under_shift/results.csv` invariato. Lettura con
  `verifica_fable/c_read.py` (`fase_c/main_read.log`).
- **Verifiche del simulatore** (VERIFICATO, `verifica_fable/c_check.py`, log accanto), con codice separato: M1 e M2
  simulati contro savings e rischio calcolati direttamente sui logit dello stream (16 confronti, scarto massimo 2,1
  errori standard); il percorso della soglia identico a quello di `controller_step` su 10.000 passi (scarto 3e-8, la
  precisione del float32); le mappe identiche a quelle di B2.02; la ripartenza con pesi uniformi uguale all'oracolo
  empirico.
- **Risultato** (savings netti medi su reti, corruzioni e ripetizioni; CIFAR10 e poi CIFAR100; a severity 3 e 5).
  - Prima dello shift: M4 0,515 e 0,223; M1, senza audit, 0,535 e 0,227.
  - **M1 dopo lo shift**: rischio 0,087 e 0,130 su CIFAR10, 0,052 e 0,066 su CIFAR100; savings 0,393 e 0,350, 0,130
    e 0,113, più alti di M2 perché la garanzia è rotta.
  - **RQ2, M4 stabilizzato** (ultimi 10.000 input) contro M2: su CIFAR10 0,312 contro 0,317 (−0,005 [−0,009,
    −0,001]) e 0,222 contro 0,223 (−0,001 [−0,004, +0,002]); su CIFAR100 0,126 contro 0,122 (+0,003 [+0,001,
    +0,005]) e 0,096 contro 0,092 (+0,004 [+0,002, +0,006]). Nei savings lordi M4 sta sopra M2 di 0,010–0,012:
    usa tutto il budget ε (rischio 0,050 contro 0,044) e paga circa 0,007–0,017 di audit.
  - Transitorio di M4 (input dopo lo shift finché la finestra di 1.000 input scende sotto ε + 0,01, mediana): 6.450
    e 7.150 su CIFAR10, 1.000 su CIFAR100 (media 2.571 e 3.606). Rischio di M4 per finestre su CIFAR10 a severity 3:
    0,087, 0,072, 0,058, 0,052, 0,050; finestra peggiore 0,103 e 0,134 su CIFAR10, 0,084 e 0,099 su CIFAR100.
  - **RQ3, M7 contro M4** negli ultimi 10.000 input: su CIFAR10 +0,032 [0,023, 0,042] e +0,043 [0,025, 0,065];
    su CIFAR100 +0,015 [0,011, 0,019] e +0,022 [0,013, 0,033]. Frazione recuperata del calo: 16% [11%, 22%], 15%,
    15% [11%, 21%], 17%.
  - Durante la fase M7 paga: su CIFAR10 a severity 3 savings netti 0,296 e 0,277 nei primi 1.000 e nei 4.000 dopo,
    contro 0,373 e 0,353 di M4.
  - **M8c** (mappe oracolo, stesso controller) negli ultimi 10.000 input: 0,348, 0,268, 0,140, 0,117. M8c − M7:
    +0,003, +0,004, −0,001, +0,000. **M8** (mappe e soglia oracolo, senza audit): 0,339, 0,258, 0,132, 0,108.
  - **Payback di M7 contro M4** (pareggio del compute netto cumulato dallo shift, curve medie sulle ripetizioni),
    mediana su tutti i casi contando come infiniti quelli non ripagati entro lo stream: 27.200 e 20.700 input su
    CIFAR10 (ripagati entro i 30.000 input nel 56% e nel 63% dei casi), 12.550 e 11.100 su CIFAR100 (85% e 84%).
    Guadagno netto medio sui 30.000 input dopo lo shift: +0,004 e +0,019 su CIFAR10, +0,009 e +0,015 su CIFAR100.
  - Accuracy negli ultimi 10.000 input: +0,8 e +0,2 punti su CIFAR10, −0,1 e −0,2 su CIFAR100.
  - **Garanzia** (solo CIFAR100: su CIFAR10 il job era partito prima che aggiungessi la finestra sull'intero stream;
    lo rilancio): rischio medio sui 40.000 input 0,0495 per M4 (massimo 0,078), 0,0502 per M7 (massimo 0,065),
    0,0445 per M8c; sotto ε + 0,01 nel 93,8%, 98,5% e 100% delle ripetizioni. Il bound di Freedman con il termine
    delle ripartenze vale 0,22–0,25 e regge sempre: con γ = 2e-4 e 40.000 input il termine deterministico da solo
    vale 0,126.
- **Previsioni.**
  1. Giusta (scarti tra −0,005 e +0,004).
  2. Giusta sul rischio medio (circa 0,056 sui 30.000 input) e sugli ultimi 10.000 (0,050); sbagliata sulla finestra
     peggiore (0,103 contro 0,12–0,20).
  3. Di poco sbagliata: 6.450 input contro 2.000–6.000. È il prezzo del γ piccolo scelto sul pulito.
  4. Giusta.
  6. Giusta su entrambi i dataset.
  7. Sbagliata per difetto: 16% e 15% contro 10–15% e 7–13%.
  8. Giusta su CIFAR10 (27.200); **sbagliata su CIFAR100**: prevedevo 25.000–60.000, è 12.550. Online su CIFAR100
     la fase costa poco: due terzi degli input arrivano comunque all'exit 9, quindi le righe sono quasi tutte gratis,
     e gli audit in più riguardano solo il terzo che esce prima.
  9. Giusta.
  14. Giusta sul bound; sbagliata in parte su ε + 0,01: per M4 il 6% delle ripetizioni lo supera.
  15. Giusta.
- **Convinzioni.**
  - RQ2: online, senza label, il controller arriva ai savings della soglia oracolo a meno di 0,005, e su CIFAR100 la
    supera. Il prezzo è il transitorio: su CIFAR10 circa 6.500 input con il rischio sopra ε (fino a 0,087 nei primi
    1.000 a severity 3), contro 0,087 per sempre con M1.
  - RQ3: la fase di adattamento recupera il 15–17% del calo dei savings a regime, quanto l'adattamento oracolo (M8c).
    Su uno stream di 30.000 input dopo lo shift si ripaga nell'85% dei casi su CIFAR100 e nel 56–63% su CIFAR10.
    Netto su tutto lo stream il guadagno è piccolo (+0,004–0,019). **Superato dalla voce C.05** (ripartenza
    corretta): 12–16% a regime, ripagato nel 39–73% dei casi, netto sullo stream da −0,003 a +0,014.
  - ~~Online su CIFAR100 il payback è più corto che offline: in B2.03 scala e bias si ripagavano dopo circa 28.900
    input contando le 3.000 righe tutte come pagate; qui dopo 12.550, perché quasi due terzi delle righe arrivano
    all'exit 9 senza audit.~~ **Superato** (voce C.05): con la ripartenza corretta il payback su CIFAR100 è 27.100.
- **H4** (`verifica_fable/c_h4.py`, `fase_c/h4.log`): nei 30.000 input dopo lo shift gli audit in più di M7 costano
  558 e 414 forward complete su CIFAR10, 149 e 132 su CIFAR100; il training delle mappe 0,4 e 0,5, 10,9 e 11,8. Il
  training pesa lo 0,1% su CIFAR10 e il 7–8% su CIFAR100. **H4 è vera con le mappe sui logit** (con le teste di M8b
  era falsa, voce P1.14). Il guadagno lordo nello stesso periodo vale 691 e 979, 444 e 595 forward.

### C.03 (15 nov) Le varianti: M3, M5, M6, M7f, M7c, M7t sulle 8 reti del seed 0

- **Esecuzione.** CIFAR10 dalle 13:05:25 alle 13:14:25 (535 s), CIFAR100 dalle 13:14:35. 4 reti per dataset, 15
  corruzioni a severity 3, 6 ripetizioni. Lettura con `verifica_fable/c_variants_read.py`.
- **Risultato su CIFAR10** (`fase_c/variants_read_cifar10.log`; M4 come riferimento: S_0 0,511, S_1 0,302).
  - M7 (audit rinforzati): +0,036 [0,022, 0,052] negli ultimi 10.000 input, frazione recuperata 17%. Fase di 6.765
    input, 3.000 righe di cui 1.260 audit. Payback mediano 25.950 input (62% dei casi entro lo stream); netto sullo
    stream +0,007.
  - **M7f (fallback completo)**: +0,042 [0,031, 0,056], 117% di M7, frazione recuperata 20%: le righe sono un
    campione non distorto del dominio. Ma la fase (3.000 input con tutta la rete) costa: netto sullo stream −0,005,
    ripagato nel 40% dei casi.
  - **M7c (adattamento continuo)**: +0,020 [0,003, 0,040] a regime, 56% di M7; ma netto sullo stream +0,016, il
    migliore, e davanti a M4 quasi subito dopo lo shift (payback mediano 100 input, 83% dei casi): non paga una fase.
  - **M7t (trigger sulla soglia)**: scatta nel 63% delle ripetizioni, in media 7.900 input dopo lo shift; nessun
    falso allarme sul pulito. +0,015 a regime, netto sullo stream +0,003.
  - **M6 (solo audit)**: +0,033, 93% di M7, ma la fase dura 17.900 input; netto sullo stream −0,024.
  - **M5 (solo input arrivati in fondo)**: +0,015 [−0,002, +0,038], **43% di M7**; fase di 11.800 input; rischio come
    M4. Netto sullo stream +0,010.
  - **M3 (monitor e fallback alla rete completa)**: scatta nel 55% delle ripetizioni, mediana 4.373 input dopo lo
    shift (da 406 a 27.863); dopo, savings a zero. Netto sullo stream −0,097 contro M4, con rischio medio 0,037
    contro 0,057.
- **CIFAR100** (`fase_c/variants_read.log`; job dalle 13:14:35 alle 14:05:36, 51 minuti invece dei 35 stimati, exit
  0): M7 +0,017 [0,012, 0,022]; M7f +0,024; M7c +0,012; M5 +0,008 (49% di M7); M6 −0,020 (la sua fase dura 26.400
  input: negli ultimi 10.000 sta ancora pagando gli audit al 25%); M7t +0,002 (scatta nel 22% delle ripetizioni);
  M3 scatta nel 18% delle ripetizioni, netto −0,019.
- **Attenzione**: i risultati di M7, M7f, M7c, M7t, M5 e M6 di C.02 e C.03 vanno rifatti, per il difetto descritto
  nella voce C.04. M1, M2, M3, M4, M8 e M8c non ne sono toccati.

### C.04 (15 nov) Un difetto della ripartenza della soglia, e la correzione

Scritta alle 14:30 del 15 novembre (ora letta da `date`: 14:28), dopo le prove sotto e prima dei rilanci.

- **Il difetto**, trovato da me rileggendo il rischio delle varianti. La soglia ripartiva da una stima del rischio
  calcolata sulle stesse righe su cui le mappe erano state fittate. Le mappe concordano con l'exit 9 più su quelle
  righe che su righe nuove, quindi la stima era ottimista e la soglia ripartiva troppo bassa. Effetti: su CIFAR100 M7
  restava sopra ε (0,054) per circa 15.000 input dopo il cambio delle teste; M7c, che ripartiva dopo ogni rifit, stava
  sopra ε di continuo (0,056–0,060, anche sul pulito). Su CIFAR10 l'effetto era piccolo. Il bound di lungo periodo
  regge comunque (il termine delle ripartenze lo copre), ma i savings di questi metodi erano un po' gonfiati, e quelli
  di M7c su CIFAR100 molto. Ho fermato il job del ritorno al pulito, che usava la versione difettosa (cartella
  `fase_c/return_restart_in_sample_STOPPED`, 21 risultati parziali, non usati).
- **Prove** (2 reti per dataset, gaussian noise e frost a severity 3, 6 ripetizioni; non sono risultati):
  1. *Righe tenute da parte.* Un quarto delle righe osservate (una ogni quattro, in ordine di arrivo) resta fuori dal
     training e serve solo alla ripartenza; la fase finisce quando restano 3.000 righe di training, cioè dopo circa
     4.000 righe osservate, come le 3.000 + 1.000 di M8b in B2. M7 su CIFAR100: rischio dopo il cambio delle teste
     0,049 invece di 0,051–0,052; savings negli ultimi 10.000 input 0,138 invece di 0,141. M7c resta sopra ε (0,057
     sul pulito): ripartire a ogni rifit scegliendo i savings massimi sotto un vincolo stimato con rumore porta sopra
     ε (la maledizione del vincitore), e il controller non fa in tempo a correggere.
  2. *Più un margine di un errore standard.* Troppo prudente: M7c a rischio 0,035–0,040 e sotto M4; M7 a 0,043–0,047.
     Scartato, anche perché tarare il margine su questi dati vorrebbe dire scegliere un iperparametro su CIFAR-C.
  3. *M7c senza ripartenza.* Pessimo: dopo ogni rifit la soglia resta troppo alta per le teste nuove (rischio
     0,01–0,03, savings sotto M4 anche sul pulito). La ripartenza serve ogni volta che le mappe cambiano.
  4. *Test prima della ripartenza.* Si riparte solo se, con le teste nuove, il rischio stimato alla soglia attuale
     sulle righe tenute da parte dista da ε più di due errori standard. M7: sempre (dopo lo shift la soglia è
     lontana), rischio 0,049–0,052. M7c: riparte nel 26–34% dei rifit; su CIFAR10 rischio 0,050 negli ultimi 10.000
     input; su CIFAR100 ancora 0,052–0,055, perché in modo continuo gli audit pesano 1/π = 20 e la stima è rumorosa.
- **Regola finale**, uguale per tutti i metodi con adattamento: righe tenute da parte (un quarto), test a due errori
  standard, nessun margine. M7c resta con il suo limite su CIFAR100, che riporterò. Le scelte 1 e 4 sono di
  principio (stima non distorta, ripartire solo quando serve); la soglia di due errori standard non l'ho variata.
- **Rilanci.** `main_v2` (M4 e M7 sulle 24 reti; M4 deve ridare esattamente i numeri di `main`: controllo di
  riproducibilità; M1, M2, M8, M8c restano da `main`), `variants_v2` (M4, M5, M6, M7, M7f, M7c, M7t sulle 8 reti del
  seed 0), il ritorno al pulito e la sensibilità a π e γ. Ciascuno sotto l'ora.
- **Previsioni per i rilanci.** Le previsioni di C.01 restano quelle. In più: il guadagno di M7 scende di 0,001–0,003
  rispetto a C.02 e il payback su CIFAR100 si allunga di 1.000–4.000 input (la fase dura un terzo in più e i savings
  non sono più gonfiati); M7c su CIFAR100 perde il primato sul netto dello stream.
- **Un secondo problema, di velocità.** Il primo rilancio era tre volte più lento: a ogni passo ricostruivo la lista
  delle righe di training di ogni ripetizione. Fermato (14:31–14:36), sostituito con un contatore, rilanciato. Su una
  prova di 960 righe i risultati sono identici prima e dopo.

### C.05 (15 nov) Il job principale con la ripartenza corretta (M4 e M7 su 24 reti)

- **Esecuzione.** `fase_c/main_v2`: CIFAR10 dalle 14:37:30 alle 14:43:30, CIFAR100 dalle 14:43:42 alle 15:19:43, exit
  0, batteria dal 63% al 48%. M1, M2, M8 e M8c restano da `main` (la correzione non li tocca). Lettura:
  `verifica_fable/c_read.py main main_v2` (`fase_c/main_v2_read.log`). md5 di `results.csv` invariato.
- **Riproducibilità** (VERIFICATO). M4 di `main_v2` è identico a M4 di `main`: 57.600 righe, differenza 0.
- **Risultato** (M7 contro M4 negli ultimi 10.000 input; tra parentesi la versione difettosa di C.02).
  - CIFAR10: +0,032 [0,022, 0,042] a severity 3 (+0,032), frazione recuperata 15,5% (15,9%); +0,042 a severity 5
    (+0,043). Rischio di M7 0,049.
  - **CIFAR100**: +0,012 [0,009, 0,016] a severity 3 (+0,015), frazione recuperata 12,5% (15,3%); +0,020 a severity
    5 (+0,022). Rischio di M7 0,049 (0,051).
  - M8c − M7: +0,004 e +0,005 su CIFAR10, +0,002 e +0,002 su CIFAR100.
  - **Payback** (mediana su tutti i casi, non ripagati contati come infiniti): CIFAR10 oltre i 30.000 input a severity
    3 (ripagato nel 39% dei casi; tra questi, mediana 20.800) e 24.700 a severity 5 (58%); **CIFAR100 27.100 a
    severity 3 (56%) e 19.350 a severity 5 (73%)**, contro 12.550 e 11.100 della versione difettosa. Guadagno netto
    sui 30.000 input dopo lo shift: −0,003 e +0,014 su CIFAR10, +0,003 e +0,010 su CIFAR100.
  - **Garanzia**, ora anche su CIFAR10 (finestra sull'intero stream): rischio medio sui 40.000 input 0,0545 per M4 e
    0,0532 per M7 su CIFAR10 (sotto ε + 0,01 nell'80% e nell'88% delle ripetizioni: pesa il transitorio dopo lo
    shift); 0,0495 e 0,0483 su CIFAR100 (94% e 98%). Il bound regge sempre.
- **Previsioni dei rilanci.** Giusta sul guadagno (−0,001 su CIFAR10, −0,003 su CIFAR100). **Sbagliata sul payback
  di CIFAR100**: si allunga di 14.550 input, non di 1.000–4.000. La versione difettosa gonfiava i savings proprio
  nei 15.000 input dopo il cambio delle teste, quelli che decidono il pareggio.
- **Cosa cambia rispetto a C.02.** Cade la frase che su CIFAR100 la fase si ripaga prima che su CIFAR10: con la
  ripartenza corretta i payback mediani sono simili (oltre 30.000 e 27.100 a severity 3). Su uno stream di 30.000
  input dopo lo shift la fase di adattamento è circa in pareggio a severity 3 (da −0,003 a +0,003) e guadagna a
  severity 5 (+0,010–0,014); a regime recupera il 12–16% del calo.
- **H4 sulla versione corretta** (`fase_c/h4_v2.log`): nei 30.000 input dopo lo shift gli audit in più costano 715 e
  512 forward complete su CIFAR10, 184 e 158 su CIFAR100; il training 0,4 e 11; il guadagno lordo vale 625 e 923, 286
  e 484. Il training pesa lo 0,1% e il 6–7%: H4 resta vera. Su CIFAR10 a severity 3 gli audit in più costano più di
  quanto la fase faccia guadagnare entro lo stream.
- **Un incidente, senza effetti.** Il worker del primo rilancio, fermato, ha scritto 3 risultati nella cartella nuova
  dopo che l'avevo cancellata, e il rilancio li ha saltati. Rifatti da capo: identici (differenza 0), sostituiti.

### C.06 (15 nov) Ritorno al pulito e sensibilità a π e γ: previsioni

Scritta alle 15:36 del 15 novembre (ora letta da `date`: 15:35), prima dei due job, mentre girano le varianti corrette
su CIFAR100.

- **Ritorno al pulito.** Stream di 10.000 input puliti, 15.000 shiftati, 15.000 puliti; le 8 reti del seed 0, le 15
  corruzioni a severity 3, 6 ripetizioni; M4, M7, M7c, M8c; π = 0,05, γ = 2e-4; ripartenza come in C.04. La fase di
  M7 parte solo allo shift (trigger oracolo), quindi al ritorno M7 e M8c tengono le mappe del dominio shiftato; M7c le
  rifitta ogni 5.000 input. Domanda: l'adattamento lascia un costo quando torna il dominio pulito, e quanto ci mette
  ogni metodo a tornare ai savings di prima dello shift? Lettura con `verifica_fable/c_return_sens_read.py return`, a
  cui aggiungo savings e rischio negli ultimi 5.000 input dello stream (dalle curve medie sulle ripetizioni).
- **Ancore già misurate** (M4, reti del seed 0, severity 3). CIFAR10 (`fase_c/variants_v2`): la soglia sale da 0,887
  prima dello shift a 0,926 negli ultimi 10.000 input shiftati; gli audit costano 0,028 di savings sul pulito e 0,016
  dopo lo shift; finestra peggiore 0,110. CIFAR100 (`fase_c/main_v2`): la soglia resta quasi uguale (0,849 e 0,851);
  audit 0,012 e 0,007; finestra peggiore 0,082. La fase di M7 finisce in media 8.900 input dopo lo shift su CIFAR10,
  quindi prima del ritorno.
- **Previsioni sul ritorno.**
  - R1. M4 nei primi 5.000 input dopo il ritorno: rischio sotto ε (al massimo 0,045 su CIFAR10, 0,050 su CIFAR100),
    perché la soglia è più alta del necessario.
  - R2. M4 torna entro 0,01 dai savings di prima dello shift dopo 4.000–12.000 input su CIFAR10 (mediana; la soglia
    deve scendere di 0,039 a circa γ(ε − rischio) per input) e dopo al massimo 2.000 su CIFAR100. Perdita sui 15.000
    input del ritorno rispetto a prima dello shift: 0,005–0,030 su CIFAR10, sotto 0,010 su CIFAR100.
  - R3. M7 e M8c, con le mappe del dominio shiftato tenute sul pulito: negli ultimi 5.000 input savings netti sotto
    quelli di M4 di 0,000–0,040 su CIFAR10 e di 0,000–0,020 su CIFAR100; rischio entro ε ± 0,005.
  - R4. M7c negli ultimi 5.000 input sta entro 0,010 da M4 (le sue mappe sono state rifittate su righe pulite), e sui
    15.000 input del ritorno sta sopra M7.
  - R5. Nessun picco di rischio al ritorno: per tutti e quattro i metodi il rischio medio nei primi 5.000 input dopo il
    ritorno resta sotto ε + 0,005.
- **Sensibilità.** Stesso stream del job principale (10.000 puliti, 30.000 shiftati), le 8 reti del seed 0, severity
  3, 6 ripetizioni: π = 0,02 e π = 0,1 per M4 e M7; γ = 1e-4, 5e-4, 1e-3 per M4 e γ = 5e-4 per M7. Riferimento: M4 e
  M7 di `variants_v2` (π = 0,05, γ = 2e-4, stesse reti; 6 ripetizioni su CIFAR10, 4 su CIFAR100). Prima CIFAR10, poi
  CIFAR100.
- **Previsioni sulla sensibilità** (ultimi 10.000 input, contro il riferimento).
  - S1. π = 0,02: M4 guadagna da +0,005 a +0,010 di savings netti su CIFAR10 e da +0,002 a +0,005 su CIFAR100 (gli
    audit costano il 60% in meno; un po' di savings lordi si perde per la soglia più rumorosa); rischio entro ±0,003;
    finestra peggiore entro ±0,01 (la decide il primo migliaio di input dopo lo shift, quando la soglia non si è
    ancora mossa).
  - S2. π = 0,1: M4 perde da 0,010 a 0,017 su CIFAR10 e da 0,005 a 0,008 su CIFAR100; rischio entro ±0,003; finestra
    peggiore entro ±0,01.
  - S3. Il guadagno di M7 su M4 resta entro ±0,005 da quello a π = 0,05, per entrambi i π (gli audit della fase sono
    al 25% comunque).
  - S4. γ: il transitorio di M4 scala circa come 1/γ (con 1e-4 da 1,5 a 2,5 volte quello con 2e-4, con 5e-4 da 0,3 a
    0,6 volte, con 1e-3 da 0,15 a 0,4 volte); la finestra peggiore scende al crescere di γ; i savings netti negli
    ultimi 10.000 input restano entro ±0,005; su CIFAR10 il rischio sull'intero stream scende da γ = 1e-4 a 5e-4.
  - S5. Con γ = 5e-4 il guadagno di M7 su M4 resta entro ±0,005 da quello con γ = 2e-4.

### C.07 (15 nov) Gradini di severity e corruzioni miste: previsioni

Scritta alle 15:44 del 15 novembre (ora letta da `date`: 15:42), prima di ogni job con questi stream.

- **Perché.** Il piano v3 (sezione 9) chiede anche stream con rampa di severity, gradini e corruzioni miste, che in
  C.01 non avevo messo. Il payback ha senso solo se il dominio resta: questi stream misurano cosa succede quando
  cambia ancora. Aggiunti a `fase_c/run.py` come stream `steps` e `mixed` (codice nuovo, i vecchi stream non cambiano).
  La rampa continua la lascio: i gradini ne sono la versione a scalini.
- **Disegno.** Le 8 reti del seed 0, 15 corruzioni, γ = 2e-4, π = 0,05, ripartenza come in C.04; 6 ripetizioni su
  CIFAR10 e 4 su CIFAR100. Gradini: 10.000 input puliti, poi la stessa corruzione a severity 1, 2, 3, 4, 5, 6.000
  input ciascuna. Miste: 10.000 puliti, poi tre corruzioni a severity 3 di famiglie diverse, 10.000 input ciascuna (la
  corruzione del caso, poi quelle 5 e 10 posti dopo nell'elenco di `src/eesavings/data/cifar.py`, per esempio
  gaussian noise, motion blur, brightness). Metodi: M2 (soglia oracolo di ogni segmento, senza audit), M4, M7 (fase al
  primo cambio, una volta sola), M7t (fase quando la soglia sale, una volta sola), M7c (continuo). La fase resta una
  sola per costruzione del simulatore: un trigger che si riarma dopo ogni fase è una scelta di disegno che lascio a
  Davide. Lettura con `verifica_fable/c_return_sens_read.py segments DIR`.
- **Previsioni sui gradini** (CIFAR10; su CIFAR100 vedi G8).
  - G1. M4: finestra peggiore dopo il primo cambio sotto 0,085 (contro 0,110 del salto a severity 3: ogni gradino è
    più piccolo); savings netti sui 30.000 input shiftati da 0,00 a +0,02 sopra M2 (il controller resta indietro a
    ogni gradino e intanto spende un po' oltre ε), con rischio su quei 30.000 tra ε ed ε + 0,01.
  - G2. M7, con la fase al primo gradino (severity 1 e 2): sui 30.000 input da −0,03 a 0,00 contro M4; negli ultimi
    5.000 (severity 5) da +0,005 a +0,025, circa metà di quello che guadagna nel salto a severity 5.
  - G3. M7t scatta in almeno il 70% delle esecuzioni, quasi sempre nei gradini 3–5; negli ultimi 5.000 input guadagna
    più di M7.
  - G4. M7c: sui 30.000 input da +0,005 a +0,025 contro M4, il migliore dei tre; negli ultimi 5.000 da +0,010 a +0,030.
- **Previsioni sulle corruzioni miste** (CIFAR10).
  - G5. M4: finestra peggiore tra 0,09 e 0,13 (ogni cambio è grande come un salto); savings netti sui 30.000 input
    entro ±0,01 da M2.
  - G6. M7: contro M4 da −0,06 a −0,02 nel primo segmento (la fase), da −0,03 a +0,01 nel secondo e nel terzo (le
    mappe della prima corruzione sulle altre), da −0,04 a −0,01 sui 30.000. Una fase singola non si ripaga se il
    dominio cambia ogni 10.000 input.
  - G7. M7c: sui 30.000 input da 0,000 a +0,015 contro M4, l'unico metodo con adattamento non sotto M4; rischio sui
    30.000 non oltre ε + 0,005.
  - G8. Su CIFAR100 gli stessi ordinamenti, con differenze circa dimezzate.

### C.08 (15 nov) Le varianti con la ripartenza corretta

- **Esecuzione.** `fase_c/variants_v2`: CIFAR10 dalle 15:22:02 alle 15:27:02 (M4, M5, M6, M7, M7f, M7c, M7t, 6
  ripetizioni), CIFAR100 dalle 15:27:24 alle 15:47:25 (4 ripetizioni e senza M6, per stare sopra il 30% di batteria),
  exit 0. L'`args.json` della cartella è quello dell'ultimo lancio (CIFAR100): ogni lancio lo riscrive, da qui in poi
  ne tengo una copia per dataset. Lettura: `verifica_fable/c_variants_read.py variants_v2`
  (`fase_c/variants_v2_read.log`), con il lettore corretto alle 15:33: ogni metodo da una sola cartella (prima, su
  CIFAR100, avrebbe ripreso per M6 le curve difettose di `variants`). M3 resta da `variants` (il difetto non lo tocca).
- **Risultato** (ultimi 10.000 input, guadagno su M4 appaiato, IC con bootstrap su reti e corruzioni; tra parentesi
  la quota del guadagno di M7).
  - CIFAR10 (S_0 0,511, S_1 0,302): M7 +0,035 [0,021, 0,051], frazione recuperata 16,5%; M7f +0,041 (119%); M7c
    +0,021 (62%); M5 +0,016 (45%); M7t +0,012 (34%); M6 +0,003 (8%).
  - CIFAR100 (S_0 0,230, S_1 0,132): M7 +0,013 [0,008, 0,018], frazione 12,9%; M7f +0,020 (159%); M7c +0,007 (56%);
    M5 +0,002 (17%); M7t +0,001 (6%).
  - Netto sui 30.000 input dopo lo shift, payback mediano e quota ripagata entro lo stream. CIFAR10: M7c +0,015 (11.400
    input, 73%), M5 +0,009 (15.850, 60%), M7 −0,001 (oltre lo stream, 40%), M7t 0,000 (30%), M7f −0,018 (27%), M6
    −0,044 (0%). CIFAR100: M7 +0,003 (27.200, 57%), M7c +0,001 (27.200, 57%), M5 −0,001 (48%), M7f −0,004 (33%), M7t
    0,000 (8%).
  - Le fasi (input dallo shift alla fine): M7 8.900 su CIFAR10 e 5.600 su CIFAR100 (4.000 righe osservate, di cui
    1.230 e 410 audit); M7f 4.000 (tutto completato: savings netti −0,001 e −0,011 nei primi 1.000 input); M5 14.600 e
    6.600; M6 22.600 su CIFAR10. M7t scatta nel 63% e nel 20% delle ripetizioni, in media 7.400 e 8.500 input dopo lo
    shift; nessun falso allarme sul pulito.
  - Rischio negli ultimi 10.000 input tra 0,049 e 0,051 per tutti, tranne M7c su CIFAR100 (0,055).
- ~~**M7c su CIFAR100 sul pulito** (VERIFICATO sugli eventi): savings netti 0,195 contro 0,230 di M4, rischio 0,037,
  exit 9 raggiunto dal 48% degli input contro il 43%.~~ **Corretto nella voce C.17**: in questo run i run
  indipendenti sul pulito sono solo 16 (il segmento pulito è lo stesso per le 15 corruzioni), e in un altro run (il
  ritorno al pulito) M7c sul pulito sta a +0,002 da M4. Quello che resta vero: M7c sul pulito di CIFAR100 è
  instabile. Il meccanismo sotto vale per le esecuzioni in cui la soglia non riparte. Le righe osservate sul pulito
  sono per il 94% input arrivati in fondo da soli, cioè i più difficili; le mappe fittate su di loro abbassano la
  confidenza delle teste su tutti gli input. La soglia riparte solo nel 38% dei rifit (test a due errori standard con
  audit pesati 20) e il controller scende di circa γ(ε − rischio) = 2,6e-6 per input, troppo poco tra un rifit e
  l'altro. Prima della correzione di C.04 l'errore era opposto (rischio 0,056–0,060). Su CIFAR10 l'effetto non c'è
  (0,510 contro 0,511).
- **Lettura.** L'ordine M7f > M7 > M5 a regime segue quanto le righe di training sono un campione non distorto del
  dominio: M7f le completa tutte, M7 aggiunge audit al 25% agli input arrivati in fondo, M5 ha solo questi ultimi. Con
  righe distorte verso gli input difficili le mappe tolgono confidenza anche dove non serve. Un fit pesato con 1/q (q
  la probabilità di essere osservata, già nel buffer) stima lo stesso bersaglio di M7f al costo di M7: lo provo come
  M7w (e M7cw per l'adattamento continuo), con le previsioni sotto, dopo i job già lanciati.
- **Previsioni di C.01.**
  - 5 (M3), da `variants`: sbagliata. Su CIFAR10 il monitor scatta solo nel 55% delle ripetizioni, con mediana 4.373
    input dopo lo shift (prevedevo 500–3.000); su CIFAR100 nel 18%, mediana 5.632. Dopo, savings a zero; netto sullo
    stream −0,097 e −0,019 contro M4.
  - 10 (H5): **sbagliata nei numeri, H5 vera**. M5 recupera il 45% e il 17% di quello che recupera M7 (prevedevo
    50–90%), con rischio non più alto (0,049 contro 0,050 su CIFAR10, 0,049 contro 0,051 su CIFAR100).
  - 11 (M6): giusta sulla durata (22.600 input), sbagliata sul resto: entro lo stream recupera l'8% di M7, perché la
    fase finisce a due terzi dello stream; netto −0,044.
  - 12 (M7f): giusta sulla fase più breve e più cara; **sbagliata sul payback**, più lungo di M7 (ripagato nel 27% e
    nel 33% dei casi contro il 40% e il 57%), anche se a regime recupera di più (119% e 159%).
  - 13 (M7c): sbagliata. A regime recupera il 62% e il 56% di M7, non tutto; ma su CIFAR10 è il migliore sul netto
    dello stream (+0,015), perché non paga una fase. Su CIFAR100 non è affidabile (sopra).

### C.09 (15 nov) Il fit pesato delle mappe (M7w, M7cw): previsioni

Scritta alle 15:51 del 15 novembre (ora letta da `date`: 15:50), prima di scrivere il codice e di ogni job.

- **Cosa cambia.** Nel fit delle mappe ogni riga osservata pesa 1/q, con q la probabilità che aveva di essere
  osservata (1 per gli input arrivati in fondo da soli, π_a = 0,25 per gli audit della fase, π = 0,05 per quelli
  dell'adattamento continuo): la loss di distillazione diventa una media pesata, stima non distorta di quella sul
  dominio intero. M7w è M7 con il fit pesato, M7cw è M7c con il fit pesato. Tutto il resto uguale (righe tenute da
  parte, test prima della ripartenza). Costo uguale a M7 e M7c. Numero effettivo di righe stimato dagli eventi di
  C.08: circa 2.100 su 3.000 per M7w; circa 350–450 per M7cw, poche per 200 parametri a testa su CIFAR100.
- **Disegno.** Stream con salto come le varianti, le 8 reti del seed 0, 15 corruzioni a severity 3; 6 ripetizioni su
  CIFAR10 e 4 su CIFAR100 (come `variants_v2`), metodi M4, M7, M7w, M7cw, cartella `fase_c/variants_w`.
- **Previsioni.**
  - W1. M4 e M7 identici a quelli di `variants_v2` (differenza 0): il percorso senza pesi non cambia.
  - W2. M7w negli ultimi 10.000 input sta tra M7 e M7f, vicino a M7f: guadagno su M4 da +0,037 a +0,043 su CIFAR10
    (M7 +0,035, M7f +0,041) e da +0,016 a +0,021 su CIFAR100 (M7 +0,013, M7f +0,020); rischio entro ε ± 0,003.
  - W3. La fase di M7w dura e costa come quella di M7, quindi si ripaga prima: ripagato entro lo stream in almeno il
    50% dei casi su CIFAR10 (M7 40%) e il 65% su CIFAR100 (M7 57%); netto sullo stream sopra M7 di +0,003–0,008.
  - W4. M7cw su CIFAR100 sul pulito: savings netti entro 0,015 da M4 (M7c perde 0,035), rischio tra 0,042 e 0,052;
    dopo lo shift guadagno su M4 almeno pari a quello di M7c (+0,007). Su CIFAR10 uguale a M7c entro ±0,005.

### C.10 (15 nov) Ritorno al pulito: risultato su CIFAR10

- **Esecuzione.** `fase_c/return` (comando in `fase_c/run_c06.sh`), CIFAR10 dalle 15:47:39 alle 15:50:40, exit 0; 4
  reti del seed 0, 15 corruzioni a severity 3, 6 ripetizioni; stream di 10.000 input puliti, 15.000 shiftati, 15.000
  puliti. Lettura: `verifica_fable/c_return_sens_read.py return` (`fase_c/return_read_cifar10.log`), a cui ho aggiunto
  dopo la prima lettura il rischio sull'intero stream e il transitorio del rischio dopo il ritorno.
- **M4** (solo controller). Al ritorno la soglia è più alta del necessario: rischio 0,034 nei primi 5.000 input
  puliti. I savings tornano entro 0,01 da quelli di prima dello shift dopo 4.950 input (mediana); la perdita sui
  15.000 input del ritorno è 0,028 (0,045 nei primi 5.000); negli ultimi 5.000 savings 0,496 contro 0,511.
- **M7, M8c e M7c** (le mappe del dominio shiftato restano; M7c le rifitta). Subito dopo il ritorno hanno più savings
  di M4 (0,497, 0,501 e 0,486 nei primi 5.000 input, contro 0,466), **ma con il rischio sopra ε**: 0,075, 0,073 e
  0,069 nei primi 1.000 input; 0,068, 0,066 e 0,064 nei primi 5.000; 0,059, 0,058 e 0,055 sui 15.000 del ritorno.
  Negli ultimi 5.000 ancora 0,054, 0,052 e 0,050, con savings sotto M4 di 0,020, 0,015 e 0,014. Rischio medio
  sull'intero stream di 40.000 input: M4 0,050, M7 0,057, M7c 0,057, M8c 0,050 (M8c era prudente durante lo shift).
- **Il meccanismo** (VERIFICATO sulle curve e sugli eventi). La ripartenza dopo il fit abbassa la soglia di M7 da
  0,92 a 0,79 in media; tutte le fasi finiscono prima del ritorno (al più 13.200 input dopo lo shift). Al ritorno le
  teste con le mappe del dominio shiftato sono più sicure sugli input puliti, e con la soglia a 0,78 escono troppi
  input che non concordano con l'exit 9. Il controller risale di γ(rischio − ε) per input, circa 5e-6: la soglia va da
  0,781 a 0,820 in 15.000 input e il rischio scende da 0,075 a 0,046 solo alla fine. Per M4 il ritorno è sicuro perché
  la sua soglia era solo salita.
- **Previsioni di C.06.** R1 giusta (0,034). R2 giusta (4.950 input, perdita 0,028). R3 giusta sui savings (−0,020 e
  −0,015), sbagliata sul rischio di M7 (0,054 negli ultimi 5.000). R4 sbagliata: M7c negli ultimi 5.000 sta 0,014
  sotto M4, e sui 15.000 del ritorno sotto M7 (0,483 contro 0,486). **R5 sbagliata**: il picco c'è, 0,064–0,068 nei
  primi 5.000 input dopo il ritorno.
- **Convinzione.** Un adattamento che abbassa la soglia trasforma il cambio di dominio successivo in un nuovo shift,
  con il rischio sopra ε, e i savings in più subito dopo il ritorno sono presi a prestito dal rischio. Rimedio
  possibile, da decidere con Davide: dopo la fase, un e-detector sugli audit (come quello di M3) che, quando scatta,
  toglie le mappe e fa ripartire la soglia da quella pulita.

### C.11 (15 nov) Sensibilità a π e γ: risultato su CIFAR10

- **Esecuzione.** `fase_c/sens_pi002`, `sens_pi01`, `sens_gamma` (comandi in `fase_c/run_c06.sh`), CIFAR10 dalle
  15:50:40 alle 15:54:10, exit 0; 4 reti del seed 0, 15 corruzioni a severity 3, 6 ripetizioni, stream con salto.
  Riferimento: M4 e M7 di `variants_v2` (π = 0,05, γ = 2e-4, stesse reti e ripetizioni). Lettura:
  `verifica_fable/c_return_sens_read.py sens` (`fase_c/sens_read_cifar10.log`).
- **Risultato** (M4; ultimi 10.000 input; transitorio come in C.02, mediana).

  | | Netti | Rischio | Finestra peggiore | Rischio su tutto lo stream | Transitorio | Guadagno di M7 su M4 |
  |---|---|---|---|---|---|---|
  | Riferimento (π 0,05, γ 2e-4) | 0,302 | 0,050 | 0,110 | 0,055 | 6.250 | +0,035 |
  | π 0,02 | 0,313 | 0,051 | 0,110 | 0,054 | 4.950 | +0,035 |
  | π 0,1 | 0,286 | 0,050 | 0,105 | 0,054 | 5.750 | +0,033 |
  | γ 1e-4 | 0,307 | 0,052 | 0,110 | 0,058 | 9.700 | |
  | γ 5e-4 | 0,299 | 0,050 | 0,108 | 0,052 | 2.900 | +0,037 |
  | γ 1e-3 | 0,298 | 0,051 | 0,109 | 0,052 | 2.150 | |

  Sul pulito, prima dello shift: savings netti 0,525 con π = 0,02 e 0,479 con π = 0,1, contro 0,511.
- **Previsioni di C.06.** S1 quasi giusta: +0,011 contro +0,005–0,010 (di 0,001 fuori), rischio e finestra peggiore
  giusti. S2 giusta (−0,016). S3 giusta (+0,035 e +0,033 contro +0,035). S4 giusta sul transitorio (1,55, 0,46 e 0,34
  volte quello con 2e-4) e sul rischio sull'intero stream (da 0,058 a 0,052), **sbagliata sulla finestra peggiore**,
  che non cambia con γ (0,108–0,110: nei primi 1.000 input la soglia non fa in tempo a muoversi nemmeno con γ = 1e-3),
  e sui savings con γ = 1e-4 (+0,006, fuori di 0,001: dopo 30.000 input il transitorio non è finito, rischio 0,052).
  S5 giusta (+0,037 contro +0,035).
- **Convinzioni.** Il tasso di audit sposta i savings netti quasi uno a uno con il costo degli audit (±0,01–0,016) e
  non cambia il guadagno della fase. γ decide la durata del transitorio: 5e-4 lo dimezza e abbassa il rischio di lungo
  periodo sotto shift, al prezzo di una finestra peggiore più alta sul pulito (0,082 contro 0,073, voce C.01). La
  regola fissata prima ha scelto 2e-4 guardando solo il pulito; resta quella, e la scelta di γ va presentata come un
  compromesso tra rumore sul pulito e velocità sotto shift.

### C.12 (15 nov) Il fit pesato delle mappe: risultato su CIFAR10

- **Codice** (dalle 15:54 alle 15:55, dopo la fine dei job su CIFAR10 che reimportavano il simulatore). In
  `fase_c/sim.py`: `fit_diag(logits, weights=None)`, con i pesi la loss è la media pesata della KL per riga; nel fit
  di una fase o del continuo, con `weighted`, ogni riga pesa 1/q. In `fase_c/run.py` i metodi M7w e M7cw. Prova su
  una rete (gaussian noise 3, 3.000 righe): pesi tutti uguali contro nessun peso, scala e bias uguali entro 0,001 e
  0,008 (tolleranza di L-BFGS, 359 e 382 valutazioni); con pesi diversi le mappe cambiano (0,075 sulla scala). Gli
  stream `steps` e `mixed` provati su una rete e una corruzione (finestre per segmento giuste).
- **Esecuzione.** `fase_c/variants_w`, CIFAR10 dalle 15:55:43 alle 15:59:13, exit 0; M4, M7, M7w, M7cw, 6 ripetizioni.
  Lettura: `verifica_fable/c_variants_read.py variants_v2 variants_w` (`fase_c/variants_w_read_cifar10.log`; il
  lettore ora sceglie la cartella per metodo e dataset, e ridà identica la lettura di `variants_v2`).
- **W1 giusta** (VERIFICATO, `verifica_fable/c_same_methods.py`, `fase_c/variants_w_same_cifar10.log`): M4 e M7
  identici a `variants_v2` su 3.600 righe, differenza 0.
- **Risultato** (ultimi 10.000 input, guadagno su M4).
  - **M7w**: +0,041 [0,030, 0,054], il 118% di M7 e il 99% di M7f, con la fase di M7 (8.900 input, stessi audit).
    Rischio 0,049. Ripagato entro lo stream nel 52% dei casi (M7 40%), mediana 29.800; netto sullo stream +0,004
    (M7 −0,001).
  - **M7cw**: +0,042 [0,030, 0,056], il 196% di M7c, quanto M7f, senza fase: netto sullo stream +0,028, ripagato nel
    98% dei casi, mediana 1.050 input. Sul pulito savings 0,519 contro 0,511 di M4, ma **rischio 0,054 sul pulito** e
    0,051 negli ultimi 10.000 input: le mappe con pochi audit pesati 20 sono rumorose e la ripartenza sceglie la
    soglia su stime rumorose.
- **Previsioni.** W2 giusta (+0,041 contro +0,037–0,043). W3 giusta (52% contro almeno 50%; netto +0,005 su M7). W4 su
  CIFAR10 sbagliata, in meglio: M7cw non è uguale a M7c, guadagna il doppio.
- **Convinzione, da confermare su CIFAR100.** Il difetto delle fasi di C.08 era il campione distorto, non le mappe:
  pesando le righe, la fase con audit al 25% vale quanto il fallback completo, e l'adattamento continuo vale quanto la
  fase senza pagarla. Resta da capire il rischio di M7cw, un po' sopra ε.

### C.13 (15 nov) M7cw sul ritorno al pulito, sui gradini e sulle corruzioni miste: previsioni

Scritta alle 16:02 del 15 novembre (ora letta da `date`: 16:00), prima dei job.

- **Perché.** Su CIFAR10 M7cw è il candidato migliore (C.12); va visto quando il dominio cambia ancora. Cartelle
  separate (`fase_c/return_w`, `steps_w`, `mixed_w`) con M4 e M7cw: il runner salta i casi già scritti, quindi non
  aggiungo metodi alle cartelle esistenti. M4 deve venire identico a quello delle cartelle di C.06 e C.07. Stesse reti,
  ripetizioni e stream di C.06 e C.07.
- **Previsioni.**
  - C1. Ritorno: M7cw ha il picco di rischio al ritorno come M7c (0,060–0,075 nei primi 1.000 input), ma rifitta su
    righe pulite entro 5.000–10.000 input: negli ultimi 5.000 savings entro 0,010 da M4 e rischio entro ε + 0,005.
  - C2. Gradini: M7cw è il migliore sui 30.000 input shiftati, da +0,010 a +0,030 contro M4.
  - C3. Corruzioni miste: M7cw è il migliore sui 30.000 input, da +0,005 a +0,025 contro M4, con un picco di rischio a
    ogni cambio (finestra peggiore sopra quella di M4 di 0,00–0,02).
  - C4. Su CIFAR100 le stesse direzioni, con differenze circa dimezzate, se M7cw su CIFAR100 regge come su CIFAR10.

### C.14 (15 nov) Gradini e corruzioni miste: risultato su CIFAR10

- **Esecuzione.** `fase_c/steps` dalle 15:59:13 alle 16:03:14, `fase_c/mixed` dalle 16:03:14 alle 16:06:44, exit 0;
  CIFAR10, 4 reti del seed 0, 15 corruzioni, 6 ripetizioni. Lettura: `verifica_fable/c_return_sens_read.py segments`
  (`fase_c/steps_read_cifar10.log`, `fase_c/mixed_read_cifar10.log`).
- **Gradini** (severity da 1 a 5, 6.000 input ciascuna; differenze da M4 appaiate, IC su reti e corruzioni).
  - M4 resta sempre indietro di un gradino: rischio 0,058–0,062 in ogni gradino, 0,060 sui 30.000 input shiftati,
    finestra peggiore 0,109; savings sopra M2 di 0,013 (M2 ha la soglia oracolo di ogni gradino, calibrata con la UCB).
  - M7 (fase al primo gradino, fit nel secondo nell'86% dei casi): −0,092 e −0,054 nei gradini 1 e 2, poi +0,021,
    +0,028, +0,032; sui 30.000 input −0,013 [−0,023, −0,002]. A severity 5 guadagna +0,032 con mappe fittate a
    severity 1–2, ma con rischio 0,060–0,062.
  - M7t scatta nel 73% delle esecuzioni, sparso su tutti i gradini (dal 12% al 20% per gradino); sui 30.000 input
    −0,002, negli ultimi 5.000 +0,013.
  - M7c: +0,009 [−0,005, +0,026] sui 30.000 input, +0,019 negli ultimi 5.000, rischio 0,056 e 0,049.
- **Corruzioni miste** (tre corruzioni di famiglie diverse a severity 3, 10.000 input ciascuna).
  - M4: rischio 0,070, 0,054, 0,051 nei tre segmenti, 0,058 sui 30.000; finestra peggiore 0,143; savings sopra M2 di
    0,004 sui 30.000.
  - **Tutti i metodi con adattamento hanno più rischio di M4**: sui 30.000 input M7 0,069, M7c 0,068, M7t 0,066;
    finestra peggiore 0,169, 0,168, 0,150. M7 nel secondo segmento, con le mappe della prima corruzione e la soglia
    ripartita in basso, ha rischio 0,077 e savings +0,034 su M4: presi a prestito dal rischio.
  - Savings sui 30.000 input contro M4: M7 −0,006 [−0,017, +0,007], M7t −0,005, M7c +0,019 [+0,007, +0,034].
- **Previsioni di C.07.** G1: sbagliata sulla finestra peggiore (0,109, non sotto 0,085: il controller non recupera
  mai il ritardo di un gradino), giusta sul resto (+0,013 su M2, rischio 0,060). G2: giusta sui 30.000 input (−0,013),
  sbagliata negli ultimi 5.000 (+0,032 contro +0,005–0,025). G3: giusta sul 73%, sbagliata su dove scatta e sul
  confronto con M7. G4 giusta (+0,009, il migliore; +0,019). G5: giusta su M2 (+0,004), sbagliata sulla finestra
  peggiore (0,143 contro 0,09–0,13). G6 sbagliata nei numeri (−0,062, +0,034, +0,009, −0,006): la fase non si ripaga,
  ma perde meno del previsto perché nel secondo segmento spende rischio. G7: giusta sull'ordine, sbagliata sul guadagno
  (+0,019 contro al massimo +0,015) e sul rischio (0,068 contro al massimo 0,055).
- **Convinzione.** L'adattamento specializza il sistema sul dominio corrente: le mappe più la soglia ripartita sono
  sicure solo lì. Quando il dominio cambia ancora (ritorno al pulito, gradino, altra corruzione) il sistema adattato
  parte più vicino al limite del sistema generico e il rischio sale di più. Con shift ripetuti i savings in più dei
  metodi adattivi sono in parte presi a prestito dal rischio. Serve una protezione dopo la fase (C.10) prima di
  raccomandare un adattamento in uno scenario con più shift.

### C.15 (15 nov) Il fit pesato su CIFAR100, e M7cw sugli stream con più cambi (CIFAR10)

- **Esecuzione.** `fase_c/variants_w` su CIFAR100 dalle 16:16:16 alle 16:36:17 (4 ripetizioni); `return_w`, `steps_w`,
  `mixed_w` su CIFAR10 dalle 16:07:46 alle 16:16:16 (6 ripetizioni); tutti exit 0. Letture in `fase_c/`:
  `variants_w_read.log`, `return_w_read_cifar10.log`, `steps_w_read_cifar10.log`, `mixed_w_read_cifar10.log`.
- **Controlli** (VERIFICATO, `c_same_methods.py`): M4 e M7 di `variants_w` identici a `variants_v2` su tutte le
  6.000 righe dei due dataset (`variants_w_same.log`); M4 delle cartelle `*_w` identico a quello di `return`, `steps`,
  `mixed` (4.320, 5.760, 5.040 righe, differenza 0; `w_streams_same_cifar10.log`).
- **Fit pesato su CIFAR100** (ultimi 10.000 input, guadagno su M4).
  - **M7w**: +0,016 [0,013, 0,020], il 129% di M7 e l'81% di M7f; rischio 0,051; ripagato nel 70% dei casi (M7 57%),
    mediana 21.300 (M7 27.200); netto sullo stream +0,005 (M7 +0,003).
  - **M7cw non regge**: +0,013 [0,008, 0,017], quanto M7; sul pulito savings 0,207 contro 0,230 di M4 (M7c 0,195 in
    questo run, ma vedi C.17) e
    rischio 0,048; negli ultimi 10.000 input shiftati **rischio 0,058**. Con 100 classi le mappe stimate da circa 400
    righe effettive (audit pesati 20) sono troppo rumorose, e la ripartenza sceglie la soglia su stime rumorose.
- **M7cw sugli stream di CIFAR10** (differenze da M4 appaiate).
  - Ritorno: picco al ritorno come gli altri (rischio 0,077 nei primi 1.000 input, 0,069 nei primi 5.000, 0,056 sui
    15.000), poi rifitta su righe pulite: negli ultimi 5.000 input savings sopra M4 di 0,010 con rischio 0,049. Rischio
    sull'intero stream 0,057 contro 0,050 di M4.
  - Gradini: +0,020 [0,011, 0,030] sui 30.000 input, +0,038 negli ultimi 5.000, **con rischio più basso di M4** (0,056
    contro 0,060; finestra peggiore 0,104 contro 0,109): rifittando segue la deriva meglio di quanto il controller da
    solo insegua la soglia.
  - Corruzioni miste: +0,027 [0,019, 0,036] sui 30.000 input, ma rischio 0,068 contro 0,058 e finestra peggiore 0,187
    contro 0,143.
- **Previsioni.** W2 su CIFAR100 giusta (+0,016 contro +0,016–0,021). W3 su CIFAR100: giusta sulla quota ripagata
  (70%), di poco sbagliata sul netto (+0,002 su M7 contro +0,003–0,008). W4 su CIFAR100 sbagliata: perde 0,023 sul
  pulito (non entro 0,015) e il rischio dopo lo shift è 0,058. C1: picco 0,077 (di poco sopra 0,060–0,075), giusta sugli
  ultimi 5.000 input. C2 giusta (+0,020). C3 sbagliata in due punti: guadagno +0,027 (sopra +0,005–0,025) e finestra
  peggiore più alta di M4 di 0,043 (non 0,00–0,02).
- **Convinzione.** Il fit pesato va tenuto nella fase (M7w): su entrambi i dataset recupera di più (118% e 129% di M7),
  allo stesso costo e con lo stesso rischio. L'adattamento continuo pesato è il migliore su CIFAR10 quando lo shift è
  graduale, ma su CIFAR100 non è affidabile, e con corruzioni che cambiano a salti spende rischio come gli altri.

### C.16 (15 nov) Ritorno al pulito: risultato su CIFAR100

- **Esecuzione.** `fase_c/return` su CIFAR100 dalle 16:36:42 alle 16:58:13 e `fase_c/return_w` dalle 16:58:13 alle
  17:19:44, exit 0, 6 ripetizioni. Batteria dal 49% al 42% pur in carica: i job seguenti girano con 2 worker. M4 di
  `return_w` identico a quello di `return` (8.640 righe dei due dataset, differenza 0). Letture: `fase_c/return_read.log`,
  `fase_c/return_w_read.log` (contengono anche CIFAR10).
- **M4.** Il ritorno non lo tocca: rischio 0,045 nei primi 5.000 input puliti, savings subito come prima dello shift
  (perdita 0,003 nei primi 5.000, 0,000 sui 15.000). Su CIFAR100 la sua soglia sotto shift quasi non si muove (0,851
  contro 0,845).
- **I metodi adattati, picco molto più forte che su CIFAR10.** Rischio nei primi 1.000 input dopo il ritorno: M7
  0,125, M8c 0,115, M7c 0,128, M7cw 0,144; nei primi 5.000: 0,108, 0,100, 0,110, 0,122; sui 15.000: 0,084, 0,080,
  0,073, 0,080; negli ultimi 5.000 ancora 0,065, 0,063, 0,054, 0,057. Il transitorio dopo il ritorno dura in mediana
  10.500 input per M7 e 9.500 per M8c, che nel 40% e nel 33% dei casi non tornano sotto ε + 0,01 entro i 15.000; 6.000
  per M7c e M7cw. Rischio sull'intero stream di 40.000 input: M4 0,048, M7 0,061, M8c 0,057, M7c 0,059, M7cw 0,060.
  Intanto i savings subito dopo il ritorno sono più alti di quelli di prima dello shift (0,264 contro 0,231 per M7):
  presi a prestito dal rischio.
- **Il meccanismo** è quello di C.10, più grande: la ripartenza porta la soglia di M7 da 0,85 a 0,68 in media (0,70 per
  M8c); sul pulito, con le mappe del dominio shiftato, a quella soglia il rischio è 0,125, e il controller risale di
  circa γ(rischio − ε) = 1,5e-5 per input.
- **Previsioni di C.06 e C.13 su CIFAR100.** R1 giusta (0,045). R2 giusta (1.000 input, perdita 0,000). R3 giusta sui
  savings (−0,014 e −0,010), sbagliata sul rischio (0,065 e 0,063). R4 giusta (+0,005 su M4 negli ultimi 5.000; sui
  15.000 sopra M7). **R5 sbagliata, di molto**: 0,100–0,110 nei primi 5.000 input. C1 sbagliata sul picco (0,144) e
  sul rischio finale (0,057), giusta sui savings finali (+0,010).
- **Convinzione.** Su CIFAR100 il problema del cambio di dominio dopo l'adattamento è più grave che su CIFAR10: il
  controller da solo attraversa il ritorno senza danni, i metodi adattati passano 5.000–10.000 input con il rischio
  a due volte ε. La protezione dopo la fase (C.10) è necessaria prima di raccomandare qualunque adattamento.

### C.17 (15 nov) Correzione: il segmento pulito è lo stesso per tutte le corruzioni

Scritta alle 17:26 del 15 novembre (ora letta da `date`: 17:24).

- **Cosa ho notato.** Nella figura del ritorno M7c sul pulito di CIFAR100 ha savings 0,233 e rischio 0,052; in
  `variants_v2` aveva 0,195 e 0,037 (voce C.08). Stesse reti e corruzioni, codice di M7c uguale (il fit pesato non
  tocca il percorso senza pesi; M4 e M7 rifatti sono identici al bit, C.12 e C.15); cambiano solo le ripetizioni (6
  contro 4) e quindi le estrazioni degli audit.
- **Spiegazione** (VERIFICATO, controllo in linea alle 17:24). Ogni caso usa lo stesso seed per lo stream e per gli
  audit, quindi per una rete e una ripetizione i primi 10.000 input puliti sono lo stesso run per tutte le 15
  corruzioni: la differenza tra corruzioni dei savings prima dello shift è esattamente 0, per M4 e per M7c. Sul pulito
  i run indipendenti sono 4 reti per il numero di ripetizioni: 16 in `variants_v2`, 24 in `return`. Al primo rifit
  (5.000 input) la soglia di M7c riparte nel 50% dei run in `variants_v2` e nel 79% in `return`; se riparte, savings
  circa come M4 e rischio sopra ε, se no savings più bassi e rischio basso. Con così pochi run indipendenti la media
  balla: M7c contro M4 sul pulito −0,035 in un run e +0,002 nell'altro (media dei 40 run circa −0,013).
- **Cosa cambia.**
  - La frase di C.08 "M7c su CIFAR100 perde 0,035 sul pulito" è sbagliata come stima: corretta lì. Resta vero che M7c
    sul pulito di CIFAR100 è instabile, e il suo rischio dopo lo shift sta sopra ε (0,055).
  - M7cw sul pulito di CIFAR100 invece è stabile tra i due run: 0,207 e 0,211 contro 0,230 e 0,231 di M4.
  - **Un limite degli intervalli.** Le stesse estrazioni (stream e audit) per tutte le corruzioni di una rete sono numeri
    casuali comuni: aiutano i confronti appaiati tra metodi, ma la parte casuale che dipende dal pulito (per esempio la
    prima ripartenza di M7c) è condivisa tra le corruzioni, e il bootstrap su reti e corruzioni la tratta come
    indipendente. Gli intervalli dei metodi con un comportamento sul pulito che dipende dal caso (M7c, M7cw) sono
    quindi troppo stretti. Per M4, M7, M7w e M8c il pulito è lo stesso per tutti i metodi a confronto, e l'effetto sui
    guadagni dopo lo shift è minore. Lo scrivo nei limiti del Checkpoint 3b; per i prossimi job conviene un seed per
    corruzione.

### C.18 (15 nov) Il Mac si è bloccato durante i gradini su CIFAR100

Scritta alle 17:57 del 15 novembre (ora letta da `date`: 17:56).

- **Cosa è successo.** Il job `steps` su CIFAR100 (partito alle 17:20:43, 2 worker, 4 ripetizioni) si è fermato
  verso le 17:33 (ultima riga di `fase_c/steps_cifar100.log` alle 17:33): Davide riferisce che il Mac si è bloccato
  e spento; riacceso verso le 17:48 (uptime di 7 minuti alle 17:55). Nessun processo del job è rimasto vivo; la riga
  di fine in `STATUS.txt` manca.
- **Cosa resta** (VERIFICATO alle 17:55). 26 casi su 60 salvati (le reti disjoint e joint), tutti integri: 320 righe
  per sintesi, curve di 400 blocchi per i 5 metodi, eventi presenti. Le sintesi si scrivono con un file temporaneo e
  una rinomina, quindi un caso o c'è tutto o manca; 4 file di curve senza sintesi appartengono a casi da rifare, che
  il runner ricalcola da capo. Le corruzioni miste non erano partite. Quaderno e documenti scritti oggi sono integri.
- **La causa** non la so. Non trovo un report di panic leggibile (al riavvio il kernel cita il servizio dei panic,
  ma non è una prova). Alle 13:32 macOS aveva registrato una forte pressione di memoria (`JetsamEvent`, circa 7 GB
  compressi, 66 MB liberi), con in testa Codex, Spotify, VS Code, Safari e Claude, ciascuno sopra 1,2 GB; i worker
  Python non erano tra i primi otto. I job su CIFAR100 aggiungono carico di CPU e un po' di memoria: possono aver
  contribuito.
- **Ripresa.** Ogni caso dipende solo dal suo seed: rilanciando, il runner salta i 26 casi fatti e i risultati sono
  quelli che avrebbe dato un job senza interruzioni.

### C.19 (15 nov) Secondo blocco: un panic del kernel sul bus I2C, la diagnosi

Scritta alle 18:47 del 15 novembre (ora letta da `date`: 18:46).

- **Cosa è successo.** Davide ha scelto di riprendere con 2 worker; il job è ripartito alle 17:57:06 e il Mac si è
  bloccato di nuovo dopo circa 6 minuti: il panic è delle 18:03:40 (orologio del report), il riavvio delle 18:37 con
  un reset dal tasto di accensione (`ResetCounter`: `btn_rst`). Prima del blocco si erano salvati altri 14 casi: ora
  40 su 60 dei gradini su CIFAR100 (joint e disjoint tutti, mixed 10), tutti integri.
- **Il panic** (report completo in `/Library/Logs/DiagnosticReports/Retired/panic-full-2026-11-15-183802.0002.panic`,
  lo stesso testo incollato da Davide): il controller I2C del bus `i2c1` trova il bus ancora in uno stato anomalo
  (`_checkBusStatus`, `AppleS5L8940XI2C.cpp:571`) per il dispositivo `mogul-las`; nello stack ci sono solo driver del
  kernel (piattaforma, I2C, AppleEmbeddedAudio, AppleCS42L84Audio).
- **Diagnosi** (VERIFICATA sull'IORegistry e sui log di sistema).
  - Sul bus `i2c1` ci sono solo due dispositivi: il codec audio del jack cuffie (Cirrus CS42L84, indirizzo 0x4B, con
    l'interfaccia dei tasti delle cuffie) e `mogul-las` (indirizzo 0x10), un chip di autenticazione di Apple gestito
    da `com.apple.driver.AppleAuthCP`. Il driver audio ha trovato il bus bloccato e il kernel si è fermato.
  - A entrambi i riavvii di oggi (17:47:52 e 18:37:53) il chip di autenticazione dà errori di bus mentre il kernel ne
    legge il certificato (`MogulAuthI2CRelayInterface::_read: bus error during certificate read`, 3 volte per avvio).
    Nei log conservati, dal 12 novembre alle 18:45, questi errori non compaiono prima del primo blocco; non so se
    c'erano agli avvii precedenti, che non sono nei log.
  - Il primo blocco (17:33) ha lasciato al riavvio le tracce del servizio dei panic ma non il testo: probabilmente è lo
    stesso panic, non verificato.
  - Una ricerca sul testo del panic: lo stesso tipo di errore (bus I2C bloccato) è riportato su un MacBook Air con il
    codec audio e su iPhone, dove è stato risolto cambiando un componente. Indica un guasto hardware del bus o di un
    dispositivo collegato, non un programma utente.
  - Il mio job non usa né l'audio né l'I2C: non è la causa. Però i due blocchi sono arrivati 13 e 6,5 minuti dopo
    l'avvio di un job con 2 worker, mentre stamattina 5 ore di job con 3 worker non hanno dato problemi. Il carico
    (calore, consumo) può fare da innesco su un componente già difettoso. In più il caricatore collegato è da 30 W
    (`pmset -g ac`): sotto carico la batteria scende anche in carica (dal 50% al 40% nel pomeriggio).
- **Decisione.** Nessun nuovo job sul Mac prima della decisione di Davide. Proposte: Apple Diagnostics, un caricatore
  da almeno 70 W per i job pesanti, e chiudere il Checkpoint 3b senza i job mancanti su CIFAR100 (20 casi dei
  gradini, le corruzioni miste, la sensibilità).
- **Decisione di Davide** (in chat, 18:46): "Riprendi con 1 worker". Ripreso alle 18:47:24 con 1 worker.

### C.20 (15 nov) Gradini, corruzioni miste e sensibilità su CIFAR100

- **Esecuzione** (1 worker, dopo i due panic). `steps` (20 casi mancanti) dalle 18:47:24 alle 19:04:24, `mixed` dalle
  19:04:24 alle 19:52:56 (4 ripetizioni), `sens_pi002`, `sens_pi01`, `sens_gamma` dalle 19:52:56 alle 20:23:57 (6
  ripetizioni); tutti exit 0, batteria dal 42% al 62% in carica, nessun errore I2C nel kernel dopo la ripresa. Letture:
  `fase_c/steps_read.log`, `mixed_read.log`, `sens_read.log` (contengono anche CIFAR10).
- **Gradini** (differenze da M4 sui 30.000 input shiftati, IC su reti e corruzioni). M4: rischio 0,053, finestra
  peggiore 0,083, savings sopra M2 di 0,007. Nessun metodo adattivo batte M4: M7 −0,005 [−0,007, −0,002] (+0,008
  negli ultimi 5.000, con rischio 0,054), M7c −0,005 [−0,009, −0,001], M7t −0,001 (scatta nel 25% delle esecuzioni).
- **Corruzioni miste.** M4: rischio 0,053, finestra peggiore 0,107. I metodi adattati spendono molto più rischio:
  M7 0,075 (0,105 nel secondo segmento, con le mappe della prima corruzione), M7c 0,075, M7t 0,061; finestre peggiori
  0,212, 0,245, 0,138. Savings sui 30.000 input contro M4: M7c +0,013 [0,006, 0,020], M7 +0,004, M7t +0,001.
- **Sensibilità** (riferimento `variants_v2` con 4 ripetizioni, quindi confronti più rumorosi che su CIFAR10). π 0,02:
  M4 +0,002 (0,133 contro 0,132), rischio 0,049; π 0,1: −0,011. Guadagno di M7 su M4: +0,014 con π 0,02, +0,013 con
  π 0,1, +0,015 con γ 5e-4, contro +0,013. Il transitorio è già al minimo misurabile (1.000 input) per tutti i γ; i
  savings di M4 con γ da 1e-4 a 1e-3 stanno tra 0,127 e 0,128, e la finestra peggiore sale con γ (da 0,079 a 0,094).
- **Previsioni.** G8 (gli stessi ordinamenti di CIFAR10, differenze dimezzate): sbagliata sui gradini (su CIFAR100
  nessun metodo adattivo batte M4, e l'ordine cambia), giusta a metà sulle corruzioni miste (M7c sempre il primo nei
  savings, +0,013 contro +0,019; ma l'eccesso di rischio è doppio, 0,022 contro 0,010). S1 su CIFAR100: di poco
  sbagliata (+0,0015 contro +0,002–0,005). S2: sbagliata (−0,011 contro −0,005–0,008). S3 e S5 giuste. S4 su CIFAR100:
  il transitorio non si può confrontare (già al minimo); savings giusti (entro ±0,005); finestra peggiore sbagliata,
  sale con γ invece di scendere.
- **Convinzione.** Su CIFAR100 l'adattamento con shift ripetuti non conviene: sui gradini non guadagna, con corruzioni
  che cambiano spende rischio fino a una finestra peggiore di 0,245. È coerente con il ritorno al pulito (C.16): su
  CIFAR100 la soglia del controller quasi non si muove sotto shift, mentre le mappe e la ripartenza la spostano molto,
  e ogni cambio successivo la trova nel posto sbagliato.

## Dopo il Checkpoint 3b

Davide, alle 20:46: "Continua pure, come stanno venendo i risultati?". Lo leggo come via libera alla proposta del
documento (decisione 2): provare una protezione dopo l'adattamento prima della Fase D. M7w resta la fase di
riferimento. Job con 1 worker (voce C.19).

### C.21 (15 nov) Una protezione dopo l'adattamento: regola e previsioni

Scritta alle 20:50 del 15 novembre (ora letta da `date`: 20:48), prima del codice, della calibrazione e dei job.

- **L'idea.** Dopo il fit, un rilevatore senza label confronta la distribuzione recente delle probabilità dell'exit 1
  (con le teste originali, quindi indipendente dalle mappe, calcolata per ogni input) con quella al momento del fit.
  Quando la distanza supera una soglia, toglie le mappe e rimette la soglia di uscita a quella che il controller aveva
  subito prima della ripartenza: da lì il sistema è M4. La fase resta una sola (nessuna riadattazione).
- **Scelta del segnale** (misure di stasera, prima di ogni simulazione; reti del seed 0, severity 3). La confidenza
  media dell'exit 1 si sposta oltre 1,5 volte il 99,9° percentile del rumore (due finestre di 1.000 input dello stesso
  dominio) nel 48–53% dei ritorni al pulito e nel 40–62% dei cambi tra corruzioni. La distanza di variazione totale
  (TV) tra le probabilità medie dell'exit 1 sulle classi lo fa nell'88% e nell'83% dei ritorni e nel 100% dei cambi.
  Uso la TV.
- **Il rilevatore.** Media mobile esponenziale del vettore di probabilità dell'exit 1 (peso 1/1.000 per input,
  partenza dalla media sulle righe di calibrazione pulite); al fit si salva come riferimento; dopo, a ogni input, si
  confronta la TV tra media mobile e riferimento con la soglia κ_d.
- **Regola per κ_d, fissata ora.** Stream solo puliti di 40.000 input, le 8 reti del seed 0, 10 ripetizioni: riferimento
  preso all'input 10.000, rilevatore attivo dall'input 10.001 al 40.000. Per ogni dataset scelgo il κ_d più piccolo
  nella griglia {0,03, 0,04, 0,05, 0,06, 0,08, 0,10, 0,12, 0,15, 0,20} per cui scatta al più nell'1% delle esecuzioni.
  Nessun dato shiftato entra nella scelta.
- **Il metodo** M7wg: M7w più il rilevatore. Job: stream con salto (M4, M7wg; M7w c'è già in `variants_w` con le stesse
  estrazioni), ritorno al pulito e corruzioni miste (M4, M7w, M7wg); 8 reti del seed 0, ripetizioni come prima (6 su
  CIFAR10; su CIFAR100 4 per salto e miste, 6 per il ritorno).
- **Previsioni.**
  - K1. La regola sceglie κ_d tra 0,04 e 0,08 su CIFAR10 e tra 0,08 e 0,15 su CIFAR100.
  - K2. Salto: il rilevatore scatta dopo la fase in al più il 5% delle esecuzioni (il dominio non cambia); il guadagno
    di M7wg negli ultimi 10.000 input sta entro 0,003 da quello di M7w.
  - K3. Ritorno: scatta entro 2.000 input dal ritorno in almeno l'80% delle esecuzioni; rischio di M7wg nei primi 5.000
    input dopo il ritorno al massimo 0,055 su CIFAR10 e 0,060 su CIFAR100 (M7 era a 0,068 e 0,108); savings sui 15.000
    input del ritorno entro 0,01 da M4.
  - K4. Corruzioni miste: rischio di M7wg sui 30.000 input al massimo quello di M4 più 0,005 (M7 era sopra di 0,011 e
    0,022); savings sotto M4 di 0,005–0,025, perché la fase si paga nel primo segmento e il suo guadagno dura solo fino
    al cambio successivo.
- **Codice** (dopo le previsioni). In `fase_c/sim.py`: `exit1_probs` e il rilevatore dentro `simulate` (chiave
  `guard`, attivo solo per M7wg; gli altri metodi non lo vedono). In `fase_c/run.py`: M7wg e `--guard-kappa`,
  `--guard-window`. Calibrazione: `fase_c/guard_calibrate.py`. Job: modo `guard` di `fase_c/run_c06.sh`.
- **Calibrazione** (`fase_c/guard_calibrate.log`, scritto alle 20:49:57). TV massima per esecuzione sugli stream puliti: mediana
  0,045 e massimo 0,054 su CIFAR10, mediana 0,074 e massimo 0,088 su CIFAR100. La regola sceglie **κ_d = 0,06 su
  CIFAR10 e 0,10 su CIFAR100** (nessun falso allarme su 40 esecuzioni per dataset). K1 giusta.
- **Prova** (una rete, gaussian noise 3, ritorno al pulito, 2 ripetizioni; non è un risultato). Fino al ritorno M7wg
  è identico a M7w (stesso fit, stessa ripartenza, nessun allarme durante lo shift); il rilevatore scatta 81 e 106
  input dopo il ritorno; nei primi 5.000 input puliti rischio 0,015 e savings come M4, contro 0,067 di M7w.
- **Job** lanciati alle 20:50:51 (`STATUS.txt`) con 1 worker: CIFAR10 (6 ripetizioni) poi CIFAR100 (6 per il ritorno, 4 per salto e
  miste).

### C.22 (15 nov) La protezione dopo l'adattamento: risultato

- **Esecuzione.** `fase_c/guard_return`, `guard_jump`, `guard_mixed`, dalle 20:50:51 alle 21:42:24 (`STATUS.txt`), 1
  worker, tutti exit 0, batteria dal 77% al 95%, nessun errore I2C nel kernel. M4 identico a quello di `return`,
  `variants_w` e `mixed` (8.640, 6.000, 8.400 righe, differenza 0; `fase_c/guard_same.log`). Letture:
  `verifica_fable/c_guard_read.py` (`fase_c/guard_read.log`), `c_return_sens_read.py return` e `segments`
  (`fase_c/guard_return_read.log`, `guard_mixed_read.log`). Figure: `fase_c/timeline_guard_return.png`,
  `timeline_guard_mixed.png`.
- **Quando scatta il rilevatore.** Salto (il dominio non cambia più dopo la fase): falsi allarmi nell'1,7% delle
  esecuzioni su CIFAR10 e nello 0% su CIFAR100. Ritorno al pulito: scatta nel 99,4% e nel 100% delle esecuzioni, in
  mediana 235 e 382 input dopo il ritorno (90° percentile circa 1.000); un allarme prima del ritorno nello 0,3% su
  CIFAR10. Corruzioni miste: scatta al primo cambio dopo la fase nel 100% delle esecuzioni, in mediana circa 200 input
  dopo.
- **Salto.** Il guadagno resta: negli ultimi 10.000 input M7wg 0,342 contro 0,342 di M7w su CIFAR10 (−0,0006), uguale su
  CIFAR100 (0,148).
- **Ritorno al pulito** (M7w qui è nuovo; si comporta come M7 in C.10 e C.16).

  | | Rischio, primi 1.000 dopo il ritorno | Primi 5.000 | Tutti i 15.000 | Rischio su tutto lo stream | Savings sui 15.000 |
  |---|---|---|---|---|---|
  | CIFAR10 M4 | 0,033 | 0,034 | 0,038 | 0,050 | 0,484 |
  | CIFAR10 M7w | 0,076 | 0,068 | 0,059 | 0,057 | 0,488 |
  | CIFAR10 **M7wg** | 0,044 | 0,037 | 0,040 | 0,050 | 0,486 |
  | CIFAR100 M4 | 0,042 | 0,045 | 0,046 | 0,048 | 0,230 |
  | CIFAR100 M7w | 0,130 | 0,111 | 0,085 | 0,062 | 0,240 |
  | CIFAR100 **M7wg** | 0,068 | 0,050 | 0,048 | 0,048 | 0,233 |

  Resta un picco breve nelle poche centinaia di input prima dell'allarme (0,044 e 0,068 nei primi 1.000), poi il
  sistema è M4. Il transitorio dopo il ritorno passa da 2.900 e 11.450 input (mediana, per M7w) a 1.000 e 1.100.
- **Corruzioni miste.** Sui 30.000 input shiftati il rischio di M7wg è quello di M4: 0,058 contro 0,058 su CIFAR10,
  0,053 contro 0,053 su CIFAR100 (M7w 0,067 e 0,075); finestra peggiore 0,146 e 0,114 contro 0,143 e 0,107 (M7w 0,173 e
  0,218). Il prezzo: con una fase sola, al primo cambio si perde anche il guadagno, e la fase pagata nel primo segmento
  non torna: savings sotto M4 di 0,019 [0,013, 0,026] su CIFAR10 e di 0,005 [0,003, 0,007] su CIFAR100.
- **Previsioni di C.21.** K1 giusta (0,06 e 0,10). K2 giusta (1,7% e 0%; −0,0006). K3 giusta (scatta nel 99–100%, in
  mediana entro 400 input; rischio nei primi 5.000 0,037 e 0,050; savings entro 0,003 da M4). K4 giusta sul rischio;
  sui savings giusta su CIFAR10 (−0,019) e appena fuori su CIFAR100 (−0,0047 contro almeno −0,005).
- **Convinzione.** Un rilevatore senza label sulla distribuzione delle probabilità dell'exit 1 (soglia scelta solo sul
  pulito) toglie il problema del cambio successivo: il sistema adattato torna sicuro come il controller da solo entro
  poche centinaia di input, e quando il dominio resta non costa quasi nulla. Il limite che resta è la fase singola:
  con shift ripetuti si paga la fase e si perde il guadagno al cambio successivo. Il passo seguente naturale è una fase
  che si riattiva dopo l'allarme, che però si ripaga solo se ogni dominio dura più del payback (qualche decina di
  migliaia di input).

## Fase D del piano v3: requisiti per pubblicare (dal 15 novembre sera)

Davide, in chat: "Passa a fase D". Poi ha approvato il download di CIFAR-10.1 (domanda con nome dei file, fonte e
dimensione); i file sono delle 22:16. Colab e job oltre l'ora solo con il suo ok, dopo un piano con le stime.

### D.01 (15 nov) Metodi congelati e previsioni di conferma

Scritta alle 22:20 del 15 novembre (ora letta da `date`: 22:17), prima di calcolare qualunque cosa sulle 4 corruzioni
extra e su CIFAR-10.1. Dei file di CIFAR-10.1 ho controllato solo forma, tipo e bilanciamento delle classi.

- **Dati di conferma** (sezione 6.1 del piano). Le 4 corruzioni extra di CIFAR-C (speckle noise, gaussian blur,
  spatter, saturate), con i logit già esportati per le 24 reti e mai usati. CIFAR-10.1 v6 (Recht et al. 2018), 2.000
  immagini, scaricato alle 22:16 in `fase_d/data/` (sha256 dei dati `2997188e…`, delle label `ae40beda…`).
- **Metodi congelati** (nessun parametro si tocca da qui): ε = 0,05, δ = 0,1, π = 0,05, γ = 2e-4, ripartenza con un
  quarto delle righe tenuto da parte e test a due errori standard, fase con audit al 25% fino a 3.000 righe di
  training, fit pesato (M7w), rilevatore con κ_d = 0,06 su CIFAR10 e 0,10 su CIFAR100 (M7wg). Stessi stream e stesse
  finestre della Fase C.
- **Disegno, 4 corruzioni extra.** Stream con salto sulle 24 reti, severity 3 e 5, 10 ripetizioni: M1, M2, M4, M7w,
  M8c. Ritorno al pulito sulle 8 reti del seed 0, severity 3, 6 ripetizioni: M4, M7w, M7wg. Letture con gli stessi
  lettori della Fase C.
- **Disegno, CIFAR-10.1** (solo le 12 reti CIFAR10). Logit calcolati con lo stesso `predict` dell'export, salvati in
  `fase_d/`. Offline: M1 su tutte le 2.000 immagini; M2 con la UCB su una metà e valutato sull'altra, nei due versi,
  confrontato con M2 pulito calibrato con lo stesso numero di righe; mappe fittate su una metà e valutate sull'altra
  alla soglia oracolo empirica. Online: stream di 10.000 input puliti e 30.000 di CIFAR-10.1 (pescati con ripetizione
  da 1.000 immagini; le altre 1.000 servono a calibrare M2), metodi M1, M2, M4. Niente fase di adattamento online su
  CIFAR-10.1: con 1.000 immagini la fase rivedrebbe le stesse immagini su cui è fittata.
- **Previsioni sulle 4 corruzioni extra.**
  - E1. RQ2: |M4 − M2| ≤ 0,010 nei savings netti degli ultimi 10.000 input, in tutte e quattro le celle (dataset ×
    severity).
  - E2. M1 negli ultimi 10.000 input shiftati ha rischio sopra ε + 0,01 su CIFAR10 alle due severity e sopra ε su
    CIFAR100 a severity 5.
  - E3. M7w − M4 > 0 con l'intervallo al 95% sopra zero in tutte e quattro le celle; frazione recuperata tra l'8% e il
    25%.
  - E4. M8c − M7w tra −0,005 e +0,010.
  - E5. Ritorno: rischio di M7wg sull'intero stream entro ±0,003 da M4, quello di M7w sopra M4 di almeno 0,004 su
    entrambi i dataset; il rilevatore scatta dopo il ritorno in almeno il 90% delle esecuzioni.
  - E6. Transitorio di M4 su CIFAR10 a severity 3 (mediana) tra 2.000 e 9.000 input.
- **Previsioni su CIFAR-10.1.**
  - N1. L'accuracy dell'exit 9 scende di 4–10 punti rispetto al test di CIFAR-10.
  - N2. M1 (soglia pulita) su CIFAR-10.1: rischio tra 0,045 e 0,070.
  - N3. I savings certificati di M2 su CIFAR-10.1 sono il 70–95% di quelli di M2 sul pulito con lo stesso numero di
    righe di calibrazione.
  - N4. Online: M4 entro ±0,010 da M2 negli ultimi 10.000 input; transitorio (mediana) al più 3.000 input.
  - N5. Mappe offline (fittate su 1.000 immagini, valutate sulle altre 1.000 alla soglia oracolo empirica): guadagno
    tra −0,005 e +0,020.
- **Soglia d'allarme.** Se E1 o E3 falliscono, i risultati principali della Fase C non si estendono oltre le corruzioni
  su cui li ho esplorati, e lo scrivo così.

### D.02 (15 nov) Latenza reale a batch 1: previsioni

Scritta alle 22:24 del 15 novembre (ora letta da `date`: 22:22), prima di misurare.

- **Disegno** (`fase_d/latency.py`, da lanciare a Mac fermo). Una rete per dataset (joint, seed 0), batch 1, CPU con un
  thread e MPS. Tempo per input (mediana su 1.000, dopo 100 di riscaldamento) della ResNet20 senza teste (il costo di
  M0) e dell'inferenza early exit fermata a ogni exit: blocchi, teste incontrate (solo il classificatore), max
  softmax e confronto con la soglia sul host (su MPS è una sincronizzazione per exit). Poi le distribuzioni degli exit
  della soglia pulita sul pulito e di M2 a severity 3 (media sulle 15 corruzioni) tradotte in tempo.
- **Previsioni.**
  - L1. CPU: il tempo per uscire a un exit cresce con i MAC; i savings in tempo sono tra il 60% e il 90% di quelli in
    MAC, per il costo fisso delle teste e dei controlli.
  - L2. MPS a batch 1: dominano il lancio dei kernel e la sincronizzazione a ogni exit; i savings in tempo sono al più
    il 40% di quelli in MAC, e possono essere negativi (l'early exit più lento della rete intera).
  - L3. Arrivare all'exit 9 con tutte le teste costa più di M0: tra il 5% e il 30% in più su CPU.

### D.03 (15 nov) La conferma: 4 corruzioni extra e CIFAR-10.1

- **Esecuzione** (`fase_d/run_d01.sh`, `fase_d/STATUS.txt`, 1 worker). `confirm_jump` dalle 22:18:49 alle 22:51:50
  (192 casi); `confirm_return` dalle 22:51:50 alle 22:57:50 (CIFAR10 con κ_d 0,06, CIFAR100 con 0,10); CIFAR-10.1:
  export dei logit (`fase_d/export_cifar101.log`), lettura offline (`fase_d/cifar101_offline.log`) e stream online
  `cifar101_online` dalle 22:58:28 alle 22:58:49; tutti exit 0. Letture: `fase_d/confirm_jump_read.log` (c_read.py
  con `ADAPT=M7w`, lettore cambiato per accettare il metodo adattato da una variabile, M7 di default),
  `confirm_return_read.log`, `cifar101_online_read.log`.
- **4 corruzioni extra** (24 reti, 10 ripetizioni; ultimi 10.000 input).
  - RQ2, M4 − M2: −0,006 [−0,016, +0,002] e −0,001 su CIFAR10 a severity 3 e 5, +0,004 [+0,000, +0,007] e +0,005 su
    CIFAR100. Transitorio di M4 su CIFAR10 a severity 3: 5.900 input (mediana).
  - M1 dopo lo shift: rischio 0,078 e 0,140 su CIFAR10, 0,052 e 0,087 su CIFAR100.
  - RQ3, M7w − M4: +0,027 [0,011, 0,044] e +0,041 [0,015, 0,068] su CIFAR10, +0,014 [0,010, 0,018] e +0,017 [0,012,
    0,023] su CIFAR100; frazione recuperata 16,7%, 16,4%, 16,3%, 13,1%. M8c − M7w da −0,004 a −0,001. Payback
    mediano oltre lo stream, 23.800, 22.100, 23.250 input (ripagato nel 35%, 54%, 65%, 69% dei casi); netto sullo
    stream −0,011, +0,007, +0,003, +0,005.
  - Garanzia: bound sempre rispettato; rischio sui 40.000 input sotto ε + 0,01 nel 79% (M4) e 87% (M7w) delle
    ripetizioni su CIFAR10, 90% e 98% su CIFAR100.
  - Ritorno al pulito (reti del seed 0): rischio sull'intero stream M4 0,050, M7w 0,055, M7wg 0,049 su CIFAR10; 0,049,
    0,059, 0,048 su CIFAR100. Il rilevatore scatta dopo il ritorno nel 79% delle esecuzioni su CIFAR10 (mediana 226
    input) e nel 100% su CIFAR100 (357); mai prima del ritorno.
- **CIFAR-10.1** (12 reti CIFAR10).
  - Accuracy dell'exit 9: da 8,4 a 10,9 punti in meno (media 9,5).
  - Offline: M1 rischio 0,066 (pulito 0,045); M2 con 1.000 righe 0,376 contro 0,506 sul pulito (rapporto 0,742, da
    0,668 a 0,786 per rete); mappe sui logit alla soglia oracolo +0,000 (da −0,010 a +0,016; sul pulito +0,001).
  - Online: M1 a rischio 0,063; M4 0,389 contro M2 0,377 negli ultimi 10.000 input (+0,011, da −0,014 a +0,039 per
    rete), rischio 0,050; transitorio 4.150 input.
- **Previsioni di D.01.** E1 giusta. E2 giusta. E3 giusta (13–17%). E4 giusta. E5 giusta sul rischio, sbagliata sulla
  rilevazione su CIFAR10 (79% invece di almeno 90%: nei casi senza allarme il cambio, e il picco, sono piccoli). E6
  giusta. N1 giusta in media (9,5), 3 reti su 12 oltre i 10 punti. N2 giusta in media (0,066). N3 giusta in media
  (74%), 2 reti sotto il 70%. N4 sbagliata di poco su entrambi i punti (+0,011 invece di ±0,010, a favore del
  controller perché M2 ha solo 1.000 righe; transitorio 4.150 invece di al più 3.000). N5 giusta, ma il risultato è
  nullo.
- **Convinzione.** I risultati della Fase C si confermano sulle corruzioni mai viste: il controller vale la soglia
  oracolo, la fase pesata recupera il 13–17% del calo quanto le mappe oracolo, la protezione rende sicuro il ritorno.
  Lo shift naturale conferma il controller ma non l'adattamento: su CIFAR-10.1 le mappe sui logit non recuperano
  nulla.

### D.04 (15 nov) Latenza reale a batch 1: risultato

- **Esecuzione.** `fase_d/latency.py` alle 23:00 (log scritto alle 23:00:40), Mac senza altri job. Log:
  `fase_d/latency.log`.
- **Risultato** (tempo per input, mediana su 1.000).
  - CPU con un thread: ResNet20 senza teste 1,20 ms su CIFAR10 e 1,05 su CIFAR100. Per uscire all'exit 1 serve il
    28% e il 19% di quel tempo, contro il 13% dei MAC; all'exit 3 il 75% e il 72%, contro il 36%. I primi tre blocchi
    lavorano a 32×32 con 16 canali: sono limitati dalla memoria e costano tre quarti del tempo con un terzo dei MAC.
    Arrivare all'exit 9 con tutte le teste costa 1,13 e 1,17 volte M0.
  - Savings in tempo contro savings in MAC, CPU: soglia pulita sul pulito 0,326 contro 0,575 su CIFAR10 (57%) e 0,034
    contro 0,245 su CIFAR100 (14%); M2 a severity 3, 0,126 contro 0,354 (36%) e −0,058 contro 0,132 (più lento di M0).
  - MPS: ogni exit costa circa 0,4 ms (lancio dei kernel e sincronizzazione per il confronto sul host); all'exit 9
    con tutte le teste 2,3 volte M0. Savings in tempo da 0,001 a −0,97: su MPS a batch 1 l'early exit è sempre più
    lento della rete intera.
- **Previsioni di D.02.** L1 sbagliata: su CPU i savings in tempo sono il 14–57% di quelli in MAC, non il 60–90%, e
  negativi su CIFAR100 sotto shift. L2 giusta (nulli o negativi). L3 giusta (13% e 17% in più).
- **Convinzione.** Con ResNet20 su immagini 32×32 a batch 1 i savings in MAC si traducono poco o niente in tempo:
  il costo non sta dove stanno i MAC. È il punto che un revisore solleverebbe. Rimedi da provare: compilare il modello
  (torch.compile o CoreML) per tagliare il costo per layer, valutare solo alcune teste, e soprattutto la scala
  ImageNet, dove a batch 1 il calcolo domina e i MAC pesano di più. Limiti della misura: una rete per dataset,
  PyTorch in modalità eager, CPU con un thread.

### D.05 (15 nov) ImageNet nella versione minima: disegno e previsioni

Scritta alle 23:09 del 15 novembre (ora letta da `date`: 23:07). Davide ha scelto l'opzione 1 del piano Colab
("1"), e ha detto che nei testi per il professore il trattino si tiene nei nomi e dove serve.

- **Modello.** ResNet50 di torchvision con i pesi ImageNet `IMAGENET1K_V1` (il riferimento classico per ImageNet-C),
  congelata. 8 exit dopo i blocchi bottleneck 2, 4, 6, 8, 10, 12, 14 e 15 (su 16), più la fc finale: 9 exit come le
  reti CIFAR, così simulatore e lettori restano gli stessi. Teste: global average pooling e una lineare verso 1.000
  classi (sulle 4×4 feature delle pooled head di CIFAR le dimensioni a 224 pixel sarebbero troppo grandi). Regime
  disjoint: si allenano solo le teste, su feature in cache.
- **Dati** (li scarica Davide; nessun ImageNet train). ImageNet val divisa una volta (permutazione con seed 0): 25.000
  immagini per allenare le teste (22.500 training, 2.500 per scegliere solo il numero di epoche), 25.000 per valutare,
  che fanno da dominio pulito. Shift: ImageNet-V2 matched frequency (10.000), ImageNet-R (30.000, 200 classi; le
  label si confrontano sulle 200 classi), e 4 corruzioni di ImageNet-C a severity 3, una per famiglia (gaussian noise,
  defocus blur, fog, jpeg compression), generate sulle 25.000 immagini di valutazione con il codice ufficiale
  (`imagecorruptions`), dopo resize a 256 e crop centrale a 224 come ImageNet-C. Generarle evita i 66 GB di ImageNet-C;
  è una scelta da dichiarare.
- **Su Colab** (lo lancia Davide): feature, teste, logit (N, 9, 1.000) in float16 per tutti i domini; da riportare in
  locale. **In locale**: le stesse analisi di CIFAR (M1, M2, controller, fase pesata, protezione) e la latenza di
  ResNet50 con le teste.
- **Previsioni sulla latenza di ResNet50** (misurabile subito in locale, con pesi casuali: il tempo non dipende dai
  valori).
  - L4. CPU, un thread, batch 1: per ogni exit la frazione di tempo sta entro ±0,08 dalla frazione di MAC (a 224
    pixel il calcolo domina); arrivare all'exit 9 con tutte le teste costa al più il 5% in più della rete senza teste.
  - L5. MPS, batch 1: entro ±0,15 dalla frazione di MAC dal secondo exit in poi; all'exit 9 con tutte le teste tra il
    5% e il 30% in più (8 sincronizzazioni contro una forward di qualche millisecondo).
- **Previsioni sui dati ImageNet** (prima del run su Colab).
  - I1. Accuracy delle teste sulla metà di valutazione: exit 1 tra il 5% e il 25%, exit 8 tra il 60% e il 75%, fc
    finale 75–77%.
  - I2. M1 sul pulito: savings tra 0,15 e 0,40 (le prime teste lineari su feature globali sono deboli).
  - I3. Sotto shift M1 supera ε su ImageNet-R e sulle corruzioni a severity 3; su ImageNet-V2 di poco o niente.
  - I4. Il controller arriva entro ±0,01 dalla soglia oracolo, come su CIFAR (RQ2).

### D.06 (15 nov) ResNet50: latenza a batch 1 e il pacchetto per Colab

- **Latenza** (`fase_d/imagenet/latency_resnet50.py`, log scritto alle 23:08:47, Mac senza altri job; pesi casuali).
  ResNet50 a 224 pixel: 4,09 GMAC. Frazioni di MAC per uscire agli exit 1–9: 0,139, 0,283, 0,390, 0,535, 0,642,
  0,749, 0,894, 0,948, 1,002.
  - CPU, un thread: rete senza teste 20,0 ms; frazioni di tempo 0,309, 0,454, 0,521, 0,639, 0,725, 0,810, 0,929,
    0,993, 1,046. Scarto dai MAC da +0,17 sui primi exit a +0,04 sugli ultimi (su ResNet20 fino a +0,39);
    all'exit 9 con tutte le teste +4,6%.
  - MPS: rete senza teste 5,84 ms; frazioni di tempo da 0,194 a 1,506; scarto da +0,06 a +0,18 sui primi sei exit,
    poi +0,32, +0,42, +0,50: le sincronizzazioni per il confronto sul host si sommano.
  - **Previsioni.** L4 giusta sugli exit dal 5° in poi e sul costo all'exit 9 (+4,6%), sbagliata sui primi quattro
    (+0,10–0,17 invece di al più 0,08). L5 giusta sugli exit 2–4, sbagliata dal 5° in poi e sul costo all'exit 9
    (+51% invece di 5–30%).
  - **Convinzione.** A 224 pixel il tempo segue i MAC molto meglio che a 32 pixel, ma i primi blocchi restano più
    cari dei loro MAC, e su MPS il confronto sul host a ogni exit va tolto (decisione sulla GPU, non sul host).
- **Pacchetto per Colab** (`fase_d/imagenet/`): `exits.py` (modello e frazioni di MAC), `colab_imagenet.py`
  (feature, teste, logit; fasi che riprendono da sole), `README_colab.md` (file da scaricare, celle, stime). Provato
  in locale da capo a fondo su 48 immagini casuali con una corruzione finta (`imagecorruptions` non è installato sul
  Mac; la chiamata vera gira su Colab): feature (48, 8.448), 8 teste, logit (48, 9, 1.000) per i 7 domini.
  Stima: circa 10 GB di dati da scaricare per Davide, 20–40 minuti su Colab.

### D.07 (16 nov) ImageNet: il pacchetto semplificato

Scritta alle 10:24 del 16 novembre (ora letta da `date`: 10:22). Davide ha chiesto di rendergli il lavoro il più
semplice possibile.

- **Cosa cambia.** Un solo notebook, `fase_d/imagenet/imagenet_minimo.ipynb` (generato da `make_notebook.py`): scarica
  i dati direttamente su Colab (ImageNet val dal dataset Hugging Face `ILSVRC/imagenet-1k` con il token di Davide;
  ImageNet-V2 e ImageNet-R dalle fonti pubbliche, saltati senza fermarsi se non si scaricano), scrive il codice,
  calcola e fa scaricare al browser uno zip di pochi MB. Sul Mac non c'è Google Drive per desktop, quindi niente Drive.
- **Cosa torna.** Non più i logit (circa 3 GB) ma tabelle compatte per dominio (max softmax degli exit 1–8, predizione
  di ogni exit, label), che bastano al simulatore senza adattamento; il test offline delle mappe sui logit gira su
  Colab (fittate su metà di ogni dominio shiftato, con tutte le righe e con 3.000, savings alla soglia oracolo sull'altra
  metà) e torna come numeri. Lo script legge ImageNet val dalla cartella (id della classe nel nome del file o nella
  sottocartella), senza devkit.
- **Prove in locale.** Lo script da capo a fondo su 32 e 48 immagini casuali (con una corruzione finta); la lettura di
  val nei due formati su file finti (classi nell'ordine degli id ordinati, come torchvision); tutte le celle del
  notebook sono Python valido. Non provati: i download su Colab e la chiamata a `imagecorruptions`.
- **Corretto** prima della consegna: la soglia oracolo restituiva 0 quando solo λ = 1 rispetta il rischio; ora
  restituisce il savings vero dell'exit 9 (−0,002).

### D.08 (16 nov) ImageNet minimo: risultato

Scritta alle 11:23 del 16 novembre (ora letta da `date`: 11:21).

- **Esecuzione** (Davide, su Colab con la T4; file del run tra le 09:50 e le 10:17, zip scaricato alle 11:17).
  Teste, pulito, ImageNet-V2, ImageNet-R, gaussian noise e defocus blur completati; **si è fermato a fog**:
  `imagecorruptions` usa `np.float_`, tolto in NumPy 2. Lo zip è stato fatto lo stesso (un comando `!` che fallisce non
  ferma il notebook). Corretto lo script (`np.float_ = np.float64` se manca) e fissato il seed delle teste; notebook
  rigenerato. Mancano fog e jpeg compression. Risultati in `fase_d/imagenet/out/` (copiati dallo zip), lettura in
  `fase_d/imagenet/imagenet_offline.py` (`imagenet_offline.log`).
- **Teste.** Accuracy sulla metà di valutazione pulita, exit 1–9: 0,023, 0,048, 0,051, 0,122, 0,135, 0,154, 0,521,
  0,608, 0,759. Diverse teste hanno il massimo sul held-out alle ultime epoche (28–30 su 30): un po' sotto-allenate.
- **Savings** (metà per calibrare, metà per valutare; ε 0,05).
  - Pulito: M1 = M2 0,032 (rischio 0,043), oracolo 0,034. Gli exit 1–6 non escono mai; 19% all'exit 7, 24% all'8,
    57% al 9.
  - Sotto shift: M2 0,021 (V2), 0,007 (R), 0,006 (gaussian noise), 0,009 (defocus blur). M1 resta **sotto ε**
    (rischio 0,008–0,040): con lo shift le teste tardive sono meno sicure ed escono meno input.
  - In tempo, con le frazioni misurate in D.06: savings negativi su CPU (da −0,011 a −0,038) e su MPS (circa −0,45):
    con queste teste l'early exit è più lento della rete intera.
  - Mappe sui logit (su Colab, alla soglia oracolo): da +0,009 a +0,012 (fittate su metà del dominio), poco meno con
    3.000 righe.
- **Previsioni di D.05.** I1 sbagliata sull'exit 1 (0,023 invece di 0,05–0,25), giusta sull'exit 8 (0,608) e sulla fc
  (0,759). I2 sbagliata di un ordine di grandezza (0,032 invece di 0,15–0,40). I3 sbagliata: M1 non supera ε da
  nessuna parte. I4 non provata: con savings così piccoli il confronto tra controller e oracolo non direbbe nulla.
- **Convinzione.** Nella versione minima (teste lineari su feature globali, allenate su 22.500 immagini) gli early exit
  di ResNet50 su ImageNet non risparmiano quasi nulla, quindi non c'è un calo sotto shift da misurare. Il motivo è
  doppio: le prime teste sono deboli (poche immagini e feature povere) e gli ultimi tre blocchi, gli unici con teste
  utili, pesano solo il 10% dei MAC. Per una scala ImageNet che dica qualcosa servono teste più forti: allenate su molte
  più immagini (una parte di ImageNet train) o più ricche, oppure un modello multi-exit già allenato per ImageNet.
  Decide Davide.

### D.09 (16 nov) ImageNet con teste allenate su ImageNet train: disegno e previsioni

Scritta alle 11:26 del 16 novembre (ora letta da `date`: 11:24). Davide ha scelto l'opzione (a) ("A").

- **Cosa cambia rispetto a D.05.** Le teste si allenano su un blocco di ImageNet train scaricato dentro Colab (il primo
  archivio di `ILSVRC/imagenet-1k`, al più 200.000 immagini; 10.000 tenute da parte per scegliere il numero di epoche,
  al più 15). Tutto il resto uguale: ResNet50 `IMAGENET1K_V1` congelata, teste lineari su global average pooling dopo
  i blocchi 2, 4, 6, 8, 10, 12, 14, 15; valutazione sulla stessa metà di ImageNet val (25.000 immagini) e sugli stessi
  shift, più fog e jpeg compression. Seed fissato per le teste. Se il blocco non copre almeno 990 classi (cioè se gli
  archivi sono ordinati per classe) il notebook si ferma prima di allenare.
- **Previsioni** (sulla metà di valutazione; confronti con D.08).
  - J1. Accuracy delle teste: exit 4 (blocco 8) da 0,12 a 0,20–0,35; exit 6 (blocco 12) da 0,15 a 0,30–0,50; exit 8
    (blocco 15) da 0,61 a 0,65–0,72; exit 1 da 0,02 a 0,03–0,08.
  - J2. Savings di M2 sul pulito da 0,032 a 0,06–0,15.
  - J3. Sotto shift M2 scende al 20–60% del pulito su ImageNet-R e sulle corruzioni, al 60–90% su ImageNet-V2.
  - J4. Con teste migliori la soglia pulita esce più spesso presto, e sotto shift il suo rischio supera ε su almeno uno
    tra ImageNet-R e le corruzioni (resta sotto su ImageNet-V2).
  - J5. In tempo, su CPU, M2 sul pulito ha savings positivi, tra il 40% e l'80% di quelli in MAC.
  - J6. Le mappe sui logit, alla soglia oracolo, guadagnano almeno +0,01 su ogni corruzione.
- **Pacchetto** (pronto alle 11:27). `colab_imagenet.py` con `--train` e `--max-train`: lettura delle immagini con l'id
  WordNet preso dalla sottocartella o dal nome del file, ovunque sia; feature in un array float16 preallocato e, per
  allenare, sulla GPU; autocast float16 per feature e logit; epoche al più 15; STOP se il blocco copre meno di 990
  classi. Notebook rigenerato: nuova cella che scarica il primo archivio di ImageNet train ed estrae fino a 200.000
  immagini (nomi appiattiti, filtro `data` di tarfile), zip con i pesi delle teste. Provato in locale su immagini
  casuali (script intero con un blocco di train finto) e sulla lettura dei nomi di train nei due formati (classi
  sconosciute scartate, mappa delle classi presa da val, sottoinsieme casuale). Non provati: l'archivio vero di train
  su Hugging Face (formato dei nomi e copertura delle classi) e i tempi su Colab.
