# Elenco dei piloti della Parte 3 (per il Checkpoint 2)

Scritto da Fable il 13 novembre 2026, prima di lanciare qualunque pilota. Ogni pilota ha una domanda, un'ipotesi
falsificabile con la mia previsione, baseline e limite superiore, il criterio di kill e l'obiezione del revisore 2.
L'esecuzione andrà ad agenti Opus con queste specifiche; io verifico i risultati.

**Regole comuni** (sezione 5 del piano).
- Esplorazione solo sulle 15 corruzioni standard e sulle 24 reti esistenti. Le 4 corruzioni extra e CIFAR-10.1 restano
  chiuse fino alla conferma.
- Rumore di fondo misurato nella Parte 1: deviazione tra seed 0,002–0,008 (0,034 per CIFAR10 joint_sd), tra split 0,009.
  **Un effetto sotto 0,02 di savings, o sotto 2 deviazioni tra seed, è nullo.**
- Protocollo: split adapt 4000, cal 3000, eval 3000; UCB con δ = 0,1; ε = 0,05; score max softmax se non detto
  altrimenti; confronti appaiati sugli stessi split; risultati anche per famiglia di corruzione.
- Ogni pilota dura meno di un'ora sul Mac con al massimo 3 processi. Niente Colab.

---

## La tesi che i piloti devono confermare o smentire

La Parte 1 e la Parte 2 hanno cambiato la tesi candidata del piano. La versione attuale:

> Sotto shift i savings certificati senza label si dimezzano. Una parte di questo calo non è capacità persa dalla rete:
> è il prezzo di certificare l'accordo con un riferimento che si degrada. Quantifichiamo quel prezzo, mostriamo dove
> sta il margine (non nella ricalibrazione, che costa poco), e giudichiamo ogni fase di adattamento con due metri: il
> compute netto e la frontiera accuracy contro compute. Adattare le teste verso un riferimento degradato recupera
> savings certificati, ma in parte copiando gli errori del riferimento e senza ripagare il proprio costo.

**Cosa la letteratura ha già detto** (Parte 2, da non presentare come nostro):
- che il disaccordo con il riferimento non misura la qualità del task, e che sotto shift il modello completo può essere
  peggiore dell'uscita anticipata: la submission 22466 lo scrive e lo mostra su un caso (pixelation, errore da 0,477 a
  0,496 quando si passa al riferimento);
- che senza label il disaccordo identifica solo il supporto e non il segno del guadagno: submission 22325, con
  l'intervallo stretto. Il nostro lemma "il consistency risk è il miglior certificato senza label" ne è un caso;
- che l'accuracy persa si può controllare con poche label più le pseudo label del modello: Einbinder et al. 2024, su un
  early exit di un LLM, con dati i.i.d.;
- che sotto corruzione un oracolo sugli exit batte l'ultimo exit di 10 punti e le regole pratiche no: Mehra et al. 2022;
- che il Brier consistency risk di Fast yet Safe usa già una label campionata dalla distribuzione del riferimento.

**Cosa non ho trovato** (al 14 novembre 2026, con le ricerche descritte in `02_letteratura.md`):
- una misura dei savings di un early exit con risk control senza label sotto shift, a garanzia mantenuta;
- la scomposizione in tre casi del consistency risk alla soglia certificata, sotto shift, e il suo effetto sui savings;
- il bilancio in compute di una fase di adattamento dentro un risk control (costo, payback, netto);
- l'osservazione che distillare le teste verso il riferimento degradato compra savings certificati copiandone gli
  errori. È la nostra idea candidata più originale, e va verificata nel pilota D.

## Il metro comune: la frontiera accuracy contro compute

La Parte 2 mostra che "savings a consistency risk fissato" non basta come metro quando il riferimento cambia (per
esempio dopo una BN adaptation). Ogni pilota riporta quindi anche:
- la curva accuracy del sistema contro compute medio al variare di λ (le label servono solo per valutare);
- su quella curva, i punti scelti da ogni certificato: M1, M2, i rischi senza label del pilota A, l'accuracy persa
  con label;
- per i metodi di adattamento, lo spostamento della curva e il compute speso per ottenerlo.

Un certificato è buono se sceglie un punto vicino al ginocchio della curva. Un adattamento è buono se sposta la curva
verso l'alto più di quanto costa.

---

## Pilota A. Quale rischio certificare

- **Domanda.** Quanta parte del calo dei savings sotto shift è prezzo del certificato senza label e quanta è perdita
  reale di accuracy?
- **Le loss**, tutte lungo la stessa griglia di λ e con la stessa UCB:
  - ℓ_c = 1[ŷ_exit ≠ ŷ_9], il consistency risk attuale;
  - ℓ_w = 1[ŷ_exit ≠ ŷ_9] · p_9(ŷ_9), il disaccordo pesato con la confidence del riferimento (senza label);
  - ℓ_s = p_9(ŷ_9) − p_9(ŷ_exit), cioè la costruzione di Fast yet Safe con la label campionata dal riferimento,
    applicata alla loss 0-1 (senza label, vale 0 se concordano, sempre ≤ ℓ_w ≤ ℓ_c);
  - il Brier consistency risk di Fast yet Safe (senza label);
  - ℓ_h = 1[ŷ_exit ≠ y, ŷ_9 = y], il danno: sbaglia solo l'exit. È il rischio controllato da Einbinder et al. 2024
    (con label; binario, quindi vale anche CRC);
  - ℓ_g = 1[ŷ_exit ≠ y] − 1[ŷ_9 ≠ y], l'accuracy persa con segno (con label), assoluta e relativa all'accuracy del
    riferimento. Vale in [−1, 1]: si riscala in [0, 1] per la UCB; CRC non si applica.
- **Cosa misurare**, su 24 reti, pulito e severity 1, 3, 5:
  1. i savings certificati con ogni loss, e la quota rimasta sotto shift;
  2. alla λ certificata con ℓ_w e ℓ_s, l'accuracy persa realizzata sulle righe di eval: il certificato senza label è
     conservativo?
  3. lungo tutta la griglia, in quanti punti (dominio, λ) la media di ℓ_s sta sopra l'accuracy persa vera;
  4. la curva delle label: savings certificati sul danno ℓ_h con m = 50, 100, 250, 500, 1000 label più 3000 forward
     complete, con il metodo semi-supervised di Einbinder et al. 2024 (pseudo label del riferimento, correzione sulle
     righe con label). È un metodo della letteratura applicato a un setting nuovo, non un nostro contributo.
- **Ipotesi A1.** Con l'accuracy persa relativa (5% dell'accuracy del riferimento) la quota di savings rimasta a severity
  3 è almeno 15 punti più alta che con il consistency risk (0,59 e 0,53) su entrambi i dataset.
  *Previsione:* 0,75–0,90 su CIFAR10, sopra 0,90 su CIFAR100.
- **Ipotesi A2.** ℓ_s è conservativa (accuracy persa realizzata ≤ ε in almeno il 90% dei casi, per ogni famiglia) e più
  stretta di ℓ_c: a severity 3 certifica almeno 0,03 di savings in più.
  *Previsione:* conservativa su CIFAR100 ovunque; su CIFAR10 pulito al limite. Guadagno 0,05–0,10 su CIFAR10 sotto shift.
- **Baseline e limiti.** Baseline: consistency risk. Letteratura: Brier consistency di Fast yet Safe e il controllo
  semi-supervised di Einbinder et al. 2024. Limite superiore: danno e accuracy persa calibrati con tutte le 3000
  label, e l'oracolo per input.
- **Kill.** Se con l'accuracy persa relativa la quota rimasta supera quella del consistency risk di meno di 5 punti, la
  direzione A si riduce a una nota di reporting. Se ℓ_s non è conservativa in almeno il 90% dei casi sotto shift, resta
  solo come diagnostica e la via pratica diventa quella con poche label.
- **Revisore 2.** "L'accuracy persa richiede label, e la calibrazione del riferimento sotto shift è nota per rompersi."
  Risposta: senza label non c'è alternativa al disaccordo (submission 22325); la rottura va nella direzione
  dell'overconfidence, che rende ℓ_s conservativa, e la misuriamo per famiglia; la curva delle label dice quante ne
  servono. Seconda obiezione: "la 22466 dice già che il disaccordo non è qualità del task." Risposta: lo dice come
  avvertenza su un caso; noi misuriamo quanto compute costa, su 24 reti e 75 domini, e cosa cambia con l'adattamento.
- **Costo.** Solo logit e label esistenti, circa 20 minuti con 3 processi.

## Pilota B. Il bilancio completo dei savings

- **Domanda.** Dove si perde il margine tra ciò che la rete potrebbe risparmiare e ciò che M2 certifica?
- **I livelli**, per dataset, regime e severity (oracoli scelti su metà delle righe e misurati sull'altra):
  1. M2 sul pulito;
  2. oracolo per input sul dominio shiftato (primo exit che concorda con il riferimento), e la sua versione con budget ε;
  3. oracolo con soglie per exit (le 8 soglie migliori a rischio ≤ ε);
  4. oracolo di soglia condivisa;
  5. M2 con n_cal = 250, 1000, 3000 (UCB) e CRC;
  6. M2 al netto delle forward di calibrazione su orizzonti di 10.000 e 100.000 input;
  7. accanto, la linea del pilota A: savings a parità di accuracy persa.
- **Ipotesi B1.** Il salto più grande è tra l'oracolo per input e l'oracolo di soglia condivisa (lo score non riconosce
  gli input che concordano): almeno il 60% del margine totale a severity 3. *Previsione:* 0,30–0,40 di savings su
  CIFAR10 a severity 5, contro 0,015 di prezzo statistico.
- **Kill.** Nessuno: è una misura. È la figura centrale candidata.
- **Revisore 2.** "L'oracolo per input è irraggiungibile." Risposta: per questo c'è il livello 3, che è raggiungibile da
  una regola a 8 soglie, e nel pilota C un limite appreso.
- **Costo.** Circa 30 minuti.

## Pilota C. Il meccanismo: traffic shift contro reliability shift

- **Domanda.** Sotto shift i savings calano perché meno input superano la soglia (traffico) o perché a pari score gli
  input concordano meno (affidabilità)?
- **Misure.**
  - La curva di affidabilità r(l, s) = P(disaccordo | exit l, score s), stimata sul pulito e sullo shiftato.
  - Il controfattuale esatto nei due ordini: score shiftati con affidabilità pulita, e viceversa; per ciascuno la λ
    che riporta il rischio a ε e i savings che ne seguono.
  - AUROC dello score nel prevedere l'accordo, per exit e severity.
  - Un limite appreso: un classificatore dell'accordo allenato sulle righe di adapt del dominio shiftato con le
    statistiche di tutti gli exit già calcolati (non solo dell'exit corrente).
- **Ipotesi C1.** Il reliability shift spiega almeno metà del calo a severity 3 nei regimi congiunti su CIFAR10: con
  l'affidabilità pulita la λ necessaria scenderebbe sotto quella pulita. *Previsione:* 50–65%.
- **Ipotesi C2.** Il classificatore appreso con le statistiche di più exit recupera almeno 0,03 di savings sul max
  softmax sotto shift. *Previsione:* incerta; sul pulito un controllo precedente non mostrava guadagni di AUROC.
- **Kill.** Se i due ordini danno risposte opposte e nessuna misura distingue le due cause, resta la parte descrittiva.
- **Revisore 2.** "È una scomposizione contabile." Risposta: il controfattuale dice quale rimedio ha senso: se domina
  l'affidabilità serve riadattare score o teste, se domina il traffico basta ricalibrare.
- **Costo.** Circa 30 minuti.

## Pilota D. Una fase di adattamento che ripaga il suo costo

- **Domanda.** Quale adattamento recupera savings al netto del suo compute, e lo fa migliorando l'accuracy o copiando
  gli errori del riferimento?
- **I candidati**, seed 0, severity 3 e 5, stessi split:
  - (a) BN adaptation: statistiche ristimate sulle righe di adapt, sole forward;
  - (b) M8b com'è;
  - (c) BN adaptation, poi teste distillate verso il riferimento adattato;
  - (d) adattamento nello spazio dei logit: per ogni testa una mappa affine sui logit, fittata per distillazione.
    Costa N·C² per valutazione invece di N·D·C: da 25 a 100 volte meno su CIFAR10, da 2,5 a 10 su CIFAR100;
  - (e) M8b economico: un solo valore di L2 (0,01) e 5, 10, 20 iterazioni.
- **Cosa misurare.** Guadagno lordo su M2, compute in forward complete, payback, savings netti su 10.000 e 100.000 input.
  E per ogni candidato: accuracy delle teste e del sistema prima e dopo, accordo con il riferimento prima e dopo, e i
  savings certificati con l'accuracy persa del pilota A.
- **Ipotesi D1.** Una variante economica, (d) o (e), tiene almeno il 70% del guadagno di M8b con al più il 10% del suo
  compute, e porta il payback a severity 3 sotto 30.000 input su entrambi i dataset. *Previsione:* (e) con 10
  iterazioni tiene l'80% del guadagno; (d) tiene meno su CIFAR100.
- **Ipotesi D2.** (c) supera (b) di almeno 0,02 di savings: le due adattano cose diverse. *Previsione:* vera a
  severity 5, incerta a severity 3.
- **Ipotesi D3 (la tassa della copia).** Con M8b l'accordo delle teste con l'exit 9 sale molto più della loro accuracy,
  e meno di metà del suo guadagno di savings sopravvive se si certifica l'accuracy persa. *Previsione:* vera su
  CIFAR100 sotto rumore, dove il riferimento è al 22–36%.
- **Kill.** Se nessuna variante tiene almeno metà del guadagno con al più il 20% del costo, l'adattamento delle teste
  non ripaga sugli orizzonti reali: si riporta come risultato negativo e l'adattamento efficiente resta la sola
  ricalibrazione. Se (c) non supera (b) di 0,02, la BN adaptation resta solo un modo per migliorare il riferimento.
- **Revisore 2.** "La letteratura TTA ha metodi più forti, e il payback dipende dall'implementazione." Risposta:
  riportiamo curve guadagno contro compute, non un punto, e aggiungiamo un metodo alla Tent se (a)–(e) lasciano margine.
- **Sotto pilota opzionale D4.** Con gli early exit attivi i blocchi profondi vedono solo gli input difficili: le
  statistiche BN stimate su quel traffico sono distorte, quelle stimate sugli audit casuali no. Misura la differenza.
- **Costo.** Due job da circa 45 minuti (forward su MPS per la BN adaptation; adattamenti su CPU).

## Pilota E. Soglie per exit con garanzia

- **Domanda.** Soglie diverse per exit, consapevoli del costo, battono la λ condivisa a parità di garanzia? E sotto
  shift vanno riapprese?
- **Le versioni.**
  - E1: score calibrato più regola di Lagrange. Si esce all'exit l se la probabilità stimata di disaccordo è sotto
    μ·(c_9 − c_l). Una famiglia a un parametro, certificata con UCB lungo μ. Sul dominio shiftato usa lo score di M8a:
    verifica se la perdita di M8a diventa un guadagno.
  - E2: cammino appreso. Sulle righe di adapt si ottimizzano le 8 soglie per budget di rischio crescenti; poi si
    certifica lungo il cammino sulle righe di cal. Valido perché il cammino non vede le righe di cal.
  - E3: limite superiore, le 8 soglie migliori scelte sulla popolazione (dal pilota B).
- **Ipotesi E1.** E2 guadagna almeno 0,02 sulla λ condivisa in almeno 5 setting su 8, sul pulito e a severity 3.
  *Previsione:* 0,02–0,04 nei regimi congiunti, meno in disjoint.
- **Ipotesi E2.** Un cammino appreso sul pulito e solo ricertificato sul dominio shiftato tiene almeno l'80% del guadagno
  del cammino riappreso. *Previsione:* vera a severity 1–3, falsa sotto rumore forte.
- **Kill.** Se E2 guadagna meno di 0,02 nella maggior parte dei setting, sul pulito e sotto shift.
- **Revisore 2.** "È incrementale rispetto a Pareto Testing e ai lavori sulle soglie per exit." Risposta: l'angolo è lo
  shift (il cammino si trasferisce o no) e il costo quasi nullo come fase di adattamento: 8 numeri invece di 100.000
  pesi. Va confrontato con D nel bilancio netto.
- **Costo.** Circa 45 minuti.

## Pilota F. Prevedere i savings senza pagarli

- **Cosa è cambiato.** La Parte 1 mostra che con 100 forward complete CRC dà già quasi tutti i savings (0,53 contro 0,54
  con 3000 righe su CIFAR10). "Prevedere invece di misurare" deve quindi battere una misura che costa 100 forward.
- **Domanda ristretta.** I savings certificati di un dominio sono una funzione dell'accuracy del riferimento su quel
  dominio, qualunque sia la corruzione? È la conseguenza attesa della tassa della copia.
- **Misure.** Per ogni rete, i savings di M2 sui 75 domini contro l'accuracy dell'exit 9 e contro l'accordo medio tra
  exit; errore di previsione in leave one corruption out; grafici probit dell'accordo tra exit.
- **Ipotesi F1.** Dentro una rete la correlazione di rango tra savings di M2 e accuracy del riferimento è almeno 0,9, e
  l'errore di previsione in leave one corruption out è sotto 0,03. *Previsione:* vera.
- **Kill.** Errore sopra 0,05, oppure nessun vantaggio rispetto alla stima con 100 audit: la direzione F si chiude e
  resta una figura descrittiva.
- **Costo.** Pochi minuti, solo `results.csv` e le accuracy salvate.

---

## Ordine di esecuzione proposto

1. **A e F** per primi: costano poco e decidono la tesi.
2. **B e C**: danno il quadro e il meccanismo.
3. **D ed E**: i rimedi, giudicati con la metrica che esce da A.

Se A viene uccisa (il calo è perdita reale), la storia torna a quella del piano originale: misura, meccanismo (C) e
rimedi (D, E) sul consistency risk, senza la parte sul certificato.

