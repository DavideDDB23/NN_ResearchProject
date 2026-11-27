> Copia parziale del documento di progetto "Compute savings degli early exit sotto shift"
> (https://claude.ai/artifact/MNreFiy1XTNWcYZcUC8axi), ricostruita il 26 novembre 2026 da una lettura del 15
> novembre (righe 27-266 della copia locale). Mancano le righe 1-26 e quelle dopo la 266.

## Domande di ricerca e ipotesi

Le domande sono tre, e la prima è quella su cui il professore ha chiesto di concentrarsi.

1. Quanto calano i compute savings sotto shift se il consistency risk deve restare sotto il risk level ε? Lo misurerei in due modi. Con la soglia calibrata su dati puliti e lasciata fissa, come in Fast yet Safe, che può conservare i savings ma rompere il risk control. E con la migliore soglia possibile sul dominio nuovo, che isola il calo dovuto solo allo shift.
2. Senza label e online, quanti savings perde un controller della soglia alimentato dagli audit rispetto a quella soglia migliore, e quanto ci mette a stabilizzarsi dopo lo shift?
3. Quanto dei savings persi recupera una fase di adattamento delle teste, al netto del costo degli audit extra e del training, e dopo quanti input l'adattamento si ripaga?

Le ipotesi da verificare sono cinque.

- H1. A risk level fissato i savings calano al crescere della severity, soprattutto perché gli exit superficiali perdono traffico.
- H2. Con la soglia fissa il consistency risk supera ε alle severity alte, anche se più input arrivano in fondo.
- H3. Il calo dipende dal training regime e dall'exit score, e uno score allenato sull'agreement con il final exit regge meglio della max softmax.
- H4. L'adattamento recupera una parte dei savings persi, e il suo costo viene soprattutto dagli audit extra, non dal training delle teste.
- H5. Allenando solo sugli input arrivati in fondo si recupera meno, oppure il rischio sale, perché tra quegli input mancano proprio quelli accettati in disagreement.

## Setting e notazione

La rete ha L exit, uno dopo ogni blocco della backbone, e l'ultimo è il final exit. Per un input x l'exit l dà la prediction ŷ\_l(x) e uno score s\_l(x) tra 0 e 1. Con soglia λ l'input esce al primo exit con score almeno λ, altrimenti arriva in fondo.

```latex
e_\lambda(x) = \min\{\, l < L : s_l(x) \ge \lambda \,\}, \qquad e_\lambda(x) = L \text{ se nessun exit supera } \lambda
```

Il costo c\_l è il numero di MAC fino all'exit l, teste incluse, e c\_L è il costo di chi arriva al final exit. Il riferimento dei savings è la full network senza teste intermedie, che costa c\_M0, poco meno di c\_L, lo 0,1% su CIFAR10 e l'1,1% su CIFAR100. Il consistency risk è la probabilità che l'exit usato dia una prediction diversa dal final exit. I savings sono la frazione di compute risparmiata rispetto alla full network.

```latex
R_P(\lambda) = \Pr_{x \sim P}\big[\hat y_{e_\lambda(x)}(x) \ne \hat y_L(x)\big], \qquad S_P(\lambda) = 1 - \frac{\mathbb{E}_{x \sim P}\big[c_{e_\lambda(x)}\big]}{c_L}
```

Entrambe si scompongono sugli exit. π\_l(λ) è la frazione di input che esce all'exit l, q\_l(λ) la frazione di questi in disagreement con il final exit.

```latex
S_P(\lambda) = \sum_{l \le L} \pi_l(\lambda)\Big(1 - \frac{c_l}{c_{M0}}\Big), \qquad R_P(\lambda) = \sum_{l<L} \pi_l(\lambda)\, q_l(\lambda)
```

Sotto shift P diventa Q e possono cambiare sia π\_l sia q\_l, anche in direzioni opposte. Siccome ŷ\_L è funzione dell'input, per il consistency risk ogni shift è un covariate shift. Con un campione non etichettato di Q la soglia si ricalibrerebbe offline eseguendo la full network, quindi il caso interessante è online, quando lo shift arriva senza preavviso.

La quantità centrale del progetto è il miglior risparmio ottenibile a risk level ε, con le teste attuali θ o con teste adattate θ'.

```latex
S^*_Q(\varepsilon;\theta) = \max_\lambda \big\{\, S_Q(\lambda;\theta) : R_Q(\lambda;\theta) \le \varepsilon \,\big\}
```

```latex
\Delta S_{\text{shift}} = S^*_P(\varepsilon;\theta) - S^*_Q(\varepsilon;\theta), \qquad \Delta S_{\text{adapt}} = S^*_Q(\varepsilon;\theta') - S^*_Q(\varepsilon;\theta)
```

ΔS\_shift è quanto risparmio fa perdere lo shift a parità di garanzia. ΔS\_adapt è quanto se ne può recuperare adattando gli exit, e il loro rapporto è la frazione recuperabile, che il test della quarta settimana stima offline.

La tabella dà i costi della ResNet20 multi exit con 9 exit, contati con un contatore di FLOPs sul codice vero, che conta convoluzioni e strati lineari. Gli exit da 1 a 8 hanno la pooled head di Kubaty et al. 2025, cioè un mix di max e average pooling fino a 4x4, uno strato lineare e un'unità di confidence. L'exit 9 è il classificatore standard della ResNet20. Il costo di un exit comprende le teste valutate fino a lì.

| Exit | Canali | Risoluzione | Costo cumulato su CIFAR10 (milioni di MAC) | Frazione di c\_M0 su CIFAR10 | Frazione di c\_M0 su CIFAR100 |
| --- | --- | --- | --- | --- | --- |
| 1 | 16 | 32x32 | 5,16 | 0,127 | 0,128 |
| 2 | 16 | 32x32 | 9,89 | 0,244 | 0,245 |
| 3 | 16 | 32x32 | 14,61 | 0,360 | 0,362 |
| 4 | 32 | 16x16 | 18,15 | 0,448 | 0,450 |
| 5 | 32 | 16x16 | 22,88 | 0,564 | 0,568 |
| 6 | 32 | 16x16 | 27,60 | 0,681 | 0,686 |
| 7 | 64 | 8x8 | 31,15 | 0,768 | 0,775 |
| 8 | 64 | 8x8 | 35,88 | 0,885 | 0,894 |
| 9, final exit | 64 | 8x8 | 40,60 | 1,001 | 1,011 |

Un input che esce al primo exit fa risparmiare l'87% del compute, e un audit su quell'input costa l'87% di un forward completo. Le teste costano in tutto lo 0,12% della rete su CIFAR10 e l'1,08% su CIFAR100, dove lo strato lineare ha cento uscite. Chi arriva al final exit paga anche tutte le teste, quindi su quegli input i savings sono leggermente negativi.

## Il bilancio del compute sotto shift

I savings che contano sono quelli netti. Nel deployment ogni input costa il percorso fino al suo exit, più il resto della rete se diventa un audit, più la sua quota del training delle teste.

```latex
C_t = c_{e_t} + B_t\,\mathbb{1}[e_t < L]\,\big(c_L - c_{e_t}\big) + A_t
```

B\_t vale 1 con probabilità π\_t, cioè quando l'input diventa un audit, e A\_t è il costo del training ripartito sugli input. Su una finestra di n input i savings netti sono i savings lordi meno due overhead.

```latex
S_{\text{net}} = 1 - \frac{\sum_t C_t}{n\, c_L} = S_{\text{lordi}} - O_{\text{audit}} - O_{\text{adapt}}, \qquad \mathbb{E}\big[O_{\text{audit}}\big] = \pi\, S_{\text{lordi}} \;\text{ con } \pi \text{ costante}
```

Con un tasso di audit costante resta quindi una frazione uno meno π del risparmio lordo, lo stesso risultato della Proposition 4 della submission 22466.

La fase di adattamento ha due costi. Il primo sono gli audit in più per raccogliere i dati, cioè N\_a input con un tasso di audit π\_a più alto di π. Il secondo è il training delle teste, K\_train MAC in tutto. Se dopo l'adattamento i savings netti salgono di ΔS, l'adattamento si ripaga dopo H\_p input.

```latex
\text{Costo}_{\text{adapt}} = N_a(\pi_a - \pi)\, S_{\text{lordi}}\, c_L + K_{\text{train}}, \qquad H_{\text{p}} = \frac{N_a(\pi_a - \pi)\, S_{\text{lordi}}}{\Delta S} + \frac{K_{\text{train}}}{\Delta S\, c_L}
```

Un esempio serve solo a mostrare il calcolo, non è un risultato. Con ResNet20 su CIFAR100, N\_a = 2.000, π\_a = 0,5, π = 0,05, savings lordi 0,3 dopo lo shift e ΔS = 0,1, il primo termine vale 2.700 input. Con le pooled head, un buffer di 1.000 esempi e 20 epoche il training costa circa 26 miliardi di MAC, contando forward e backward delle teste come tre forward, cioè circa 650 input. L'adattamento si ripagherebbe dopo circa 3.350 input, e quasi tutto il costo viene comunque dagli audit.

Per ogni stream misurerei i savings netti prima dello shift S\_0, quelli stabilizzati dopo lo shift senza adattamento S\_1 e quelli dopo l'adattamento S\_2. Da questi vengono la frazione recuperata e il payback H\_p.

```latex
\text{frazione recuperata} = \frac{S_2 - S_1}{S_0 - S_1}
```

&#91;embedded content: savings netti nel tempo · schema qualitativo\]

Lo shift abbassa i savings da S\_0 a S\_1. La fase di adattamento costa altri savings, in giallo, e li restituisce dopo, in verde. L'adattamento conviene se il verde supera il giallo prima del prossimo shift.

### Come cambiano i savings, nel dettaglio

Il calo dei savings sotto shift ha due cause, e conviene misurarle separatamente. La prima è lo shift a soglia ferma, che cambia quali input superano λ\_P. La seconda è l'aumento della soglia da λ\_P a λ\_Q, che serve a riportare il rischio sotto ε.

```latex
S_P(\lambda_P) - S_Q(\lambda_Q) = \underbrace{S_P(\lambda_P) - S_Q(\lambda_P)}_{\text{shift a soglia ferma}} + \underbrace{S_Q(\lambda_P) - S_Q(\lambda_Q)}_{\text{prezzo del risk control}}
```

Il primo termine può essere negativo. Se lo shift rende gli exit superficiali più sicuri di sé ma più spesso in disagreement, a soglia ferma escono più input e i savings salgono, ma il rischio supera ε. Quei savings sono illusori, e nei confronti conterei solo i savings validi, cioè quelli ottenuti con il rischio sotto ε.

Ogni termine si scompone sugli exit, con Δπ\_l il cambiamento della frazione di input che esce all'exit l. Così si vede quali exit perdono traffico e quanto pesa ciascuno, perché un input perso al primo exit vale 0,87 di full network e uno perso all'ottavo solo circa 0,11.

```latex
\Delta S = \sum_{l<L} \Delta\pi_l \Big(1 - \frac{c_l}{c_L}\Big)
```

Anche il transitorio del controller ha un costo. In media la soglia si muove di γ per l'eccesso di rischio a ogni input, quindi per salire da λ\_P a λ\_Q servono circa τ input, con R̄\_Q il rischio medio lungo la salita. Durante il transitorio i savings sono più alti del dovuto ma il rischio sta sopra ε, e anche questo va contato. È una stima approssimata, da confrontare con gli stream.

```latex
\tau \approx \frac{\lambda_Q - \lambda_P}{\gamma\,\big(\overline{R}_Q - \varepsilon\big)}
```

Gli audit non costano tutti uguale. Un audit su un input uscito all'exit l costa la parte di rete che manca, quindi auditare il primo exit costa circa sette volte più che auditare l'ottavo. Con una probabilità di audit p\_l diversa per ogni exit l'overhead diventa questo.

```latex
\mathbb{E}\big[O_{\text{audit}}\big] = \sum_{l<L} \pi_l(\lambda)\, p_l \Big(1 - \frac{c_l}{c_L}\Big)
```

A parità di overhead la varianza della stima del rischio è minima con p\_l proporzionale alla radice di q\_l diviso il costo, dove q\_l si stima dagli audit stessi. È l'allocazione di Neyman con costi, già usata per la valutazione dei modelli da Angelopoulos et al. 2025, quindi non è un contributo. È però la leva più semplice per abbassare O\_audit, e la garanzia resta valida finché ogni p\_l resta sopra un minimo p\_min.

```latex
p_l \propto \sqrt{\frac{q_l}{1 - c_l/c_L}}, \qquad p_l \ge p_{\min}
```

La fase di adattamento si può progettare in tre modi, e il progetto li confronta.

- Fallback completo. Durante la fase tutti gli input vanno fino in fondo, quindi i dati arrivano in fretta ma i savings scendono a zero per N\_a input.
- Audit rinforzati. Il tasso sale a π\_a minore di 1, la fase dura di più e costa meno per input. È il caso dello schema sopra.
- Adattamento continuo. Nessuna fase, piccoli aggiornamenti ogni k input con il tasso normale π. Costa poco ma recupera più lentamente.

In tutti e tre i casi gli input arrivati in fondo da soli danno dati gratis. Usarli riduce il costo, ma tra loro mancano gli input accettati in disagreement, ed è per questo che serve una quota di audit.

Le figure principali per rispondere al professore sarebbero tre. La curva dei savings validi a ε fissato contro la severity, per training regime ed exit score. La scomposizione per exit del calo. E la timeline dei savings netti attorno allo shift, con frazione recuperata e payback, per le tre versioni della fase di adattamento.

## Il sistema online

Il sistema ha due modalità. In quella normale gli input escono con la soglia λ\_t, una piccola frazione π degli input usciti viene completata come audit, e un controller aggiorna λ\_t. Quando un trigger segnala lo shift parte la fase di adattamento, che alza il tasso di audit a π\_a per N\_a input, riallena le teste e poi torna alla modalità normale.

&#91;embedded content: sistema online · inferenza, feedback, adattamento\]

Gli input non usciti e gli audit arrivano al final exit. Da lì partono sia z\_t per il controller sia i dati del buffer per la fase di adattamento.

Per ogni input il sistema fa sei passi.

1. L'input x\_t attraversa la backbone. A ogni exit l si calcolano ŷ\_l e s\_l, e l'input esce al primo exit con s\_l sopra λ\_t.
2. Se esce prima del final exit, con probabilità π\_t diventa un audit. π\_t si decide prima di vedere il final exit, la risposta dell'exit parte subito e il forward continua fino in fondo.
3. Se l'input arriva al final exit, da solo o per un audit, si conoscono le prediction di tutti gli exit e del final exit. Nel buffer vanno le feature in ingresso a ogni testa e i logit del final exit.
4. Per un audit si osserva z\_t, che vale 1 se l'exit usato è in disagreement con il final exit. Il controller alza λ\_t se l'audit trova un disagreement e la abbassa di poco altrimenti, con la regola scritta nella sezione sulle garanzie.
5. Il trigger guarda λ\_t e la stima dei savings. Se λ\_t sale oltre λ̂ + κ, oppure scatta un test sequenziale sul disagreement, parte la fase di adattamento.
6. Nella fase di adattamento il tasso di audit sale a π\_a per N\_a input. Poi si riallenano le teste sul buffer e le teste nuove sostituiscono le vecchie tutte insieme.

Come variante al trigger confronterei un adattamento continuo, con un piccolo aggiornamento delle teste ogni k input e tasso di audit costante.

Ogni testa ha due uscite. La testa di classificazione si allena con una distillazione verso il softmax del final exit. La testa di confidence, come in Pomponi et al. 2022 ma con un target diverso, si allena con una binary cross entropy contro l'agreement 1\[ŷ\_l = ŷ\_L\], ricalcolato con la testa di classificazione corrente. Dopo l'adattamento è questa confidence a fare da score s\_l.

Nel training gli input arrivati in fondo da soli pesano 1 e gli audit pesano uno su π\_t, tagliato a un massimo w\_max per ridurre la varianza. Per non dimenticare il dominio pulito quando lo shift torna indietro si tiene un piccolo buffer di esempi puliti, oppure una penalità alla Learning without Forgetting sulle uscite delle teste vecchie.

## Garanzie

Il controller aggiorna la soglia dopo ogni input con questa regola. B\_t z\_t vale 1 solo per un audit che trova un disagreement, quindi senza audit la soglia scende di γε.

```latex
\lambda_{t+1} = \min\big\{\lambda_{\max},\; \max\{0,\; \lambda_t + \gamma\,(B_t z_t/\pi_t - \varepsilon)\}\big\}
```

Il controller ha un bound sul consistency risk medio di lungo periodo che vale per qualunque stream, anche avversario. È un adattamento del Theorem 5.1 di Kiyani et al. 2026 e della sezione 5.1 di Gibbs et al. 2026, e va dimostrato con cura nelle prime settimane. Con tasso di audit costante π, step γ e λ\_max = 1, con probabilità almeno uno meno δ vale questo bound.

```latex
\frac{1}{T}\sum_{t=1}^{T} z_t \;\le\; \varepsilon + \frac{1 + \gamma/\pi}{\gamma T} + \sqrt{\frac{2(1-\pi)\log(1/\delta)}{\pi T}} + \frac{2\log(1/\delta)}{3T}
```

Il bound richiede quattro condizioni.

1. Ogni audit ha una probabilità nota π\_t ≥ π\_min > 0, decisa prima di vedere l'output del final exit e indipendente da quello che l'audit deve rivelare.
2. Con λ ≥ 1 nessun exit scatta, quindi z\_t = 0. Con score tra 0 e 1 basta che l'input esca solo se s\_l supera strettamente λ.
3. Teste e soglia al passo t dipendono solo dai dati fino al passo precedente. Riallenare le teste sul buffer rispetta la condizione se le teste nuove entrano in uso solo dopo il training.
4. Backbone e final exit restano fissi, così z\_t misura sempre il disagreement con la stessa full network.

Il bound riguarda la media di lungo periodo. Dopo uno shift il rischio può restare sopra ε per un transitorio, finché λ non sale abbastanza, e la durata dipende da γ e da quanto deve salire λ.

Il termine dei pesi inversi è grande. Con π = 0,05 e δ = 0,1 vale circa 3 punti percentuali dopo 100.000 input e circa 1 dopo un milione. Per ridurlo si può usare la testa di confidence come control variate, con la stima AIPW della submission 22466, che dà anche un certificato anytime valid sul rischio cumulativo.

Il secondo risultato da scrivere riguarda il training. Con i soli input arrivati in fondo i dati vengono dalla regione in cui nessun exit intermedio supera λ, mentre il consistency risk sta tutto nella regione complementare. Con audit a probabilità nota la loss pesata con uno su π è non distorta su tutta Q. Per il training però non serve, perché la garanzia non dipende da come si allenano le teste, quindi i pesi si possono tagliare.

Il bound non dice nulla sull'accuracy, non vale istante per istante e non promette che l'adattamento aumenti i savings. Quest'ultima è proprio la domanda sperimentale.

## Esperimenti

La prima domanda si risolve tutta offline, con logit e feature salvati, e solo la seconda e la terza richiedono gli stream. La tabella fissa le scelte di partenza.

| Componente | Scelta | Note |
| --- | --- | --- |
| Rete | ResNet20 multi exit, 9 exit, pooled head di Kubaty et al. 2025 con strato lineare e unità di confidence | Backbone di Pomponi et al. 2022, 3 seed di training |
| Dati | CIFAR10 e CIFAR100, shift con CIFAR10C e CIFAR100C, 19 corruzioni e 5 severity | [Hendrycks e Dietterich 2019](https://arxiv.org/abs/1903.12261) |
| Shift naturale | ImageNetV2 e ImageNet R con una MSDNet ImageNet già allenata | Opzionale, all'inizio solo per la prima domanda |
| Training regime | Joint, disjoint, mixed, più una variante con self distillation | Mixed significa backbone prima, poi tutta la rete ([Kubaty et al. 2025](https://proceedings.mlr.press/v267/kubaty25a.html)) |
| Exit score | Max softmax, temperature scaling per exit, confidence contro la label, confidence contro l'agreement con il final exit | Lilli et al. 2021, Pomponi et al. 2022, Schuster et al. 2022 |
| Calibrazione | Fast yet Safe high probability con UCB, δ = 0,1, ε tra 0,01 e 0,1 | 5.000 immagini pulite del test set |
| Valutazione | Le altre 5.000 immagini e le loro versioni corrotte | Verificato che CIFAR C segua l'ordine del test set, per ogni corruzione e severity |
| Stream | Salto, rampa di severity, gradini, ritorno al pulito, corruzioni miste, da 20.000 a 50.000 input, 50 ripetizioni | Costruiti con gli indici di valutazione |
| Audit | π tra 0,01 e 0,2, π\_a e N\_a su una griglia | Il costo segue la sezione sul bilancio |

I metodi a confronto sono nove, da M0 a M8.

- M0. Full network, nessun risparmio e nessun rischio.
- M1. Fast yet Safe statico, soglia calibrata su dati puliti.
- M2. Soglia ottima sul dominio nuovo, cioè il limite superiore senza adattamento.
- M3. Monitor con fallback alla full network, come la submission 22466.
- M4. Controller con audit, solo soglia.
- M5. Controller più adattamento con i soli input arrivati in fondo, come ADEx.
- M6. Controller più adattamento con i soli audit.
- M7. Controller più adattamento con entrambi, il metodo proposto.
- M8. Adattamento offline con accesso al dominio nuovo, il limite superiore con adattamento.

Per M7 confronterei le tre versioni della fase di adattamento descritte nel bilancio del compute, cioè fallback completo, audit rinforzati e adattamento continuo.

Per ogni metodo misurerei il consistency risk nel tempo rispetto a ε (media di lungo periodo e finestra peggiore), i savings lordi e netti, S\_0, S\_1, S\_2, la frazione recuperata e il payback H\_p. Accanto riporterei l'accuracy contro le label vere, usate solo per valutare, e il forgetting quando lo shift torna indietro.

Con backbone congelato le feature in ingresso a ogni testa si calcolano una volta sola. Con le pooled head sono 4.352 numeri per immagine, cioè le mappe 4x4 già mescolate con α, che dopo il training resta fisso. In float16 fanno circa 8,3 GB per rete sui 950.000 esempi di CIFAR C di un dataset, e sono salvate per le 8 reti del seed 0. Gli esperimenti online allenano solo teste piccole e girano su CPU o su una GPU qualsiasi.

I savings si misurano in MAC. Il tempo reale dipende da batch e hardware e può divergere dai MAC, quindi lo riporterei solo come controllo a batch 1.

## Piano di lavoro

Il piano dura circa dieci settimane, con una decisione alla fine della quarta.

1. Settimana 1. Riprodurre Fast yet Safe sui logit ImageNet pubblici del repository RC EENN. Scrivere la dimostrazione del bound del controller con le quattro condizioni. Verificare l'ordine degli indici di CIFAR C.
2. Settimane 2 e 3. Codice della ResNet20 multi exit con le teste di confidence. Training dei regimi su CIFAR10 e CIFAR100 con 3 seed, prima misurando il tempo di un'epoca. Salvataggio di logit e feature su test pulito e CIFAR C, e conteggio dei MAC.
3. Settimana 4. Prima domanda offline, cioè savings a ε fissato contro severity, per regime e score, con la scomposizione. Test sull'adattamento, confrontando ΔS\_adapt con ΔS\_shift tramite l'adattamento offline M8. Decisione sul metodo.
4. Settimane 5, 6 e 7. Stream, controller e fase di adattamento, metodi da M1 a M7, griglie di π, π\_a e N\_a.
