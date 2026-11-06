# Risk control per early exit sotto distribution shift

Come mi aveva suggerito ho reimpostato l'idea 2 sulla linea del risk control per early exit e ho fatto un check di letteratura molto più ampio. Il check ha cambiato parecchio l'idea. Il monitor che le avevo descritto esiste già, quindi la frase del documento precedente secondo cui nessuno lo faceva così era sbagliata.

In breve vorrei studiare come una early exit network può mantenere il risk control di Fast yet Safe sotto shift, senza label, riadattando gli exit con il proprio final exit come target. Le scrivo cosa ho trovato, l'idea per il progetto di Neural Networks e come la continuerei nella tesi, dove la finanza tornerebbe come applicazione.

## Cosa ho trovato

Ho controllato in modo sistematico le submission a ICLR 2027, i lavori accettati a ICML 2026, ICLR 2026, NeurIPS 2025 e NeurIPS 2026, TMLR e arXiv fino al 29 ottobre. Le cose che contano per l'idea sono cinque.

Il risk control offline sugli early exit è ormai una linea affollata. I lavori principali sono [Schuster et al. 2021](https://aclanthology.org/2021.emnlp-main.406) e [Schuster et al. 2022](https://arxiv.org/abs/2207.07061), che controllano la consistenza con la full network, e [Laufer Goldshtein et al. 2023](https://arxiv.org/abs/2210.07913), che controlla più parametri insieme. Poi ci sono [Ringel et al. 2024](https://proceedings.mlr.press/v235/ringel24a.html) per la classificazione nel tempo, Fast yet Safe ([Jazbec et al. 2024](https://arxiv.org/abs/2405.20915)) e [Wynn et al. 2026](https://arxiv.org/abs/2510.02480). A questi si aggiungono più di dieci lavori del 2026, quasi tutti offline e con dati i.i.d. o scambiabili. L'eccezione più vicina è UAT ([Bajpai e Hanawal 2025](https://arxiv.org/abs/2509.23666)), che adatta la soglia online con un bandit e ha un bound sul rischio che passa per un proxy allenato con label. Nella sua discussione Fast yet Safe lascia aperti proprio tre punti, cioè la soglia unica per tutti gli exit, lo shift a test time e l'aggiornamento online della soglia. Per gli ultimi due rimanda a [Zollo et al. 2024](https://arxiv.org/abs/2311.13628) e [Feldman et al. 2023](https://arxiv.org/abs/2205.09095).

Il monitor che le avevo proposto è già stato fatto. [Una submission anonima a ICLR 2027](https://openreview.net/forum?id=wbXdPogQnU), pubblica su OpenReview dal 3 ottobre, fa audit random con probabilità nota, un sequential test con false alarm control e il passaggio alla full network quando scatta l'allarme. Decide anche quanti audit fare in base all'evidenza accumulata, che era vicino al contributo che avevo in mente. Però non usa reti multi exit allenate (lavora su cascate di modelli diversi, budget di token e acquisizioni MRI), e dopo l'allarme non recupera più il compute.

Anche il passo successivo esiste, fuori dagli early exit. [Yu et al. 2026](https://arxiv.org/abs/2601.22446) e [Kiyani et al. 2026](https://arxiv.org/abs/2602.17633) aggiornano online la soglia tra un modello economico e uno costoso, usando solo le risposte del modello costoso ottenute con probabilità nota e pesi inversi. [Gibbs et al. 2026](https://arxiv.org/abs/2610.31662) fanno lo stesso per decisioni controllate da un auditor umano o da un modello più forte, con garanzie anche per data stream avversari. Con una soglia unica questi metodi si trasferiscono quasi direttamente a una rete multi exit, quindi il trasferimento da solo non sarebbe un contributo.

Anche scegliere gli audit in base alla confidence, che nel documento precedente indicavo come contributo, è noto. La politica ottima a costo uguale è in [Xu et al. 2024](https://arxiv.org/abs/2406.10490). Con costi diversi la regola segue dall'allocazione di Neyman dei manuali di campionamento, e per la valutazione dei modelli esistono già politiche cost optimal ([Angelopoulos et al. 2025](https://arxiv.org/abs/2506.07949)).

Riallenare gli exit usando come target l'output del final exit si fa già, ma senza garanzie sul rischio. [Zhao et al. 2026](https://www.musmet.eu/scientific_publications/ADEx-_Adaptive_Early_Exit_DNNs_for_Inference_Robustness_to_Popularity_Drift.pdf) riallenano le teste degli exit quando cambia la frequenza delle classi, usando solo gli input che gli exit non hanno fatto uscire. [Nie et al. 2024](https://arxiv.org/abs/2402.04513) fanno qualcosa di simile con una cascata di modelli. Raccolgono le risposte del modello grande sugli input rimandati e su qualche salto random, con una probabilità che decresce nel tempo.

Fuori dagli early exit, [un'altra submission anonima a ICLR 2027](https://openreview.net/forum?id=a79fe9gjJE) fa test time adaptation usando come segnale le predizioni dei layer intermedi della rete stessa. Lo fa per migliorare le predizioni, non per risparmiare compute, e senza garanzie sul rischio.

Da quello che ho trovato al 31 ottobre restano aperte due cose. Nessuno misura quanto il risk control di Fast yet Safe si rompe sotto shift e come questo dipende da come sono stati allenati gli exit, cioè la domanda di [Kubaty et al. 2025](https://proceedings.mlr.press/v267/kubaty25a.html) portata sotto shift. Il lavoro più vicino studia come si rompe il routing di una cascata di due modelli sotto corruzioni ([Mitra et al. 2026](https://arxiv.org/abs/2608.17711)). Varia solo il peso della loss con cui si allena il modello piccolo e non usa risk control. E nessuno usa gli stessi audit sia per tenere sotto controllo il consistency risk sia per riadattare gli exit.

## L'idea

La domanda diventa questa. Una early exit network può mantenere il risk control di Fast yet Safe quando i dati cambiano, senza label, e recuperare il compute risparmiato imparando dal proprio final exit?

È la direzione dei due punti aperti di Fast yet Safe sullo shift e sull'aggiornamento online, con una differenza che conta per gli early exit. Feldman et al. aggiornano la soglia vedendo la loss a ogni passo, mentre in una early exit network la loss di un input uscito presto si vede solo pagando il resto della rete.

Oggi le soluzioni senza label perdono il risparmio dopo uno shift. Il monitor con fallback passa alla full network e ci resta. Un controller che alza la soglia tiene il consistency risk sotto il risk level, ma gli exit restano peggiori sui nuovi dati e quindi escono meno input.

In una early exit network il riferimento del consistency risk si ottiene per due gruppi di input. Il primo, gratis, sono gli input che arrivano comunque in fondo. Il secondo sono gli audit, cioè gli input campionati random che completo fino al final exit pagando il resto della rete. Per questi input conosco l'output di tutti gli exit e del final exit, quindi posso usarli sia per stimare il rischio sia come dati di training per gli exit.

Terrei congelati il backbone e il final exit, così il riferimento resta lo stesso per tutto il data stream. Aggiornerei online le teste degli exit intermedi con il final exit come target. Partirei dalle teste di confidence del vostro lavoro ([Scardapane et al. 2020](https://doi.org/10.1109/ICASSP40776.2020.9054209), [Pomponi et al. 2022](https://doi.org/10.3390/e24010001)), che oggi si allenano contro la label, e le allenerei invece contro l'agreement con il final exit, che non richiede label. Schuster et al. 2022 allenano offline un predittore simile, ma non lo aggiornano sotto shift. Come variante aggiornerei anche le teste di classificazione con una distillazione verso il final exit. La soglia la gestirebbe un controller come quelli di Kiyani et al. e Gibbs et al., alimentato dagli audit.

Il bound del controller sul consistency risk resta valido anche mentre le teste cambiano, perché ogni aggiornamento usa solo dati passati, ma servono tre condizioni. Ogni audit deve avere una probabilità nota e mai nulla, decisa prima di vedere l'output del final exit. Con la soglia al massimo nessun exit deve scattare. E il bound riguarda la media di lungo periodo, quindi dopo ogni shift c'è un transitorio in cui il rischio può stare sopra il risk level. Questa parte è un adattamento di risultati noti e la presenterei così.

Gli audit servono per un motivo preciso. Il consistency risk nasce solo dagli input che escono presto con una confidence alta ma danno una previsione diversa dalla full network. Con i soli input arrivati in fondo, come in Zhao et al., questi input non entrano mai nel training, perché sono proprio quelli che gli exit hanno accettato.

Nie et al. ne vedono alcuni con i salti random, ma con una probabilità che decresce, quindi dopo uno shift tardivo ne vedono sempre meno. Gli audit invece hanno una probabilità che non scende mai sotto un minimo. Per il training non serve che la loss sia non distorta, perché la garanzia non dipende da come alleno le teste, quindi posso tagliare i pesi inversi per ridurre la varianza.

La parte neural del progetto parte da un problema che il vostro gruppo aveva già indicato. Pomponi et al. 2022 notano che fissare la soglia di ogni exit è difficile, perché dipende da accuracy e calibrazione. Il risk control fissa la soglia con una garanzia, e il progetto chiede quali exit, e quali modi di allenarli, la mantengono sotto shift e si riadattano meglio. Confronterei il training joint, disjoint e mixed studiato da Kubaty et al. con Pomponi e una versione con distillazione. Guarderei anche la capacità delle teste e cosa succede quando lo shift torna indietro e gli exit rischiano di dimenticare il dominio iniziale.

## Il progetto di Neural Networks

Il progetto avrebbe tre parti, ognuna utile anche da sola.

La prima è la misura. Userei come rete principale una ResNet20 multi exit su CIFAR10 e CIFAR100, come nei vostri lavori ([Pomponi et al. 2022](https://doi.org/10.3390/e24010001), [Pomponi et al. 2024](https://arxiv.org/abs/2412.19587)), allenata con i diversi regimi di training. Confronterei quattro score di uscita. Il primo è la max softmax di Fast yet Safe, il secondo la stessa con temperature scaling come in [Lilli et al. 2021](https://doi.org/10.1155/2021/5569458), il terzo la confidence appresa di Pomponi et al. 2022 e il quarto un predittore di agreement come quello di Schuster et al. 2022. Calibrerei la soglia con la versione high probability di Fast yet Safe su dati puliti e misurerei il consistency risk realizzato su CIFAR10C e CIFAR100C al crescere della severity.

Per lo shift naturale userei ImageNetV2 e ImageNet R con una MSDNet ImageNet già allenata e pubblica, perché i dataset di Wild Time richiederebbero di allenare reti diverse. Scomporrei il rischio in quanti input escono a ogni exit e quanto spesso ogni exit è in disagreement con la full network. Lo shift può muovere le due cose in direzioni opposte ([Mehra et al. 2022](https://arxiv.org/abs/2212.01562)).

La seconda è il confronto su data stream con uno shift brusco, uno graduale e uno che torna indietro. Metterei a confronto Fast yet Safe statico, il monitor con fallback descritto sopra, il controller con audit e il controller con gli exit che si riadattano. Misurerei il consistency risk nel tempo rispetto al risk level e il compute risparmiato al netto di audit e training. Misurerei anche il tempo per recuperare il risparmio dopo lo shift e l'accuracy, con le label usate solo per valutare.

La terza è il riadattamento vero e proprio. Confronterei il training con i soli input arrivati in fondo, con i soli audit e con entrambi, al variare del tasso di audit e del regime di training.

Con il backbone congelato le feature in ingresso a ogni exit si calcolano una volta sola e si salvano. Così gli esperimenti sui data stream allenano solo teste piccole e girano anche su una GPU modesta. Il costo principale è allenare le ResNet20 multi exit su CIFAR10 e CIFAR100 per tre o quattro regimi, e sono reti piccole.

Il rischio principale è che riadattare gli exit aggiunga poco rispetto alla sola soglia. [Kaplan 2026](https://arxiv.org/abs/2609.14144) trova su LLM che ricalibrare la soglia per ogni deployment alza fino a 59 punti la frazione di token che escono presto, e conclude che conviene personalizzare solo la soglia. Specializzare il readout invece guadagna circa due punti rispetto a un readout già allenato su tutti i tipi di traffico. Un readout allenato su un solo tipo è quasi inutile sugli altri, che è proprio la situazione dopo uno shift, e Zhao et al. trovano un guadagno chiaro riadattando le teste.

Per questo nella quarta settimana farei un test offline che confronta, a parità di risk level, la sola soglia ricalibrata con gli exit riadattati. Se il riadattamento non aiuta, il progetto resta con la misura e il confronto, e il risultato negativo è comunque un risultato.

Il piano sarebbe di circa dieci settimane. Nella prima riprodurrei Fast yet Safe sui logit ImageNet pubblici. Nella seconda e nella terza allenerei le reti e salverei feature e predizioni. Nella quarta farei la misura e il test sul riadattamento. Dalla quinta alla settima farei i data stream e le ablation, nell'ottava lo shift naturale e nelle ultime due scriverei.

## La tesi

Nella tesi continuerei in due direzioni, e la prima sarebbe la finanza come mi aveva suggerito.

Allenerei una piccola early exit network che stima se un fill passivo su Hyperliquid subirà adverse selection. La calibrerei prima del crash del 10 ottobre 2025 e la farei girare durante il crash. I dati di mercato non sono i.i.d., quindi la garanzia statica di Fast yet Safe non vale e il controller con audit è lo strumento adatto. I markout servirebbero solo a capire se peggiora anche la full network. L'adverse selection è rara, quindi servirebbe un risk control per classe.

La seconda direzione è la versione device e server, sul modello del vostro lavoro sugli early exit ricorsivi per le goal oriented communications ([Pomponi et al. 2024](https://arxiv.org/abs/2412.19587)). Lì un audit è un offload forzato al server, e l'output del server diventa anche il target per riadattare gli exit sul device. Il costo dell'audit include la banda, e userei l'allocazione degli audit con costi come strumento già noto.

Se resta tempo affronterei anche l'altro limite che Fast yet Safe dichiara, cioè la soglia unica, controllando online una soglia diversa per ogni exit. L'obiettivo sarebbe una pubblicazione tra ottobre e dicembre 2027.

## Limiti e domande

Il limite principale resta quello del documento precedente. Il consistency risk misura l'agreement con la full network e non l'accuracy, quindi se lo shift peggiora anche la full network servono alcune label, e nella tesi le userei. La garanzia è un adattamento di risultati esistenti, quindi la novità sta nella misura e nel riadattamento neurale. Infine il campo si muove in fretta, e lo stesso gruppo di Fast yet Safe sta lavorando sul monitoraggio online, quindi vorrei partire presto.

Le chiederei tre cose. Le sembra la direzione giusta su questa linea? Preferisce che parta dalla versione standard o direttamente da quella device e server? Ci sono GPU del gruppo che potrei usare per allenare le reti su CIFAR100?

Se ha in mente un'altra direzione sulla stessa linea, la approfondisco volentieri.
