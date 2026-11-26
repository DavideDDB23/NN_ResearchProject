# Compute savings degli early exit sotto shift

Sotto shift i compute savings calano molto anche quando il risk control di Fast yet Safe resta valido. Adattare gli exit head senza label recupera solo una parte dei savings persi, e riallenarli per intero si ripaga solo su horizon molto lunghi. Le affine map leggere sui logit, invece, migliorano i net savings anche sui primi data stream online.

## Setup

Ho usato il backbone ResNet-20 di [Pomponi et al. 2022](https://doi.org/10.3390/e24010001) su CIFAR-10 e CIFAR-100, con un exit dopo ognuno dei nove residual block. Come exit head ho preso i pooled head di [Kubaty et al. 2025](https://proceedings.mlr.press/v267/kubaty25a.html), perché con i branch di Pomponi et al. 2022 un solo early exit head costa circa il 41% del backbone. Ho allenato i training regime joint, disjoint e mixed di Kubaty et al. 2025 e una versione di joint con self distillation dal final exit (joint_sd), con tre seed ciascuno, per 24 network.

Come exit score confronto la max softmax, anche con temperature scaling, e due learned score che stimano se l'exit è corretto e se è d'accordo con il final exit. Li costruisco su tre statistiche dell'output dell'exit head, perché la confidence di Pomponi et al. 2022, approssimata post hoc sulle feature, separa le prediction corrette da quelle sbagliate peggio della max softmax.

Come shift uso le 15 corruption standard di CIFAR-10-C e CIFAR-100-C ([Hendrycks e Dietterich 2019](https://arxiv.org/abs/1903.12261)) a cinque severity. Calibro uno shared threshold con la versione high probability di Fast yet Safe ([Jazbec et al. 2024](https://arxiv.org/abs/2405.20915)), imponendo che il consistency risk rispetto al final exit resti sotto ε = 0,05, e misuro i savings in MAC rispetto alla ResNet-20 senza exit head. Del test set, 4.000 immagini servono per l'adaptation, 3.000 per la calibration e 3.000 per l'evaluation.

Offline confronto quattro metodi (le sigle da M3 a M7 sono dei metodi online). M1 mantiene il threshold calibrato sul clean. M2 lo ricalibra su 3.000 immagini del nuovo domain ed è una baseline ottimistica, perché assume di conoscere il domain. M8a riallena sul nuovo domain solo l'agreement score, che era l'idea iniziale della mia proposta. M8b adatta senza label il linear layer degli exit head con distillation verso il final exit, tenendo frozen backbone e final exit. Dopo l'adaptation, M8a e M8b ricalibrano il threshold come M2.

## Risultati offline

Ho riprodotto esattamente i risultati di Fast yet Safe sui loro logit ImageNet.

Sotto shift, con un fixed threshold calibrato sul clean (M1), il risk control si rompe. Su CIFAR-10 il consistency risk sale da 0,065 a severity 1 fino a 0,127 a severity 5 (figura 2), e supera ε nel 58% dei casi a severity 1 e nel 90% a severity 5. Ricalibrando sul nuovo domain (M2) il risk control tiene, ma a severity 3 resta solo il 59% dei savings clean su CIFAR-10 e il 53% su CIFAR-100 (figura 1, confidence interval al 95% dal 56 al 62% e dal 50 al 56%). Nella figura 1 M1 ha più savings di M2 solo perché il suo risk supera ε, come si vede nella figura 2.

![Savings contro severity per M1, M2 e M8b](docs/figures/savings_by_method.png)

*Figura 1. Savings contro severity per M1, M2 e M8b.*

![Consistency risk contro severity per M1 e M2](docs/figures/risk_by_method.png)

*Figura 2. Consistency risk contro severity per M1 e M2.*

Il drop dei savings viene soprattutto dagli shallow exit, che perdono traffic, in particolare su CIFAR-10 con i training regime che allenano tutta la network, mentre con disjoint e su CIFAR-100 è distribuito su più exit (figura 3). Su CIFAR-10 dal 61 al 71% del drop viene dagli score che cambiano, per cui con lo stesso threshold escono meno input, e su CIFAR-100 dall'85 all'88%. Il resto viene dal threshold, che deve salire per riportare il risk sotto ε.

![Drop dei savings da clean a M2 per exit](docs/figures/savings_drop_by_exit.png)

*Figura 3. Drop dei savings da clean a M2 per exit (severity da 3 a 5), diviso tra shift a parità di threshold e threshold che sale.*

Il training regime conta più della scelta dello score, che cambia i savings al massimo di 0,04 (figura 4). A severity 3 su CIFAR-100 joint_sd ha 0,036 di savings in più rispetto a joint, ma circa 2,4 punti di accuracy in meno sul final exit. Disjoint è il regime con meno savings ovunque.

![Savings di M2 contro severity per i quattro exit score](docs/figures/savings_by_score.png)

*Figura 4. Savings di M2 contro severity per max softmax, temperature scaling e i due learned score (label e agreement).*

Riallenare l'agreement score (M8a) peggiora i savings rispetto a M2 con lo stesso agreement score (figura 4). Su CIFAR-10, in media sulle severity da 3 a 5, sono 0,255 con M8a contro 0,272 con M2. Probabilmente succede perché, con un solo shared threshold, uno score calibrato sposta gli input verso i deep exit, che fanno risparmiare meno.

Adattare gli exit head (M8b) va meglio. Dalla severity 3 in su recupera in mediana dall'11 al 25% del drop, a seconda di dataset e training regime, con lo stesso risk control di M2 e un gain quasi identico sui tre seed (0,047, 0,043 e 0,044). Il criterio che mi ero fissato, recuperare almeno un quarto del drop in metà dei casi, è rispettato però solo per CIFAR-10 joint_sd, e solo al limite. Il training degli exit head costa poi circa 2.600 full forward su CIFAR-10 e 30.900 su CIFAR-100 (network joint). Contando anche i 4.000 input di adaptation, a severity 3 il costo si ripaga solo dopo circa 81.000 input su CIFAR-10 e 1,7 milioni su CIFAR-100.

Ho anche dimostrato il long run bound sul consistency risk del controller con audit, adattando [Kiyani et al. 2026](https://arxiv.org/abs/2602.17633) e [Gibbs et al. 2026](https://arxiv.org/abs/2610.31662). Sugli horizon di CIFAR però è largo, perché con ε = 0,05 e un audit rate π = 0,05 vale 0,156 dopo 10.000 input e 0,081 dopo 100.000.

## Margin, cheap adaptation e primi data stream online

Nel primo pilot ho cercato dove si perde il margin tra i savings possibili sotto shift e quelli certificati da M2. Sulle 24 network quasi tutto il margin sta tra due livelli. In alto c'è un per input oracle, che fa uscire ogni input al primo exit d'accordo con il final exit (circa 0,68 a severity 3 su CIFAR-10 e 0,47 su CIFAR-100, figura 5). In basso c'è qualunque exit rule basata sullo score (circa 0,31 e 0,12 per M2). Il salto c'è perché lo score non separa bene gli input d'accordo con il final exit da quelli in disaccordo. Con il miglior shared threshold si guadagnano solo da 0,009 a 0,015 su M2, e otto per exit threshold a parità di risk aggiungono circa 0,02 o 0,03 su CIFAR-10 rispetto a uno shared threshold. Se invece il constraint è sull'accuracy loss, misurata con le label, i savings non calano sotto shift (linee tratteggiate della figura 5), quindi il drop viene dal consistency constraint rispetto al final exit.

![Savings contro severity per oracle, per exit threshold, shared threshold e constraint sull'accuracy loss](docs/figures/b1_margin.png)

*Figura 5. Savings contro severity per il per input oracle, il per exit threshold, lo shared threshold e il constraint sull'accuracy loss.*

Nel secondo pilot ho cercato un'adaptation phase che si ripaghi. Con il gain di M8b a severity 3, già i 4.000 input di adaptation, che passano per tutta la network, portano il payback a circa 26.000 input su CIFAR-10 e 29.000 su CIFAR-100, anche se il training costasse zero. Con le affine map sui logit e budget di adaptation ridotti fino a 500 input, il payback scende sotto 30.000 input su CIFAR-10, e sotto 10.000 se l'adaptation usa solo gli input che arrivano comunque al final exit. Su CIFAR-100, però, con 500 input l'affine map va in overfitting.

La BatchNorm adaptation non alza i savings a parità di consistency risk, ma migliora nettamente la system accuracy (a severity 5 da circa 0,46 a 0,63 su CIFAR-10, figura 6). M8b invece riduce il compute a parità di accuracy, il che suggerisce che guadagna allineandosi al final exit degradato. Usate insieme, le due adaptation migliorano sia l'accuracy sia il compute (figura 6).

![Frontier tra accuracy e compute](docs/figures/b2_frontier.png)

*Figura 6. Frontier tra accuracy e compute per M2, adaptation degli exit head, BatchNorm adaptation e le due insieme.*

Ho poi provato le affine map sui primi data stream online, a severity 3 e con 180 run per dataset (figura 7). Con uno step shift M1 resta sopra ε, mentre il controller con audit (M4) riporta il consistency risk a ε e converge ai net savings di M2. L'adaptation phase online (M7) recupera una parte di net savings in più rispetto a M4, fino a circa 0,34 o 0,35 su CIFAR-10 e 0,14 su CIFAR-100. L'ho provata con boosted audit, cioè più audit per un periodo dopo lo shift, e con continuous adaptation. Il payback è nell'ordine delle decine di migliaia di input, mentre la versione con full fallback, che durante l'adaptation manda tutti gli input alla full network, non si ripaga entro i 30.000 input dello stream. Quando lo stream torna al clean o cambia domain più volte, adattare senza forgetting protection provoca uno spike di consistency risk al domain change (fino a circa 0,08 su CIFAR-10 e 0,14 su CIFAR-100, figura 8). La forgetting protection lo elimina e preserva i savings sul clean.

![Data stream online con step shift a severity 3](docs/figures/online_step_shift.png)

*Figura 7. Data stream online con step shift a severity 3, media su 180 run per dataset.*

![Return to clean sullo stream online a severity 3](docs/figures/online_return_to_clean.png)

*Figura 8. Return to clean sullo stream online a severity 3, con e senza forgetting protection, media su 60 run per dataset.*

Su ImageNet una prima prova, con un backbone pretrained e frozen e gli exit head allenati su 22.500 immagini, ha dato savings clean intorno al 3%. Ho già pronto un secondo run con fino a 200.000 immagini di ImageNet train, in cui mi aspetto savings clean maggiori.

Come prossimo passo estenderei gli esperimenti online a tutti gli shift scenario (gradual, staircase, multiple shift e return to clean), confrontando anche adaptation phase alimentate dai soli input arrivati al final exit, dai soli audit o da entrambi. Per la conferma finale userei CIFAR-10.1 e le 4 extra corruption, che ho tenuto finora come held out set, e misurerei la latency reale con batch size 1.

Il limite principale è che il consistency risk misura l'agreement con il final exit e non l'accuracy, e nei training regime che allenano tutta la network il final exit non è più accurato dell'exit che lo precede. I savings poi sono ancora misurati in MAC e non in latency, e la BatchNorm adaptation cambia la reference del consistency risk, cioè il final exit stesso.
