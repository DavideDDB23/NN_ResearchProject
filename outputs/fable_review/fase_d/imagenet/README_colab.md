# ImageNet con teste allenate su ImageNet train: cosa fare

Stesso procedimento della volta scorsa, con il notebook nuovo `imagenet_minimo.ipynb` (questa cartella). Il token di
Hugging Face che hai già va bene.

## Passi

1. Apri colab.research.google.com, poi File > Carica blocco note, e scegli il nuovo `imagenet_minimo.ipynb`.
2. Runtime > Cambia tipo di runtime: scegli **L4** o **A100** se ci sono (hanno più CPU: circa un'ora), altrimenti
   **T4** (2–3 ore, meno CU). Salva.
3. Runtime > Esegui tutte (se compare l'avviso sul notebook non creato da Google, "Esegui comunque").
4. Quando la cella 2 chiede il token di Hugging Face, incollalo e premi Invio.
5. Lascia la scheda aperta. Alla fine il browser scarica `imagenet_risultati.zip` (circa 40 MB) nella cartella
   Download e l'ultima cella stampa "FATTO". Scrivimi "fatto".

Se compare un messaggio che comincia con **STOP** (per esempio se il blocco di ImageNet train non copre tutte le
classi), scrivimelo: non serve rilanciare. Per qualunque altro errore, mandami il testo rosso.

Costo stimato, da verificare sul pannello di Colab: circa 6 CU con la T4, 12–15 con l'A100.

## Per me: cosa fa il notebook

Cella 3: ImageNet val da `ILSVRC/imagenet-1k`. Cella 4: ImageNet-V2 e ImageNet-R. Cella 5: il primo archivio di
ImageNet train dello stesso dataset, estratto fino a 200.000 immagini. Celle 6–7: scrive `exits.py` e
`colab_imagenet.py`. Cella 8: feature delle immagini di train, 8 teste (10.000 immagini di train tenute da parte per
scegliere le epoche, al più 15), tabelle compatte per 7 domini (pulito, V2, R, 4 corruzioni a severity 3), test delle
mappe; si ferma con STOP se il blocco di train copre meno di 990 classi. Cella 9: zip con anche i pesi delle teste.
