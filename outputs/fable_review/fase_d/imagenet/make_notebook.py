# Phase D, entry D.07: builds imagenet_minimo.ipynb, a one-click Colab notebook that downloads the data (ImageNet val
# from the gated Hugging Face dataset ILSVRC/imagenet-1k with Davide's token, ImageNet-V2 and ImageNet-R from their
# public sources), writes exits.py and colab_imagenet.py from this folder, runs them, and downloads a small zip of
# results. Written by Opus 5.5, 16 November 2026. Usage (from the repository root):
#   python outputs/fable_review/fase_d/imagenet/make_notebook.py

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def main():
    exits = open(os.path.join(HERE, "exits.py")).read()
    script = open(os.path.join(HERE, "colab_imagenet.py")).read()
    cells = [
        md("""
# ImageNet con teste allenate su ImageNet train (Fase D)

Premi **Runtime > Esegui tutte**. Il notebook ti chiede una sola cosa: il token di Hugging Face (cella 2). Alla fine
il browser scarica `imagenet_risultati.zip` (circa 40 MB) nella cartella Download. Tempo: 2–3 ore con la T4, circa
un'ora con l'A100. Se si ferma con un messaggio che comincia con STOP, scrivilo a Claude.
"""),
        code("""
# 1. GPU e pacchetti
!nvidia-smi --query-gpu=name --format=csv,noheader || echo "ATTENZIONE: nessuna GPU, scegli Runtime > Cambia tipo di runtime > T4 GPU"
!pip -q install imagecorruptions huggingface_hub
"""),
        code("""
# 2. Il token di Hugging Face (serve solo per ImageNet val). Incollalo e premi Invio.
from getpass import getpass
from huggingface_hub import login
login(token=getpass("Incolla il token di Hugging Face e premi Invio: "))
"""),
        code("""
# 3. ImageNet val (50.000 immagini) da ILSVRC/imagenet-1k
import os, tarfile, io
from huggingface_hub import hf_hub_download, list_repo_files
os.makedirs("/content/val", exist_ok=True)
repo = "ILSVRC/imagenet-1k"
files = list_repo_files(repo, repo_type="dataset")
tars = [f for f in files if "val" in f.lower() and f.endswith((".tar.gz", ".tar"))]
shards = [f for f in files if "valid" in f.lower() and f.endswith(".parquet")]
if tars:
    for f in tars:
        path = hf_hub_download(repo, f, repo_type="dataset", local_dir="/content/hf")
        with tarfile.open(path) as t:
            t.extractall("/content/val", filter="data")
        os.remove(path)
elif shards:
    import pandas as pd
    classes = {}
    exec(open(hf_hub_download(repo, "classes.py", repo_type="dataset")).read(), classes)
    wnids = list(classes["IMAGENET2012_CLASSES"].keys())
    k = 0
    for f in shards:
        df = pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset", local_dir="/content/hf"))
        for _, row in df.iterrows():
            folder = os.path.join("/content/val", wnids[int(row["label"])])
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, f"val_{k:05d}.JPEG"), "wb") as out:
                out.write(row["image"]["bytes"])
            k += 1
else:
    raise SystemExit("Non trovo ImageNet val in " + repo + ": hai accettato le condizioni sulla pagina del dataset?")
n = sum(len(fs) for _, _, fs in os.walk("/content/val"))
print("immagini di ImageNet val:", n)
"""),
        code("""
# 4. ImageNet-V2 (matched frequency) e ImageNet-R: pubblici; se uno non si scarica, il resto va avanti senza
import subprocess
try:
    p = hf_hub_download("vaishaal/ImageNetV2", "imagenetv2-matched-frequency.tar.gz", repo_type="dataset")
    with tarfile.open(p) as t:
        t.extractall("/content", filter="data")
    print("ImageNet-V2 ok")
except Exception as e:
    print("ImageNet-V2 non scaricato, lo salto:", e)
r = subprocess.run("wget -q https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar -O /content/imagenet-r.tar"
                   " && tar -xf /content/imagenet-r.tar -C /content && rm /content/imagenet-r.tar", shell=True)
print("ImageNet-R ok" if r.returncode == 0 else "ImageNet-R non scaricato, lo salto")
"""),
        code("""
# 5. Un blocco di ImageNet train per allenare le teste: il primo archivio, al massimo 200.000 immagini
import shutil
print(f"spazio libero: {shutil.disk_usage('/content').free / 1e9:.0f} GB")
os.makedirs("/content/train", exist_ok=True)
train_tars = sorted(f for f in files if "train" in f.lower() and f.endswith((".tar.gz", ".tar")))
train_shards = sorted(f for f in files if "train" in f.lower() and f.endswith(".parquet"))
k = 0
if train_tars:
    path = hf_hub_download(repo, train_tars[0], repo_type="dataset", local_dir="/content/hf")
    with tarfile.open(path) as t:
        for m in t:
            if m.isfile() and m.name.lower().endswith((".jpeg", ".jpg")):
                m.name = os.path.basename(m.name)
                t.extract(m, "/content/train", filter="data")
                k += 1
                if k >= 200000:
                    break
    os.remove(path)
elif train_shards:
    import pandas as pd
    if "wnids" not in globals():
        classes = {}
        exec(open(hf_hub_download(repo, "classes.py", repo_type="dataset")).read(), classes)
        wnids = list(classes["IMAGENET2012_CLASSES"].keys())
    for f in train_shards:
        df = pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset", local_dir="/content/hf"))
        for _, row in df.iterrows():
            with open(os.path.join("/content/train", f"train_{k:06d}_{wnids[int(row['label'])]}.JPEG"), "wb") as out:
                out.write(row["image"]["bytes"])
            k += 1
        if k >= 200000:
            break
else:
    raise SystemExit("STOP: non trovo ImageNet train in " + repo)
print("immagini di ImageNet train estratte:", k)
"""),
        code("%%writefile /content/exits.py\n" + exits),
        code("%%writefile /content/colab_imagenet.py\n" + script),
        code("""
# 8. Il calcolo: feature di ImageNet train, teste, tabelle dei 7 domini, test delle mappe
!cd /content && python colab_imagenet.py --val /content/val --train /content/train --v2 /content/imagenetv2-matched-frequency-format-val --r /content/imagenet-r --out /content/risultati
"""),
        code("""
# 9. Lo zip dei risultati (con i pesi delle teste) arriva nella cartella Download
import shutil, os
shutil.make_archive("/content/imagenet_risultati", "zip", "/content/risultati")
from google.colab import files
files.download("/content/imagenet_risultati.zip")
print("FATTO: scrivi a Claude che il file e' nella cartella Download")
"""),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    path = os.path.join(HERE, "imagenet_minimo.ipynb")
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("written", path)


if __name__ == "__main__":
    main()
