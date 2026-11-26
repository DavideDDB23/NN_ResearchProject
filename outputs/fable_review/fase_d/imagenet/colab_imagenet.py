# Phase D, entries D.05 and D.07: the Colab part of the minimal ImageNet experiment. ResNet50 IMAGENET1K_V1 (frozen)
# with 8 linear heads on pooled features (exits.py), trained on a block of ImageNet train (entry D.09; before, in D.05,
# on half of ImageNet val, which is still possible without --train). For the evaluation half of val (clean
# domain), ImageNet-V2, ImageNet-R and 4 ImageNet-C corruptions at severity 3 (generated on the fly with the official
# code, package imagecorruptions) it saves small tables for the local simulator: max softmax of exits 1..8, prediction
# of every exit, label. On Colab it also runs the offline test of the logit maps (one scale and one bias per class for
# every head, fitted on half of each shifted domain, savings at the empirical oracle threshold on the other half), so
# that only a few MB come back to the Mac. Written by Opus 5.5, 15 and 16 November 2026.
#
# Usage on Colab (the notebook imagenet_minimo.ipynb runs it):
#   python colab_imagenet.py --val /content/val --train /content/train \
#          --v2 /content/imagenetv2-matched-frequency-format-val --r /content/imagenet-r --out /content/risultati
#   --val: a folder of ImageNet val images, either flat with the WordNet id at the end of the file name
#          (ILSVRC2012_val_00000001_n01751748.JPEG) or with one subfolder per WordNet id
# Local test without data (random images; a stub corruption if imagecorruptions is missing):
#   python colab_imagenet.py --fake 96 --out /tmp/test_imagenet --device cpu

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if not hasattr(np, "float_"):  # imagecorruptions (fog) still uses np.float_, removed in NumPy 2 (first Colab run)
    np.float_ = np.float64
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exits import EXIT_BLOCKS, N_CLASSES, ResNet50Exits, mac_fractions  # noqa: E402

CORRUPTIONS = ["gaussian_noise", "defocus_blur", "fog", "jpeg_compression"]
SEVERITY, EPS = 3, 0.05
N_TRAIN_HALF, N_HELD_OUT = 25000, 2500
N_HELD_OUT_TRAIN, MIN_CLASSES = 10000, 990
WNID = re.compile(r"n\d{8}")
LAMBDAS = np.concatenate([[1.0], 1.0 - np.logspace(-4, 0, 200)])
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CROP = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
NORMALIZE = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class ValImages(Dataset):
    # ImageNet images from a folder, labels as the index of the WordNet id among the 1,000 sorted ids (the order of
    # torchvision and of its pretrained weights). The WordNet id (n + 8 digits) is taken from the subfolder or from
    # the file name, wherever it is (ILSVRC2012_val_00000001_n01751748.JPEG, n01440764_10026_n01440764.JPEG).
    # wnid_to_idx: the mapping of val, for ImageNet train; max_images: a random subset (seed 0)
    def __init__(self, root, wnid_to_idx=None, max_images=None):
        files = []
        for dirpath, _, names in os.walk(root):
            files += [os.path.join(dirpath, n) for n in names if n.lower().endswith((".jpeg", ".jpg", ".png"))]
        files.sort()

        def wnid(path):
            m = WNID.search(os.path.basename(os.path.dirname(path))) or WNID.search(os.path.basename(path))
            return m.group(0) if m else None

        ids = [wnid(f) for f in files]
        keep = [i for i, w in enumerate(ids) if w is not None and (wnid_to_idx is None or w in wnid_to_idx)]
        if max_images is not None and len(keep) > max_images:
            keep = sorted(np.random.default_rng(0).choice(keep, max_images, replace=False).tolist())
        self.files = [files[i] for i in keep]
        ids = [ids[i] for i in keep]
        self.wnids = sorted(set(ids))
        self.wnid_to_idx = wnid_to_idx or {w: i for i, w in enumerate(self.wnids)}
        self.labels = [self.wnid_to_idx[w] for w in ids]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return Image.open(self.files[i]), self.labels[i]


class Domain(Dataset):
    # ImageNet-C preprocessing: resize 256, center crop 224, an optional corruption at SEVERITY (numpy seeded with the
    # index, so the corruption is the same at every run), then normalization
    def __init__(self, source, corruption=None, fake=False):
        self.source, self.corruption, self.fake = source, corruption, fake

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        image, label = self.source[i]
        image = CROP(image.convert("RGB"))
        if self.corruption is not None:
            np.random.seed(i)
            array = np.asarray(image)
            try:
                from imagecorruptions import corrupt
                array = corrupt(array, corruption_name=self.corruption, severity=SEVERITY)
            except ImportError:
                if not self.fake:
                    raise
                array = np.clip(array + np.random.normal(0, 20, array.shape), 0, 255)  # stub, local test only
            image = Image.fromarray(np.uint8(np.clip(array, 0, 255)))
        return NORMALIZE(image), label


class Remap(Dataset):
    # a folder dataset whose folder names are not the class indices: label -> index through a table
    def __init__(self, folder, table):
        self.folder, self.table = folder, table

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, i):
        image, label = self.folder[i]
        return image, self.table[label]


def sources(args):
    # source datasets (PIL images, ImageNet class indices); a missing optional domain is None
    if args.fake:
        fake = lambda n, seed: datasets.FakeData(n, (3, 256, 320), N_CLASSES, random_offset=seed)  # noqa: E731
        train = fake(4 * args.fake, 3) if args.train else None
        return fake(2 * args.fake, 0), train, fake(args.fake, 1), fake(args.fake, 2), np.arange(200)
    val = ValImages(args.val)
    if len(val.wnids) != N_CLASSES or len(val) != 50000:
        raise SystemExit(f"ImageNet val: found {len(val)} images of {len(val.wnids)} classes, expected 50000 of 1000")
    train = None
    if args.train:
        train = ValImages(args.train, val.wnid_to_idx, args.max_train)
        n_classes = len(set(train.labels))
        print(f"ImageNet train: {len(train)} images of {n_classes} classes", flush=True)
        if n_classes < MIN_CLASSES:
            raise SystemExit(f"STOP: il blocco di ImageNet train copre solo {n_classes} classi su 1000 (gli archivi "
                             f"sono ordinati per classe). Dillo a Claude: non serve rilanciare.")
    v2 = r = r_classes = None
    if args.v2 and os.path.isdir(args.v2):
        folder = datasets.ImageFolder(args.v2)       # folders named 0..999 = class index
        v2 = Remap(folder, [int(c) for c in folder.classes])
    else:
        print("ImageNet-V2 not found: skipped", flush=True)
    if args.r and os.path.isdir(args.r):
        folder = datasets.ImageFolder(args.r)        # folders named by WordNet id, 200 classes
        table = [val.wnid_to_idx[w] for w in folder.classes]
        r, r_classes = Remap(folder, table), np.array(sorted(table))
    else:
        print("ImageNet-R not found: skipped", flush=True)
    return val, train, v2, r, r_classes


def halves(n, fake):
    perm = np.random.default_rng(0).permutation(n)
    k = n // 2 if fake else N_TRAIN_HALF
    return perm[:k], perm[k:]


def loader(dataset, args):
    return DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                      pin_memory=args.device == "cuda")


def autocast(args):
    return torch.autocast("cuda", dtype=torch.float16, enabled=args.device == "cuda")


@torch.no_grad()
def extract(model, data, args):
    # pooled features of the 8 exits in one preallocated float16 array (200,000 images: about 3.4 GB), and labels
    dim = sum(h.in_features for h in model.heads)
    feats, labels, i = np.empty((len(data), dim), np.float16), np.empty(len(data), np.int64), 0
    for x, y in loader(data, args):
        with autocast(args):
            f, _ = model.features(x.to(args.device, non_blocking=True))
        feats[i:i + len(y)] = torch.cat(f, dim=1).half().cpu().numpy()
        labels[i:i + len(y)] = y.numpy()
        i += len(y)
        if i % 20480 < len(y):
            print(f"   features: {i} of {len(data)}", flush=True)
    return feats, labels


def train_heads(model, feats, labels, args):
    # 8 linear heads on the cached features: Adam, the number of epochs chosen per head on held-out images of the
    # training half
    dims = [h.in_features for h in model.heads]
    ends = np.cumsum(dims)
    n_held = (N_HELD_OUT_TRAIN if args.train else N_HELD_OUT) if not args.fake else len(labels) // 5
    perm = torch.as_tensor(np.random.default_rng(0).permutation(len(labels)))
    store = args.device if args.device == "cuda" else "cpu"   # the float16 features fit on a 16 GB GPU
    x = torch.as_tensor(feats).to(store)
    y = torch.as_tensor(labels, dtype=torch.long).to(store)
    xt, yt, xv, yv = x[perm[n_held:]], y[perm[n_held:]], x[perm[:n_held]], y[perm[:n_held]]
    del x
    log = []
    for k, head in enumerate(model.heads):
        a, b = ends[k] - dims[k], ends[k]
        head = head.to(args.device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state, best_epoch = -1.0, None, 0
        g = torch.Generator().manual_seed(k)
        for epoch in range(1, args.epochs + 1):
            order = torch.randperm(len(yt), generator=g).to(store)
            for i in range(0, len(order), 512):
                idx = order[i:i + 512]
                loss = F.cross_entropy(head(xt[idx, a:b].to(args.device).float()), yt[idx].to(args.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                acc = (head(xv[:, a:b].to(args.device).float()).argmax(1) == yv.to(args.device)).float().mean().item()
            if acc > best:
                best, best_epoch = acc, epoch
                best_state = {n: p.detach().cpu().clone() for n, p in head.state_dict().items()}
        head.load_state_dict(best_state)
        log.append({"exit": k + 1, "block": EXIT_BLOCKS[k], "held_out_accuracy": round(best, 4), "epochs": best_epoch})
        print(f"head {k + 1} (block {EXIT_BLOCKS[k]}): held-out accuracy {best:.4f} at epoch {best_epoch}", flush=True)
    return log


@torch.no_grad()
def logits_of(model, data, args):
    out, labels = [], []
    for x, y in loader(data, args):
        with autocast(args):
            out.append(model(x.to(args.device, non_blocking=True)).half().cpu())
        labels.append(y)
    return torch.cat(out), torch.cat(labels).numpy()


def tables(logits, scale=None, bias=None):
    # max softmax of exits 1..8 and prediction of every exit, with optional per-class maps on the heads
    z = logits.float()
    if scale is not None:
        z = torch.cat([z[:, :8] * scale[None] + bias[None], z[:, 8:]], dim=1)
    p = F.softmax(z, dim=2)
    return p[:, :8].max(dim=2).values, z.argmax(dim=2)


def oracle_savings(scores, preds, fr):
    # the largest savings with empirical consistency risk <= EPS over the threshold grid (lambda = 1 sends everything
    # to exit 9, risk 0, so the set is never empty)
    best = -float("inf")
    for lam in LAMBDAS:
        above = scores > float(lam)
        e = torch.where(above.any(dim=1), above.float().argmax(dim=1), torch.full_like(above[:, 0], 8, dtype=torch.long))
        risk = (preds.gather(1, e[:, None])[:, 0] != preds[:, 8]).float().mean().item()
        if risk <= EPS:
            best = max(best, 1.0 - fr[e].mean().item())
    return best


def fit_diag(logits, max_iter=100):
    # one scale and one bias per class for every head, distillation towards the softmax of exit 9 (as in sim.py)
    teacher = F.softmax(logits[:, 8].float(), dim=1)
    c = logits.shape[2]
    scales, biases = torch.ones(8, c, device=logits.device), torch.zeros(8, c, device=logits.device)
    for l in range(8):
        z = logits[:, l].float()
        d = torch.ones(c, device=logits.device, requires_grad=True)
        a = torch.zeros(c, device=logits.device, requires_grad=True)
        opt = torch.optim.LBFGS([d, a], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.kl_div(F.log_softmax(z * d + a, dim=1), teacher, reduction="batchmean")
            loss.backward()
            return loss

        opt.step(closure)
        scales[l], biases[l] = d.detach(), a.detach()
    return scales, biases


def maps_test(logits, fr, device):
    # fit on one half of the domain (all of it, and only 3,000 rows as in the online phase), savings at the oracle
    # threshold on the other half with and without the maps, both ways
    perm = np.random.default_rng(1).permutation(len(logits))
    halves_ = [torch.as_tensor(perm[:len(perm) // 2]), torch.as_tensor(perm[len(perm) // 2:])]
    out = {"plain": [], "maps_half": [], "maps_3000": []}
    for fit_idx, ev_idx in [(halves_[0], halves_[1]), (halves_[1], halves_[0])]:
        ev = logits[ev_idx].to(device)
        s, p = tables(ev)
        out["plain"].append(oracle_savings(s, p, fr))
        for key, rows in [("maps_half", fit_idx), ("maps_3000", fit_idx[:3000])]:
            scale, bias = fit_diag(logits[rows].to(device))
            s, p = tables(ev, scale, bias)
            out[key].append(oracle_savings(s, p, fr))
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", default="/content/val")
    p.add_argument("--train", default=None, help="folder of ImageNet train images for the heads (entry D.09)")
    p.add_argument("--max-train", type=int, default=200000)
    p.add_argument("--v2", default="/content/imagenetv2-matched-frequency-format-val")
    p.add_argument("--r", default="/content/imagenet-r")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--fake", type=int, default=0, help="local test: this many random images per domain")
    args = p.parse_args()
    if args.fake:
        args.workers, args.batch, args.epochs = 0, 16, 2
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    start = time.time()
    torch.manual_seed(0)  # the initial weights of the heads (added after the first Colab run, which had no seed)
    model = ResNet50Exits(pretrained=not args.fake).eval()
    fr_list, _ = mac_fractions(model)
    fr = torch.tensor(fr_list, device=args.device)
    model.to(args.device)
    val, train, v2, r, r_classes = sources(args)
    train_idx, eval_idx = halves(len(val), args.fake)
    np.save(os.path.join(args.out, "split_eval_indices.npy"), eval_idx)
    if r_classes is not None:
        np.save(os.path.join(args.out, "r_classes.npy"), r_classes)

    heads_path = os.path.join(args.out, "heads.pt")
    if not os.path.exists(heads_path):
        t = time.time()
        heads_data = Domain(train) if train is not None else Domain(Subset(val, train_idx))
        feats, labels = extract(model, heads_data, args)
        print(f"features for the heads: {feats.shape} in {time.time() - t:.0f} s", flush=True)
        log = train_heads(model, feats, labels, args)
        torch.save({n: v.cpu() for n, v in model.heads.state_dict().items()}, heads_path)
        with open(os.path.join(args.out, "heads_log.json"), "w") as f:
            json.dump(log, f, indent=1)
        del feats
    model.heads.load_state_dict(torch.load(heads_path, map_location="cpu"))
    model.to(args.device).eval()

    domains = {"clean": Domain(Subset(val, eval_idx))}
    if v2 is not None:
        domains["v2"] = Domain(v2)
    if r is not None:
        domains["r"] = Domain(r)
    for c in CORRUPTIONS:
        domains[f"{c}_{SEVERITY}"] = Domain(Subset(val, eval_idx), corruption=c, fake=bool(args.fake))
    for name, data in domains.items():
        path = os.path.join(args.out, "tables", f"{name}.npz")
        if os.path.exists(path):
            continue
        t = time.time()
        logits, labels = logits_of(model, data, args)
        scores, preds = tables(logits.to(args.device))
        acc_logits = logits.float()
        if name == "r":  # ImageNet-R is scored on its 200 classes
            mask = torch.full((N_CLASSES,), float("-inf"))
            mask[torch.as_tensor(r_classes)] = 0
            acc_logits = acc_logits + mask
        acc = (acc_logits.argmax(-1).numpy() == labels[:, None]).mean(axis=0)
        summary = {"n": int(len(labels)), "seconds": round(time.time() - t, 1),
                   "accuracy_per_exit": [round(float(a), 4) for a in acc],
                   "oracle_savings": round(oracle_savings(scores, preds, fr), 4)}
        if name != "clean":
            summary["maps"] = maps_test(logits, fr, args.device)
        np.savez_compressed(path, scores=scores.cpu().numpy().astype(np.float16),
                            preds=preds.cpu().numpy().astype(np.int16), labels=labels.astype(np.int16))
        with open(os.path.join(args.out, f"summary_{name}.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(f"{name}: {summary}", flush=True)
        del logits, acc_logits
    with open(os.path.join(args.out, "mac_fractions.json"), "w") as f:
        json.dump([round(float(v), 4) for v in fr_list], f)
    print(f"done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
