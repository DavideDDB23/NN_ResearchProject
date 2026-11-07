# Training of the multi exit ResNet20 with the regimes of Kubaty et al. 2025 (section 2):
#   phase "backbone": backbone and exit 9 only, cross entropy of the final exit
#   phase "all":      the whole network, mean of the cross entropies of the 9 exits
#   phase "all_sd":   as "all", with self distillation from exit 9 to the other exits
#   phase "heads":    backbone and exit 9 frozen (BatchNorm included), mean of the cross entropies of exits 1..8
# Regimes: joint = all, joint_sd = all_sd, disjoint = backbone + heads, mixed = backbone + all.
# Every regime gets the same total number of epochs (2 x epochs), as in scripts/regimes_resnet.py of
# Kubaty et al.; the backbone phase is shared by disjoint and mixed of the same seed.
# Recipe of He et al. 2016 for every phase: SGD with lr 0.1, momentum 0.9, weight decay 1e-4, batch 128,
# lr divided by 10 at 50% and 75% of the phase. The model of the last epoch is kept.

import csv
import os
import time

import torch
import torch.nn.functional as F

REGIMES = ["joint", "disjoint", "mixed", "joint_sd"]
PHASES = ["backbone", "all", "all_sd", "heads"]

LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# self distillation (Zhang et al. 2019 use T = 3; the tau^2 factor is from Hinton et al. and Phuong and Lampert 2019)
SD_ALPHA = 0.5
SD_TAU = 3.0


# ---------------- losses ----------------

def joint_loss(logits, y, exits=None):
    # mean of the cross entropies of the chosen exits (all of them by default); logits (B, L, C)
    if exits is None:
        exits = range(logits.shape[1])
    return torch.stack([F.cross_entropy(logits[:, l], y) for l in exits]).mean()


def self_distillation_loss(logits, y, alpha=SD_ALPHA, tau=SD_TAU):
    # exits 1..8: (1 - alpha) * CE + alpha * tau^2 * KL(teacher || exit), the teacher is exit 9 and is detached;
    # exit 9: CE only. Mean over the 9 exits, like joint_loss.
    teacher = F.softmax(logits[:, -1].detach() / tau, dim=1)
    losses = []
    for l in range(logits.shape[1] - 1):
        ce = F.cross_entropy(logits[:, l], y)
        kl = F.kl_div(F.log_softmax(logits[:, l] / tau, dim=1), teacher, reduction="batchmean")
        losses.append((1 - alpha) * ce + alpha * tau ** 2 * kl)
    losses.append(F.cross_entropy(logits[:, -1], y))
    return torch.stack(losses).mean()


def phase_loss(phase, logits, y):
    n_exits = logits.shape[1]
    if phase == "backbone":
        return F.cross_entropy(logits[:, -1], y)
    if phase == "all":
        return joint_loss(logits, y)
    if phase == "all_sd":
        return self_distillation_loss(logits, y)
    if phase == "heads":
        return joint_loss(logits, y, exits=range(n_exits - 1))
    raise ValueError(phase)


def phase_parameters(model, phase):
    # parameters updated in each phase (the confidence units get no gradient here: they are fitted later)
    if phase == "backbone":
        return model.backbone_parameters()
    if phase == "heads":
        return model.head_parameters()
    return list(model.parameters())


# ---------------- training loop ----------------

def accuracy_per_exit(model, loader, device):
    model.eval()
    correct, total = None, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            hits = (logits.argmax(dim=2) == y[:, None]).sum(dim=0)
            correct = hits if correct is None else correct + hits
            total += len(y)
    return (correct.float() / total).tolist()


def train_phase(model, phase, train_loader, val_loader, epochs, device, log_path=None, seed=None):
    # trains `model` for `epochs` epochs; after each epoch evaluates every exit on the val set and logs it
    assert phase in PHASES, phase
    if seed is not None:
        # each phase gets its own random stream, otherwise phase 2 of mixed would replay exactly the batches of
        # phase 1. The order of the batches always depends only on (seed, phase). The augmentations too, except
        # when one process trains two phases with persistent DataLoader workers (disjoint or mixed started by hand
        # without a backbone): the workers of phase 1 then keep their random state. run_trainings.py avoids this by
        # training the backbone as a job of its own.
        phase_seed = 1000 * seed + PHASES.index(phase)
        torch.manual_seed(phase_seed)
        if getattr(train_loader, "generator", None) is not None:
            train_loader.generator.manual_seed(phase_seed)
    if phase == "heads":
        model.freeze_backbone()  # no gradient for the backbone, and its BatchNorm stays in eval mode
    params = [p for p in phase_parameters(model, phase) if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # lr divided by 10 at 50% and 75% of the phase (at least after the first epoch, for very short test runs)
    milestones = [max(1, epochs // 2), max(1, (3 * epochs) // 4)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    rows = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = phase_loss(phase, logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_time = time.time() - start

        val_acc = accuracy_per_exit(model, val_loader, device)
        row = {"phase": phase, "epoch": epoch, "lr": lr, "train_loss": total_loss / n,
               "train_time": round(train_time, 2), "epoch_time": round(time.time() - start, 2)}
        row.update({f"val_acc_{l + 1}": round(a, 4) for l, a in enumerate(val_acc)})
        rows.append(row)
        if log_path is not None:
            write_log(rows, log_path)
        print(f"{phase} epoch {epoch}/{epochs} loss {row['train_loss']:.3f} "
              f"val acc exit 1 {val_acc[0]:.3f} exit 9 {val_acc[-1]:.3f} ({row['epoch_time']:.0f}s)", flush=True)
    return rows


def write_log(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_model(model, path):
    # write to a temporary file first, so an interrupted run never leaves a half written model.pt
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def train_backbone(make_model, train_loader, val_loader, epochs, device, backbone_dir, seed=None):
    # phase 1, shared by disjoint and mixed of the same seed: trained once and saved in backbone_dir
    backbone_path = os.path.join(backbone_dir, "model.pt")
    if os.path.exists(backbone_path):
        print("skip", backbone_dir, "(already trained)")
        return
    os.makedirs(backbone_dir, exist_ok=True)
    model = make_model().to(device)
    train_phase(model, "backbone", train_loader, val_loader, epochs, device, os.path.join(backbone_dir, "log.csv"),
                seed=seed)
    save_model(model, backbone_path)


def train_regime(make_model, regime, train_loader, val_loader, epochs, device, run_dir, backbone_dir, seed=None):
    # trains one regime and saves run_dir/model.pt; a model.pt already present means the work is done.
    # make_model() must return a freshly initialized network on the cpu (seeded by the caller).
    assert regime in REGIMES, regime
    model_path = os.path.join(run_dir, "model.pt")
    if os.path.exists(model_path):
        print("skip", run_dir, "(already trained)")
        return load_model(make_model(), model_path)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.csv")

    if regime in ("joint", "joint_sd"):
        model = make_model().to(device)
        phase = "all" if regime == "joint" else "all_sd"
        train_phase(model, phase, train_loader, val_loader, 2 * epochs, device, log_path, seed=seed)
    else:
        train_backbone(make_model, train_loader, val_loader, epochs, device, backbone_dir, seed=seed)
        model = load_model(make_model(), os.path.join(backbone_dir, "model.pt")).to(device)
        phase = "heads" if regime == "disjoint" else "all"
        train_phase(model, phase, train_loader, val_loader, epochs, device, log_path, seed=seed)

    save_model(model, model_path)
    return model
