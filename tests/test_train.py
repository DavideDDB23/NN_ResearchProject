import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import eesavings.train as train_module
from eesavings.models import MultiExitResNet20
from eesavings.train import (joint_loss, load_model, phase_loss, self_distillation_loss, train_phase,
                             train_regime)


def tiny_loaders(n=64, num_classes=10):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    data = TensorDataset(x, y)
    return DataLoader(data, batch_size=16, shuffle=True, generator=g), DataLoader(data, batch_size=32)


def split_state(model):
    state = {k: v.clone() for k, v in model.state_dict().items()}
    heads = {k: v for k, v in state.items() if k.startswith("heads")}
    backbone = {k: v for k, v in state.items() if not k.startswith("heads")}
    return backbone, heads


def test_joint_loss_is_the_mean_of_the_exit_losses():
    logits, y = torch.randn(8, 9, 10), torch.randint(0, 10, (8,))
    expected = sum(F.cross_entropy(logits[:, l], y) for l in range(9)) / 9
    assert torch.isclose(joint_loss(logits, y), expected)


def test_self_distillation_teacher_is_detached():
    # the KL terms must not push exit 9: its gradient is the one of its cross entropy only
    y = torch.randint(0, 10, (8,))
    logits = torch.randn(8, 9, 10, requires_grad=True)
    self_distillation_loss(logits, y).backward()
    logits2 = logits.detach().clone().requires_grad_(True)
    (F.cross_entropy(logits2[:, -1], y) / 9).backward()
    assert torch.allclose(logits.grad[:, -1], logits2.grad[:, -1], atol=1e-7)


def test_self_distillation_matches_a_hand_computation():
    torch.manual_seed(0)
    logits, y = torch.randn(4, 3, 5), torch.randint(0, 5, (4,))
    alpha, tau = 0.5, 3.0

    def softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    z = logits.numpy()
    p_teacher = softmax(z[:, 2] / tau)
    total = 0.0
    for l in range(2):
        p_student = softmax(z[:, l] / tau)
        kl = np.mean(np.sum(p_teacher * (np.log(p_teacher) - np.log(p_student)), axis=1))
        total += (1 - alpha) * F.cross_entropy(logits[:, l], y).item() + alpha * tau ** 2 * kl
    total += F.cross_entropy(logits[:, 2], y).item()
    assert np.isclose(self_distillation_loss(logits, y, alpha, tau).item(), total / 3, atol=1e-5)


def test_self_distillation_without_disagreement_is_only_cross_entropy():
    z, y = torch.randn(4, 10), torch.randint(0, 10, (4,))
    logits = z[:, None].repeat(1, 9, 1)  # every exit equals the teacher, so every KL is zero
    assert torch.isclose(self_distillation_loss(logits, y, alpha=1.0), F.cross_entropy(z, y) / 9, atol=1e-6)


def test_heads_phase_leaves_the_backbone_untouched():
    torch.manual_seed(0)
    model = MultiExitResNet20(10)
    train_loader, val_loader = tiny_loaders()
    train_phase(model, "backbone", train_loader, val_loader, 1, "cpu")  # BatchNorm statistics are not the initial ones
    backbone_before, heads_before = split_state(model)
    x = next(iter(val_loader))[0]
    exit9_before = model.eval()(x)[0][:, -1].detach()

    # two real epochs, with the val evaluation (model.eval() then model.train()) in between
    train_phase(model, "heads", train_loader, val_loader, 2, "cpu")
    backbone_after, heads_after = split_state(model)
    for k, v in backbone_before.items():  # weights, running mean and var, num_batches_tracked
        assert torch.equal(v, backbone_after[k]), k
    assert torch.equal(exit9_before, model.eval()(x)[0][:, -1])
    changed = [k for k, v in heads_before.items() if not torch.equal(v, heads_after[k])]
    assert any("classifier" in k for k in changed) and any("alpha" in k for k in changed)
    assert not any("confidence" in k for k in changed)  # the confidence units are fitted later, on val


def test_backbone_phase_leaves_the_heads_untouched():
    torch.manual_seed(0)
    model = MultiExitResNet20(10)
    backbone_before, heads_before = split_state(model)
    train_phase(model, "backbone", *tiny_loaders(), 1, "cpu")
    backbone_after, heads_after = split_state(model)
    assert all(torch.equal(v, heads_after[k]) for k, v in heads_before.items())
    assert not torch.equal(backbone_before["fc.weight"], backbone_after["fc.weight"])
    assert not torch.equal(backbone_before["layer1.0.conv1.weight"], backbone_after["layer1.0.conv1.weight"])


def test_disjoint_and_mixed_reuse_the_saved_backbone(tmp_path, monkeypatch):
    train_loader, val_loader = tiny_loaders()
    torch.manual_seed(0)

    def make():
        return MultiExitResNet20(10)

    backbone_dir = tmp_path / "backbone"
    model = train_regime(make, "disjoint", train_loader, val_loader, 1, "cpu",
                         str(tmp_path / "disjoint"), str(backbone_dir))
    # disjoint: the backbone of the final model is exactly the saved one
    saved, _ = split_state(load_model(make(), str(backbone_dir / "model.pt")))
    final, _ = split_state(model)
    assert all(torch.equal(v, final[k]) for k, v in saved.items())

    # mixed: phase 2 starts from the backbone already on disk (checked at the start of the phase),
    # and the backbone is not trained again
    starts = []

    def checked_train_phase(model, phase, *args, **kwargs):
        start, _ = split_state(model.cpu())
        starts.append((phase, all(torch.equal(v, start[k]) for k, v in saved.items())))
        return train_phase(model, phase, *args, **kwargs)

    mtime = os.path.getmtime(backbone_dir / "model.pt")
    monkeypatch.setattr(train_module, "train_phase", checked_train_phase)
    train_regime(make, "mixed", train_loader, val_loader, 1, "cpu", str(tmp_path / "mixed"), str(backbone_dir))
    assert starts == [("all", True)]
    assert os.path.getmtime(backbone_dir / "model.pt") == mtime

    # a finished run is not trained again
    before = (tmp_path / "mixed" / "model.pt").read_bytes()
    train_regime(make, "mixed", train_loader, val_loader, 1, "cpu", str(tmp_path / "mixed"), str(backbone_dir))
    assert (tmp_path / "mixed" / "model.pt").read_bytes() == before


def test_each_phase_uses_the_right_loss():
    logits, y = torch.randn(8, 9, 10), torch.randint(0, 10, (8,))
    assert torch.isclose(phase_loss("backbone", logits, y), F.cross_entropy(logits[:, -1], y))
    assert torch.isclose(phase_loss("all", logits, y), joint_loss(logits, y))
    assert torch.isclose(phase_loss("all_sd", logits, y), self_distillation_loss(logits, y))
    assert torch.isclose(phase_loss("heads", logits, y), joint_loss(logits, y, exits=range(8)))  # exit 9 is frozen


def test_every_regime_gets_the_same_budget(tmp_path):
    # joint and joint_sd train 2 x epochs in one phase, disjoint and mixed 1 x epochs after the 1 x epochs backbone
    train_loader, val_loader = tiny_loaders()
    torch.manual_seed(0)
    for regime in ["joint", "joint_sd", "disjoint", "mixed"]:
        train_regime(lambda: MultiExitResNet20(10), regime, train_loader, val_loader, 2, "cpu",
                     str(tmp_path / regime), str(tmp_path / "backbone"), seed=0)
    regimes = ["joint", "joint_sd", "disjoint", "mixed", "backbone"]
    rows = {r: len(pd.read_csv(tmp_path / r / "log.csv")) for r in regimes}
    assert rows == {"joint": 4, "joint_sd": 4, "disjoint": 2, "mixed": 2, "backbone": 2}
    # the lr is divided at 50% and 75% of the phase, and a 2 epoch phase starts at lr 0.1
    lr = pd.read_csv(tmp_path / "joint" / "log.csv").lr.round(6).tolist()
    assert lr == [0.1, 0.1, 0.01, 0.001]


def test_each_phase_has_its_own_random_stream():
    # the same phase with the same seed gives the same network, whatever ran before it in the process
    torch.manual_seed(0)
    start = MultiExitResNet20(10).state_dict()
    a, b = MultiExitResNet20(10), MultiExitResNet20(10)
    a.load_state_dict(start)
    b.load_state_dict(start)
    loaders = tiny_loaders()  # the same loaders for all phases, as in one training process
    train_phase(a, "all", *loaders, 1, "cpu", seed=3)
    train_phase(MultiExitResNet20(10), "backbone", *loaders, 1, "cpu", seed=3)  # moves the random state on
    train_phase(b, "all", *loaders, 1, "cpu", seed=3)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
