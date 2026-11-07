import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import ROOT
from eesavings.models import FEATURE_DIMS, N_EXITS, MultiExitResNet20, exit_costs, exit_macs
from eesavings.savings import RESNET20_CIFAR_CUM_MMACS, RESNET20_EXIT_MACS, RESNET20_FULL_MACS

POMPONI = ROOT / "third_party" / "ConfidenceBranchNetwok"
# MACs of a CIFAR ResNet20 after each block (first convolution included), computed by hand
BACKBONE_CUM_MACS = [5160960, 9879552, 14598144, 18137088, 22855680, 27574272, 31113216, 35831808, 40550400]


def make_model(num_classes=10, seed=0):
    torch.manual_seed(seed)
    return MultiExitResNet20(num_classes).eval()


def test_output_shapes():
    model = make_model(100)
    logits, conf, feats = model(torch.randn(5, 3, 32, 32), return_features=True)
    assert logits.shape == (5, 9, 100) and conf.shape == (5, 9)
    assert [f.shape[1] for f in feats] == FEATURE_DIMS and sum(FEATURE_DIMS) == 4352
    assert torch.all(conf[:, -1] == 1) and torch.all((conf >= 0) & (conf <= 1))


def test_exit_l_reads_block_l_and_not_the_later_ones():
    # changing block l (0-indexed) must leave the exits before it unchanged and change exit l
    x = torch.randn(4, 3, 32, 32)
    for l in range(N_EXITS):
        model = make_model()
        before = model(x)[0]
        with torch.no_grad():
            for p in model.blocks()[l].parameters():
                p.add_(1.0)
        after = model(x)[0]
        assert torch.allclose(before[:, :l], after[:, :l]), l
        assert not torch.allclose(before[:, l], after[:, l]), l


@pytest.mark.skipif(not POMPONI.exists(), reason="third_party/ConfidenceBranchNetwok not cloned")
def test_backbone_matches_the_resnet20_of_pomponi():
    sys.path.insert(0, str(POMPONI))
    try:
        from models.resnet import resnet20
    finally:
        sys.path.remove(str(POMPONI))
    theirs = resnet20().eval()
    ours = make_model()
    backbone = {k: v for k, v in ours.state_dict().items() if not k.startswith(("heads", "fc"))}
    missing, unexpected = theirs.load_state_dict(backbone, strict=False)
    assert not missing and not unexpected

    outputs = []
    hooks = [b.register_forward_hook(lambda m, i, o: outputs.append(o)) for b in ours.blocks()]
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        ours(x)
        their_outputs = theirs(x)
    for h in hooks:
        h.remove()
    # their list: first convolution, blocks 1..8, then the pooled output of block 9
    for l in range(8):
        assert torch.allclose(outputs[l], their_outputs[l + 1], atol=1e-5), l
    assert torch.allclose(F.avg_pool2d(outputs[8], 8), their_outputs[9], atol=1e-5)


@pytest.mark.parametrize("num_classes", [10, 100])
def test_macs_of_blocks_heads_and_exits(num_classes):
    model = make_model(num_classes)
    block_macs, head_macs, fc_macs = exit_macs(model)
    assert np.cumsum(block_macs).tolist() == BACKBONE_CUM_MACS
    assert np.allclose(np.array(BACKBONE_CUM_MACS) / 1e6, RESNET20_CIFAR_CUM_MMACS, atol=0.005)
    assert fc_macs == 64 * num_classes
    # a pooled head is the classifier (d x C) plus the confidence unit (d)
    assert head_macs == [d * num_classes + d for d in FEATURE_DIMS]

    costs, full = exit_costs(model)
    assert np.all(np.diff(costs) > 0)
    assert full == BACKBONE_CUM_MACS[-1] + fc_macs
    assert costs[0] == BACKBONE_CUM_MACS[0] + head_macs[0]
    assert costs[-1] == full + sum(head_macs)  # no head fired: all heads were evaluated
    dataset = "cifar10" if num_classes == 10 else "cifar100"
    assert costs == RESNET20_EXIT_MACS[dataset].tolist() and full == RESNET20_FULL_MACS[dataset]


def test_frozen_backbone_stays_in_eval_mode():
    model = make_model()
    model.freeze_backbone()
    model.train()
    assert not model.bn1.training and not model.layer2[0].bn1.training and not model.fc.training
    assert model.heads[0].training
    assert all(not p.requires_grad for p in model.backbone_parameters())
    assert all(p.requires_grad for p in model.head_parameters())


def test_mac_counting_does_not_change_the_model():
    model = make_model().train()
    exit_macs(model)
    assert model.training and next(model.parameters()).device.type == "cpu"
