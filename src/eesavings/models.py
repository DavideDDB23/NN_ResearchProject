# Multi exit ResNet20 for CIFAR.
#
# Backbone: the CIFAR ResNet20 of He et al. 2016 with option A shortcuts, copied from Pomponi et al. 2022
# (jaryP/ConfidenceBranchNetwok, models/resnet.py, commit 405d273), who take it from Y. Idelbayev
# (pytorch_resnet_cifar10). Same layer names (conv1, bn1, layer1..3), so their weights can be loaded.
# Changes: the forward does not compute costs, and there is no exit on the first convolution.
#
# Exits: one after each of the 9 residual blocks.
#   - exits 1..8: pooled head, the default head of Kubaty et al. 2025
#     (kamadforge/early-exit-benchmark, architectures/early_exits/heads.py, "pooled_head" = SDN pooling):
#     a learnable mix of max and average pooling down to 4x4, flatten, one linear layer.
#     We add a confidence unit (one linear output and a sigmoid) on the same features, as in Pomponi et al. 2022.
#   - exit 9: the usual global average pooling and linear layer, so the backbone alone is a standard ResNet20.

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.utils.flop_counter import FlopCounterMode

N_EXITS = 9
# size of the features of the pooled heads of exits 1..8 (channels x 4 x 4)
FEATURE_DIMS = [16 * 16] * 3 + [32 * 16] * 3 + [64 * 16] * 2


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # option A: subsample and pad the missing channels with zeros (no parameters)
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class PooledHead(nn.Module):
    # SDN pooling: alpha * maxpool + (1 - alpha) * avgpool down to 4x4, with alpha learnable and not constrained,
    # as in Kubaty et al. Fixed kernels instead of adaptive pooling: same result for 32, 16 and 8 pixels.
    def __init__(self, channels, size, num_classes):
        super().__init__()
        self.alpha = nn.Parameter(torch.rand(1))
        self.max_pool = nn.MaxPool2d(size // 4)
        self.avg_pool = nn.AvgPool2d(size // 4)
        self.classifier = nn.Linear(channels * 16, num_classes)
        self.confidence = nn.Linear(channels * 16, 1)

    def features(self, x):
        mixed = self.alpha * self.max_pool(x) + (1 - self.alpha) * self.avg_pool(x)
        return mixed.flatten(1)

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f), torch.sigmoid(self.confidence(f)).squeeze(1), f


class MultiExitResNet20(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.ModuleList([BasicBlock(16, 16), BasicBlock(16, 16), BasicBlock(16, 16)])
        self.layer2 = nn.ModuleList([BasicBlock(16, 32, stride=2), BasicBlock(32, 32), BasicBlock(32, 32)])
        self.layer3 = nn.ModuleList([BasicBlock(32, 64, stride=2), BasicBlock(64, 64), BasicBlock(64, 64)])
        self.fc = nn.Linear(64, num_classes)  # exit 9
        self.apply(_weights_init)

        # heads of exits 1..8: (channels, spatial size) after each block
        shapes = [(16, 32)] * 3 + [(32, 16)] * 3 + [(64, 8)] * 2
        self.heads = nn.ModuleList([PooledHead(c, s, num_classes) for c, s in shapes])

        self.backbone_frozen = False

    def blocks(self):
        return list(self.layer1) + list(self.layer2) + list(self.layer3)

    def forward(self, x, return_features=False):
        # logits (B, 9, C) and confidences (B, 9); the confidence of exit 9 is 1 by convention.
        # With return_features=True also the list of the 8 head features (B, 256 / 512 / 1024).
        out = F.relu(self.bn1(self.conv1(x)))
        logits, confs, feats = [], [], []
        for l, block in enumerate(self.blocks()):
            out = block(out)
            if l < N_EXITS - 1:
                z, c, f = self.heads[l](out)
                logits.append(z)
                confs.append(c)
                feats.append(f)
        pooled = F.avg_pool2d(out, out.size(3)).flatten(1)
        logits.append(self.fc(pooled))
        confs.append(torch.ones_like(confs[0]))

        logits, confs = torch.stack(logits, dim=1), torch.stack(confs, dim=1)
        if return_features:
            return logits, confs, feats
        return logits, confs

    # parameters of the backbone (exit 9 included) and of the heads of exits 1..8
    def backbone_modules(self):
        return [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.fc]

    def backbone_parameters(self):
        return [p for m in self.backbone_modules() for p in m.parameters()]

    def head_parameters(self):
        return list(self.heads.parameters())

    def freeze_backbone(self):
        # used in phase 3: no gradient for the backbone and its BatchNorm statistics stay fixed
        self.backbone_frozen = True
        for p in self.backbone_parameters():
            p.requires_grad = False
        self.train(self.training)

    def train(self, mode=True):
        # model.train() would put the BatchNorm of a frozen backbone back in training mode: keep it in eval
        super().train(mode)
        if self.backbone_frozen:
            for m in self.backbone_modules():
                m.eval()
        return self


def exit_macs(model):
    # multiply-accumulates (MACs) of each block and head for one 32x32 image, counted on CPU with
    # FlopCounterMode, which counts only convolutions and linear layers (FLOPs = 2 MACs).
    # Pooling, BatchNorm, ReLU and additions are not counted.
    # Returns block_macs (9,), head_macs (8,) and fc_macs (exit 9). Works on a copy of the model.
    model = copy.deepcopy(model).to("cpu").eval()
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))
    counts = {name: sum(c.values()) // 2 for name, c in counter.get_flop_counts().items()}
    prefix = type(model).__name__

    block_names = [f"layer{s}.{b}" for s in (1, 2, 3) for b in range(3)]
    block_macs = [counts[f"{prefix}.{n}"] for n in block_names]
    block_macs[0] += counts[f"{prefix}.conv1"]  # the first convolution is paid by every input
    head_macs = [counts[f"{prefix}.heads.{l}"] for l in range(N_EXITS - 1)]
    fc_macs = counts[f"{prefix}.fc"]

    assert sum(block_macs) + sum(head_macs) + fc_macs == counts["Global"]
    return block_macs, head_macs, fc_macs


def exit_costs(model):
    # cumulative cost (MACs) of each exit: an input that leaves at exit l has run the blocks 1..l
    # and evaluated the heads 1..l. If no head fires the input pays all heads and the final layer.
    # full_cost is the plain network without intermediate heads (backbone and final layer): the reference
    # of the savings.
    block_macs, head_macs, fc_macs = exit_macs(model)
    costs = []
    for l in range(N_EXITS):
        cost = sum(block_macs[:l + 1]) + sum(head_macs[:min(l + 1, N_EXITS - 1)])
        if l == N_EXITS - 1:
            cost += fc_macs
        costs.append(cost)
    full_cost = sum(block_macs) + fc_macs
    return costs, full_cost
