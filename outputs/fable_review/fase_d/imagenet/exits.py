# Phase D, entry D.05: ResNet50 of torchvision with 8 early exits for the ImageNet experiment, shared by the Colab
# script (features, heads, logits) and by the local measures (MACs, latency). Written by Opus 5.5, 15 November 2026.
#   - backbone: torchvision resnet50, weights IMAGENET1K_V1 on Colab (pretrained=False locally: random weights, the
#     same compute), frozen
#   - exits 1..8 after the bottleneck blocks EXIT_BLOCKS (of 16): global average pooling and one linear layer to the
#     1,000 classes; exit 9 is the final fc of the network, so 9 exits as in the CIFAR networks
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from torchvision.models import ResNet50_Weights, resnet50

EXIT_BLOCKS = [2, 4, 6, 8, 10, 12, 14, 15]
N_CLASSES = 1000


class ResNet50Exits(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.blocks = nn.ModuleList(list(net.layer1) + list(net.layer2) + list(net.layer3) + list(net.layer4))
        self.fc = net.fc
        self.heads = nn.ModuleList([nn.Linear(self.blocks[b - 1].conv3.out_channels, N_CLASSES) for b in EXIT_BLOCKS])

    def features(self, x):
        # pooled features at the 8 exits, and at the end of the network (input of the fc)
        out = self.stem(x)
        feats = []
        for i, block in enumerate(self.blocks, 1):
            out = block(out)
            if i in EXIT_BLOCKS:
                feats.append(out.mean(dim=(2, 3)))
        return feats, out.mean(dim=(2, 3))

    def forward(self, x):
        # logits (B, 9, 1000)
        feats, final = self.features(x)
        return torch.stack([h(f) for h, f in zip(self.heads, feats)] + [self.fc(final)], dim=1)

    def plain(self, x):
        # the backbone alone (M0): no heads
        out = self.stem(x)
        for block in self.blocks:
            out = block(out)
        return self.fc(out.mean(dim=(2, 3)))

    def early_exit(self, x, stop):
        # early exit inference stopped at exit `stop` (0-based, 8 = the final fc): every head on the way is evaluated
        # and its max softmax compared with the threshold on the host
        out = self.stem(x)
        k = 0
        for i, block in enumerate(self.blocks, 1):
            out = block(out)
            if i in EXIT_BLOCKS:
                score = F.softmax(self.heads[k](out.mean(dim=(2, 3))), dim=1).max().item()
                if k == stop or score > 2.0:  # the comparison costs the same; `stop` forces the exit
                    return k
                k += 1
        return self.fc(out.mean(dim=(2, 3))).argmax(1).item()


def mac_fractions(model, size=224):
    # cumulative FLOPs to reach each exit (blocks and heads on the way) over the FLOPs of the backbone alone
    x = torch.zeros(1, 3, size, size)
    model.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            model.plain(x)
        total = fc.get_total_flops()
        fractions = []
        for stop in range(9):
            with FlopCounterMode(display=False) as fc:
                model.early_exit(x, stop)
            fractions.append(fc.get_total_flops() / total)
    return fractions, total
