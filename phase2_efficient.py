# =========================
# PHASE 2: BACKBONE
# =========================
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c)
        self.pointwise = nn.Conv2d(in_c, out_c, 1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class MobileNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(3, 32),
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 128)
        )

    def forward(self, x):
        return self.layers(x)
