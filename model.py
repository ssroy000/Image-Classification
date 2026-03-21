# =========================
# MODEL
# =========================
import torch.nn as nn

from phase2_efficient import MobileNetBackbone
from phase3_attention import CBAM

class LungDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = MobileNetBackbone()
        self.attention = CBAM(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)