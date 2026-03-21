# =========================
# PHASE 3: CBAM ATTENTION
# =========================
import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1)
        )

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        x = x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

        # Spatial Attention
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sigmoid(self.spatial(torch.cat([avg, max_], dim=1)))

        return x