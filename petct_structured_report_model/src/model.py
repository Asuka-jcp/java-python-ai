from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out


class ResNet3DEncoder(nn.Module):
    def __init__(self, in_channels: int = 2, channels=(32, 64, 128, 256)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        blocks = []
        in_ch = channels[0]
        for idx, out_ch in enumerate(channels):
            stride = 1 if idx == 0 else 2
            blocks.append(ResidualBlock3D(in_ch, out_ch, stride=stride))
            blocks.append(ResidualBlock3D(out_ch, out_ch, stride=1))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return x


class MultiRegionReportModel(nn.Module):
    def __init__(
        self,
        num_regions: int,
        num_pet_classes: int,
        num_ct_classes: int,
        encoder_channels=(32, 64, 128, 256),
        region_embed_dim: int = 128,
        hidden_dim: int = 256,
        use_suv_head: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ResNet3DEncoder(in_channels=2, channels=encoder_channels)
        self.region_embedding = nn.Embedding(num_regions, region_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.encoder.out_dim + region_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.pet_head = nn.Linear(hidden_dim, num_pet_classes)
        self.ct_head = nn.Linear(hidden_dim, num_ct_classes)
        self.use_suv_head = use_suv_head
        self.suv_head = nn.Linear(hidden_dim, 1) if use_suv_head else None

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz = image.size(0)
        feat = self.encoder(image)
        region_ids = torch.arange(self.region_embedding.num_embeddings, device=image.device)
        region_embed = self.region_embedding(region_ids)
        feat_expand = feat.unsqueeze(1).expand(-1, region_embed.size(0), -1)
        region_expand = region_embed.unsqueeze(0).expand(bsz, -1, -1)
        fused = self.fusion(torch.cat([feat_expand, region_expand], dim=-1))

        outputs = {
            "pet_logits": self.pet_head(fused),
            "ct_logits": self.ct_head(fused),
        }
        if self.use_suv_head and self.suv_head is not None:
            outputs["suv_pred"] = self.suv_head(fused).squeeze(-1)
        return outputs
