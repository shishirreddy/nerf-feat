"""ResNet-18 U-Net encoder used for dense feature and mask prediction."""


import torch
import torchvision
from torch import nn
from torchvision.models import ResNet18_Weights


def conv_relu(in_channels: int, out_channels: int, kernel_size: int, padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    """Compact ResNet-18 U-Net with one dense prediction head."""

    def __init__(self, num_classes: int, pre_head_channels: int = 64, num_decoders: int = 1):
        super().__init__()
        if num_decoders != 1:
            raise ValueError("ResNetUNet supports a single decoder.")

        backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        layers = list(backbone.children())
        self.layer0 = nn.Sequential(*layers[:3])
        self.layer1 = nn.Sequential(*layers[3:5])
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.layer0_projection = conv_relu(64, 64, 1, 0)
        self.layer1_projection = conv_relu(64, 64, 1, 0)
        self.layer2_projection = conv_relu(128, 128, 1, 0)
        self.layer3_projection = conv_relu(256, 256, 1, 0)
        self.layer4_projection = conv_relu(512, 512, 1, 0)
        self.up3 = conv_relu(256 + 512, 512, 3, 1)
        self.up2 = conv_relu(128 + 512, 256, 3, 1)
        self.up1 = conv_relu(64 + 256, 256, 3, 1)
        self.up0 = conv_relu(64 + 256, 128, 3, 1)
        self.pre_head = conv_relu(128, pre_head_channels, 3, 1)
        self.head = nn.Conv2d(pre_head_channels, num_classes, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.layer4_projection(layer4)
        x = self.upsample(x)
        x = self._fuse_skip(x, self.layer3_projection(layer3), self.up3)
        x = self._fuse_skip(x, self.layer2_projection(layer2), self.up2)
        x = self._fuse_skip(x, self.layer1_projection(layer1), self.up1)
        x = self._fuse_skip(x, self.layer0_projection(layer0), self.up0)
        return self.head(self.pre_head(x))

    def _fuse_skip(
        self, x: torch.Tensor, skip: torch.Tensor, projection: nn.Module
    ) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        x = projection(x)
        return self.upsample(x)
