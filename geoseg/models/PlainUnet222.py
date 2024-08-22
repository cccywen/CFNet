import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

# 纯backbone
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 看之后可不可以改成bes中的上采样
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def Upsample(x, size):
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)


class PlainUNet(nn.Module):
    def __init__(self, n_classes, backbone='resnet18', bilinear=False):
        super(PlainUNet, self).__init__()

        self.backbone = timm.create_model(backbone, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)

        self.down = nn.Conv2d(64, 64, kernel_size=2, stride=2)

        self.n_classes = n_classes
        self.bilinear = bilinear

        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.backbone(x)

        x = self.up1(c4, c3)
        x = self.up2(x, c2)
        x = self.up3(x, c1)
        x = self.up4(x)
        logits = self.outc(x)
        x = Upsample(logits, imsize)
        return x