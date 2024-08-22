import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

# 下采 不加权上采 每次上采后加两次卷积 unet0+poolNet 第三次上采没加高级特征
# 每个stage都加注意力的效果不好 80epoch 86    new vai100epoch 81
# New 70epoch potsdam 864
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

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def Upsample(x, size):
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

class PAM_Module(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.softplus_feature = torch.nn.functional.softplus
        self.eps = eps

        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, height, width = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.softplus_feature(Q).permute(-3, -1, -2)
        K = self.softplus_feature(K)

        KV = torch.einsum("bmn, bcn->bmc", K, V)

        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)

        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, nn.BatchNorm2d)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PoolUNet(nn.Module):
    def __init__(self, n_classes, backbone='resnet18',bilinear=False):
        super(PoolUNet, self).__init__()

        self.backbone = timm.create_model(backbone, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)

        self.down = nn.Conv2d(64, 64, kernel_size=2, stride=2)

        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = PAM_Module(512)

        self.up1 = Up(512, 256, bilinear)
        self.uni1 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.up2 = Up(256, 128, bilinear)
        self.uni2 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.up3 = Up(128, 64, bilinear)
        self.uni3 = nn.Conv2d(70, 64, kernel_size=1, bias=False)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.backbone(x)
        csize = [c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:]]

        fa = self.attention(c4)
        x = self.up1(torch.add(fa, c4), c3)
        fa = self.uni1(Upsample(fa, csize[2]))
        x = self.up2(torch.add(x, fa), c2)

        fa = self.uni2(Upsample(fa, csize[1]))
        x = self.up3(torch.add(x, fa), c1)
        x = self.up4(x)
        logits = self.outc(x)
        x = Upsample(logits, imsize)
        return x