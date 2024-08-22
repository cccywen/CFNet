import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

# unet0加C4经FPN后进5个空间注意力模块
# 上采样用反卷积
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
        self.ta = TripletAttention()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.ta(x2)
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


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


class PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, nn.BatchNorm2d)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels,
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
        pyramids.extend([stage(features)for stage in self.stages])
        pyramids[1] = F.interpolate(pyramids[1], size=(2, 2), mode='bilinear', align_corners=True)
        pyramids[2] = F.interpolate(pyramids[2] + pyramids[1], size=(3, 3), mode='bilinear', align_corners=True)
        pyramids[3] = F.interpolate(pyramids[3] + pyramids[2], size=(6, 6), mode='bilinear', align_corners=True)
        pyramids[4] = F.interpolate(pyramids[4] + pyramids[3], size=(h, w), mode='bilinear', align_corners=True)

        # pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',align_corners=True) for stage in self.stages])

        a = torch.cat([pyramids[0], pyramids[4]], dim=1)

        output = self.bottleneck(torch.cat([pyramids[0], pyramids[4]], dim=1))
        return output


class PoolUNet(nn.Module):
    def __init__(self, n_classes, backbone='resnet18', bilinear=False):
        super(PoolUNet, self).__init__()

        self.backbone = timm.create_model(backbone, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)

        self.down = nn.Conv2d(64, 64, kernel_size=2, stride=2)

        self.n_classes = n_classes
        self.bilinear = bilinear

        m_out_sz = 512
        self.ppm = PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6])
        self.attention1 = nn.Sequential(PAM_Module(128), nn.Conv2d(m_out_sz // 4, 64, kernel_size=1))
        self.attention2 = nn.Sequential(PAM_Module(128), nn.Conv2d(m_out_sz // 4, 128, kernel_size=1))
        self.attention3 = nn.Sequential(PAM_Module(128), nn.Conv2d(m_out_sz // 4, 256, kernel_size=1))
        self.attention4 = nn.Sequential(PAM_Module(128), nn.Conv2d(m_out_sz // 4, 512, kernel_size=1))

        self.ta = TripletAttention()

        self.up1 = Up(512, 256, bilinear)
        self.uni1 = nn.Conv2d(262, 256, kernel_size=1, bias=False)
        self.up2 = Up(256, 128, bilinear)
        self.uni2 = nn.Conv2d(134, 128, kernel_size=1, bias=False)
        self.up3 = Up(128, 64, bilinear)
        self.uni3 = nn.Conv2d(70, 64, kernel_size=1, bias=False)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.backbone(x)
        csize = [c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:]]

        fp = self.ppm(c4)
        fa1 = self.attention1(fp)
        fa2 = self.attention2(fp)
        fa3 = self.attention3(fp)
        fa4 = self.attention4(fp)

        x = self.up1(torch.add(c4, fa4), c3)
        p1 = torch.add(x, Upsample(fa3, csize[2]))
        x = self.up2(p1, c2)
        p2 = torch.add(x, Upsample(fa2, csize[1]))
        x = self.up3(p2, c1)
        p3 = torch.add(x, Upsample(fa1, csize[0]))
        x = self.up4(p3)
        logits = self.outc(x)
        x = Upsample(logits, imsize)
        return x