import math

import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from geoseg.models.components.transformer2d_parts import *

# 验证阶段不用随机crop
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
            nn.Conv2d(out_channels, out_channels,
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

        output = self.bottleneck(pyramids[4])
        return output

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def relative_pos_dis(height=512, weight=512, sita=0.9):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2
    #dis = torch.exp(-dis*(1/(2*sita**2)))
    return  dis


class CNNAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_patches=512):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_patches = num_patches

        #self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, padding=0, bias=False)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=3, padding=1, bias=False)
        self.dis = relative_pos_dis(math.sqrt(num_patches), math.sqrt(num_patches), sita=0.9).cuda()
        self.headsita = nn.Parameter(torch.randn(heads), requires_grad=True)
        self.sig = nn.Sigmoid()

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim), # inner_dim
            nn.ReLU(inplace=True),
        ) if project_out else nn.Identity()

    def forward(self, x, mode="train", smooth=1e-4):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (g d) h w -> b g (h w) d', g=self.heads), qkv)
        attn = torch.matmul(q, k.transpose(-1, -2)) # b g n n
        qk_norm = torch.sqrt(torch.sum(q ** 2, dim=-1)+smooth)[:, :, :, None] * torch.sqrt(torch.sum(k ** 2, dim=-1)+smooth)[:, :, None, :] + smooth
        attn = attn/qk_norm
        #attentionheatmap_visual2(attn, self.sig(self.headsita), out_dir='./Visualization/ACDC/SETR_plane2', value=1)
        #factor = 1/(2*(self.sig(self.headsita)+0.01)**2) # h
        factor = 1/(2*(self.sig(self.headsita)*(0.4-0.003)+0.003)**2) # af3 + limited setting this, or using the above line code
        dis = factor[:, None, None]*self.dis[None, :, :] # g n n
        dis = torch.exp(-dis)
        dis = dis/torch.sum(dis, dim=-1)[:, :, None]
        #attentionheatmap_visual2(dis[None, :, :, :], self.sig(self.headsita), out_dir='./Visualization/ACDC/dis', value=0.003)
        attn = attn * dis[None, :, :, :]
        #attentionheatmap_visual2(attn, self.sig(self.headsita), out_dir='./Visualization/ACDC/after', value=0.003)
        #attentionheatmap_visual(attn, out_dir='./Visualization/attention_af3/')
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b g (h w) d -> b (g d) h w', h=x.shape[2])
        if mode=="train":
            return self.to_out(out)
        else:
            return self.to_out(out), attn

class CNNFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class CNNTransformer_record(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=1024):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CNNAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches),
                CNNFeedForward(dim, mlp_dim, dropout=dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    def infere(self, x):
        ftokens, attmaps = [], []
        for attn, ff in self.layers:
            ax, amap = attn(x, mode="record")
            x = ax + x
            x = ff(x) + x
            ftokens.append(rearrange(x, 'b c h w -> b (h w) c'))
            attmaps.append(amap)
        return x, ftokens, attmaps

class Setr_ConvFormer(nn.Module):
    def __init__(self, n_channels, n_classes, imgsize, patch_num=32, dim=512, depth=12, heads=8, mlp_dim=512*4, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.image_height, self.image_width = pair(imgsize)
        self.patch_height, self.patch_width = pair(imgsize//patch_num)
        self.dmodel = dim

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = patch_num * patch_num

        self.cnn_encoder = CNNEncoder2(n_channels, dim, self.patch_height, self.patch_width) # the original is CNNs

        self.transformer = CNNTransformer_record(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dmodel, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel//4, self.dmodel // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dmodel // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dmodel // 4, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, img):
        x = self.cnn_encoder(img)
        # encoder
        x = self.transformer(x)  # b c h w -> b c h w
        x = self.decoder(x)
        return x
    def infere(self, img):
        x0 = self.cnn_encoder(img)
        # encoder
        x, ftokens, attmaps = self.transformer.infere(x0)
        ftokens.insert(0, rearrange(x0, 'b c h w -> b (h w) c'))
        # decoder
        x = self.decoder(x)
        return x, ftokens, attmaps


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
        self.attention1 = nn.Sequential(PAM_Module(512), nn.Conv2d(m_out_sz, 64, kernel_size=1))
        self.attention2 = nn.Sequential(PAM_Module(512), nn.Conv2d(m_out_sz, 128, kernel_size=1))
        self.attention3 = nn.Sequential(PAM_Module(512), nn.Conv2d(m_out_sz, 256, kernel_size=1))
        self.attention4 = nn.Sequential(PAM_Module(512), nn.Conv2d(m_out_sz, 512, kernel_size=1))

        self.transformer = CNNTransformer_record(dim=512, depth=12, heads=8, dim_head=64, dropout=0.1, num_patches=256)

        self.ta = TripletAttention()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, bias=False)
        self.up1 = Up(512, 256, bilinear)
        self.uni1 = nn.Conv2d(256, n_classes, kernel_size=1, bias=False)
        self.up2 = Up(256, 128, bilinear)
        self.uni2 = nn.Conv2d(128, n_classes, kernel_size=1, bias=False)
        self.up3 = Up(128, 64, bilinear)
        self.uni3 = nn.Conv2d(64, n_classes, kernel_size=1, bias=False)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.avg = nn.AvgPool2d(2)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        imsize = x.size()[2:]
        # if x.size()[3]==1024:
        #     x = self.conv(x)
        if self.training == False:
            x = self.avg(x)
        c1, c2, c3, c4 = self.backbone(x)
        csize = [c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:]]

        # if (csize[3][0] == 32):
        #     c4 = self.conv(c4)

        fp = self.transformer(c4)
        # if fp.size()[3] == 16:
        #     fp = Upsample(fp,32)
        fa1 = self.attention1(fp)
        fa2 = self.attention2(fp)
        fa3 = self.attention3(fp)
        fa4 = self.attention4(fp)

        x = self.up1(torch.add(c4, fa4), c3)
        f1 = x
        p1 = torch.add(x, Upsample(fa3, csize[2]))
        x = self.up2(p1, c2)
        f2 = x
        p2 = torch.add(x, Upsample(fa2, csize[1]))
        x = self.up3(p2, c1)
        f3 = x
        p3 = torch.add(x, Upsample(fa1, csize[0]))
        x = self.up4(p3)
        logits = self.outc(x)
        x = Upsample(logits, imsize)
        if self.training:
            feat_out1 = Upsample(self.uni1(f1), imsize)
            feat_out2 = Upsample(self.uni2(f2), imsize)
            feat_out3 = Upsample(self.uni3(f3), imsize)
            return x, feat_out1, feat_out2, feat_out3

        return x