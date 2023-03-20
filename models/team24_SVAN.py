import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from basicsr.archs.arch_util import default_init_weights

class VAN_Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)  # (dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

class VAB_Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn


class SLKABlock(nn.Module):
    def __init__(self, num_feat, d_atten):
        super(SLKABlock, self).__init__()

        self.proj_1 = nn.Conv2d(num_feat, d_atten, 1)
        self.activation = nn.GELU()

        self.spatial_gating_unit = VAN_Attention(d_atten)

        self.proj_2 = nn.Conv2d(d_atten, d_atten, 1)

        self.atten_branch = VAB_Attention(d_atten)

        self.proj_3 = nn.Conv2d(d_atten, num_feat, 1)

        self.pixel_norm = nn.LayerNorm(num_feat)
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x):

        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)

        x = self.proj_2(x)
        x = self.activation(x)
        x = self.atten_branch(x)

        x = self.proj_3(x)
        x = x + shorcut

        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.pixel_norm(x)
        out = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return out


def make_layer(block, n_layers, *kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(*kwargs))
    return nn.Sequential(*layers)


class SVAN(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=48, num_block=11, d_atten=64, conv_groups=2):
        super(SVAN, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(SLKABlock, num_block, num_feat, d_atten)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=conv_groups) #conv_groups=2 for VapSR-S

        self.upsampler = pixelshuffle_block(num_feat, num_out_ch, upscale_factor=scale)

    def forward(self, feat):
        feat = self.conv_first(feat)
        body_feat = self.body(feat)
        body_out = self.conv_body(body_feat)
        feat = feat + body_out
        out = self.upsampler(feat)
        return out


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'silu':
        layer = nn.SiLU(inplace)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle(in_channels, out_channels, upscale_factor=4):
    upconv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(2)
    upconv2 = nn.Conv2d(16, out_channels * 4, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, pixel_shuffle, lrelu, upconv2, pixel_shuffle])


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - stride) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

