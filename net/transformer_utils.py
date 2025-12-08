"""
Transformer Utilities: LayerNorm, Downsampling, Upsampling blocks for encoder-decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_first (B,C,H,W) format for CNNs"""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(normalized_shape)
        )  # learnable scale (gamma)
        self.bias = nn.Parameter(
            torch.zeros(normalized_shape)
        )  # learnable shift (beta)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)  # mean across channels
            s = (x - u).pow(2).mean(1, keepdim=True)  # variance across channels
            x = (x - u) / torch.sqrt(s + self.eps)  # normalize
            x = (
                self.weight[:, None, None] * x + self.bias[:, None, None]
            )  # scale & shift
            return x


class NormDownsample(nn.Module):
    """Downsample 2x: Conv3x3 -> Bilinear(0.5x) -> PReLU -> [LayerNorm]
    Input: (B, in_ch, H, W) -> Output: (B, out_ch, H/2, W/2)
    """

    def __init__(self, in_ch, out_ch, scale=0.5, use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()  # allows negative values (important for HVI)
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale),
        )  # smooth downsample

    def forward(self, x):
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x


class NormUpsample(nn.Module):
    """Upsample 2x with skip connection: Conv3x3 -> Bilinear(2x) -> Concat(skip) -> Conv1x1 -> PReLU
    Input x: (B, in_ch, H, W), skip y: (B, out_ch, 2H, 2W) -> Output: (B, out_ch, 2H, 2W)
    """

    def __init__(self, in_ch, out_ch, scale=2, use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale),
        )  # smooth upsample
        self.up = nn.Conv2d(
            out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False
        )  # fuse skip

    def forward(self, x, y):
        x = self.up_scale(x)  # upsample to match skip size
        x = torch.cat([x, y], dim=1)  # concat with skip connection
        x = self.up(x)  # reduce channels
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x
