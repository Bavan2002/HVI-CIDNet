import torch
import torch.nn as nn
from einops import rearrange
from net.transformer_utils import *


# Cross Attention Block - enables communication between HV and I branches
# Uses multi-head attention where Q comes from one branch and K,V from another
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1)
        )  # learnable attention scaling

        # Query projection: 1x1 conv + 3x3 depthwise conv for local context
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        # Key-Value projection: produces 2*dim channels (split later)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            dim * 2,
            dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x: query source (the branch being updated)
        # y: key/value source (the other branch providing context)
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))  # query from x
        kv = self.kv_dwconv(self.kv(y))  # key-value from y
        k, v = kv.chunk(2, dim=1)  # split into key and value

        # Reshape for multi-head attention: (B, heads, C/heads, H*W)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        # L2 normalize Q and K for stable attention (cosine similarity)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Compute attention: (C/heads, C/heads) per head - channel attention, not spatial
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = attn @ v  # apply attention to values

        # Reshape back to spatial format
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


# Intensity Enhancement Layer - gated feedforward network
# Uses dual-path with Tanh activation and multiplicative gating
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)  # expand channels

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias
        )  # 2x for dual path

        # Depthwise convs for spatial mixing
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.dwconv1 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )
        self.dwconv2 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()  # bounded activation [-1, 1]

    def forward(self, x):
        x = self.project_in(x)  # expand to 2*hidden
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # split into two paths
        # Each path: depthwise conv with Tanh + residual
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2  # multiplicative gating
        x = self.project_out(x)  # project back to original dim
        return x


# Lightweight Cross Attention for HV branch
# Receives context from I branch, updates HV features
class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim)  # CDL (Color Denoise Layer) - same structure as IEL
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)  # cross attention

    def forward(self, x, y):
        # x: HV features, y: I features
        x = x + self.ffn(self.norm(x), self.norm(y))  # cross-attn with residual
        x = self.gdfn(
            self.norm(x)
        )  # feedforward WITHOUT residual (different from I_LCA)
        return x


# Lightweight Cross Attention for I branch
# Receives context from HV branch, updates I features
class I_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)  # Intensity Enhancement Layer
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x, y):
        # x: I features, y: HV features
        x = x + self.ffn(self.norm(x), self.norm(y))  # cross-attn with residual
        x = x + self.gdfn(
            self.norm(x)
        )  # feedforward WITH residual (preserves intensity info)
        return x
