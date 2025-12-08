"""
HVI Color Space Transformation: RGB <-> HVI (Hue-chroma-H, Hue-chroma-V, Intensity)
Key innovation: learnable density_k parameter weights color reliability by intensity level
"""

import torch
import torch.nn as nn

pi = 3.141592653589793


class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(
            torch.full([1], 0.2)
        )  # learnable color sensitivity param
        self.gated = False  # saturation scaling flag for inference
        self.gated2 = False  # RGB intensity scaling flag for inference
        self.alpha = 1.0  # RGB intensity multiplier
        self.alpha_s = 1.3  # saturation multiplier
        self.this_k = 0  # stores k value for inverse transform

    def HVIT(self, img):
        """RGB to HVI transform"""
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = (
            torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        )

        # Extract Value (brightness) and min for saturation calc
        value = img.max(1)[0].to(dtypes)  # V = max(R,G,B)
        img_min = img.min(1)[0].to(dtypes)  # for saturation calculation

        # Compute Hue based on which channel is max (standard HSV formula)
        hue[img[:, 2] == value] = (
            4.0
            + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        )  # Blue max
        hue[img[:, 1] == value] = (
            2.0
            + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        )  # Green max
        hue[img[:, 0] == value] = (
            0.0
            + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]
        ) % 6  # Red max
        hue[img.min(1)[0] == value] = 0.0  # achromatic (grayscale)
        hue = hue / 6.0  # normalize to [0,1]

        # Saturation = (V - min) / V
        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        # Color sensitivity: bell curve - LOW at dark/bright, HIGH at medium intensity
        k = self.density_k
        self.this_k = k.item()
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)

        # Convert polar hue to Cartesian (H,V) coordinates
        ch = (2.0 * pi * hue).cos()  # x-component
        cv = (2.0 * pi * hue).sin()  # y-component

        # Final HVI: color weighted by saturation and color_sensitivity
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        xyz = torch.cat([H, V, I], dim=1)
        return xyz

    def PHVIT(self, img):
        """HVI to RGB inverse transform"""
        eps = 1e-8
        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        # Clamp to valid ranges
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)

        # Reverse color sensitivity transform
        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)

        # Cartesian to polar: recover hue and saturation
        h = torch.atan2(V + eps, H + eps) / (2 * pi)
        h = h % 1  # ensure [0,1]
        s = torch.sqrt(H**2 + V**2 + eps)  # magnitude = saturation

        if self.gated:
            s = s * self.alpha_s  # optional saturation boost

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        # Standard HSV to RGB conversion (6 sectors)
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)  # sector index
        f = h * 6.0 - hi  # fractional part
        p = v * (1.0 - s)  # min RGB
        q = v * (1.0 - (f * s))  # descending
        t = v * (1.0 - ((1.0 - f) * s))  # ascending

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        # Sector 0-5 RGB assignments
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha  # optional intensity scaling
        return rgb
