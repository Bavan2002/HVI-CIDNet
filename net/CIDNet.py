import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin


# Main CIDNet: Dual-branch U-Net with cross-attention for low-light enhancement
# HV branch processes color (Hue-Value), I branch processes Intensity
# Branches communicate via Lightweight Cross Attention (LCA) at each scale
class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        channels=[36, 36, 72, 144],  # channels at each scale
        heads=[1, 2, 4, 8],  # attention heads at each scale
        norm=False,  # whether to use LayerNorm in up/downsample
    ):
        super(CIDNet, self).__init__()

        [ch1, ch2, ch3, ch4] = channels  # 36, 36, 72, 144
        [head1, head2, head3, head4] = heads  # 1, 2, 4, 8

        # ===== HV Branch (Color): processes 2-channel HV from HVI space =====
        # Encoder: progressively downsample and increase channels
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),  # pad to avoid border artifacts
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False),  # input is 3-ch HVI
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)  # scale 1: 1x -> 1/2x
        self.HVE_block2 = NormDownsample(
            ch2, ch3, use_norm=norm
        )  # scale 2: 1/2x -> 1/4x
        self.HVE_block3 = NormDownsample(
            ch3, ch4, use_norm=norm
        )  # scale 3: 1/4x -> 1/8x

        # Decoder: progressively upsample and decrease channels
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)  # 1/8x -> 1/4x
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)  # 1/4x -> 1/2x
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)  # 1/2x -> 1x
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(
                ch1, 2, 3, stride=1, padding=0, bias=False
            ),  # output 2-ch HV residual
        )

        # ===== I Branch (Intensity): processes 1-channel I =====
        # Encoder
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),  # input is 1-ch I
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        # Decoder
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(
                ch1, 1, 3, stride=1, padding=0, bias=False
            ),  # output 1-ch I residual
        )

        # ===== Cross-Attention Blocks =====
        # HV_LCA: HV branch queries I branch for intensity guidance
        # Encoder cross-attention (scales 2,3,4)
        self.HV_LCA1 = HV_LCA(ch2, head2)  # at 1/2x scale
        self.HV_LCA2 = HV_LCA(ch3, head3)  # at 1/4x scale
        self.HV_LCA3 = HV_LCA(ch4, head4)  # at 1/8x scale (bottleneck)
        # Decoder cross-attention (scales 4,3,2)
        self.HV_LCA4 = HV_LCA(ch4, head4)  # bottleneck
        self.HV_LCA5 = HV_LCA(ch3, head3)  # at 1/4x scale
        self.HV_LCA6 = HV_LCA(ch2, head2)  # at 1/2x scale

        # I_LCA: I branch queries HV branch for color guidance
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        self.trans = RGB_HVI()  # color space transform with learnable density_k

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)  # RGB -> HVI transform
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)  # extract I channel (index 2)

        # ===== Scale 0: Initial projection =====
        i_enc0 = self.IE_block0(i)  # I: (B,1,H,W) -> (B,ch1,H,W)
        i_enc1 = self.IE_block1(i_enc0)  # I: downsample to 1/2x
        hv_0 = self.HVE_block0(hvi)  # HV: (B,3,H,W) -> (B,ch1,H,W)
        hv_1 = self.HVE_block1(hv_0)  # HV: downsample to 1/2x
        i_jump0 = i_enc0  # skip connection for I decoder
        hv_jump0 = hv_0  # skip connection for HV decoder

        # ===== Scale 1: First cross-attention + downsample =====
        i_enc2 = self.I_LCA1(i_enc1, hv_1)  # I queries HV
        hv_2 = self.HV_LCA1(hv_1, i_enc1)  # HV queries I
        v_jump1 = i_enc2  # skip for decoder
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)  # downsample to 1/4x
        hv_2 = self.HVE_block2(hv_2)

        # ===== Scale 2: Second cross-attention + downsample =====
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)  # downsample to 1/8x (bottleneck)
        hv_3 = self.HVE_block3(hv_2)

        # ===== Scale 3: Bottleneck cross-attention =====
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)

        # ===== Decoder: Bottleneck =====
        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)

        # ===== Decoder Scale 2: Upsample + cross-attention =====
        hv_3 = self.HVD_block3(hv_4, hv_jump2)  # upsample with skip connection
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)

        # ===== Decoder Scale 1: Upsample + cross-attention =====
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)

        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)

        # ===== Decoder Scale 0: Final projection =====
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)  # -> (B,1,H,W) I residual
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)  # -> (B,2,H,W) HV residual

        # Combine residuals with input and convert back to RGB
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi  # residual learning
        output_rgb = self.trans.PHVIT(output_hvi)  # HVI -> RGB transform

        return output_rgb

    def HVIT(self, x):
        # Utility: expose RGB->HVI transform for external use
        hvi = self.trans.HVIT(x)
        return hvi
