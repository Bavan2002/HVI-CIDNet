import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg_arch import VGGFeatureExtractor, Registry
from loss.loss_utils import *


_reduction_modes = ["none", "mean", "sum"]


# Simple L1 (Mean Absolute Error) loss with configurable weight
class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(L1Loss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        # Uses weighted l1_loss from loss_utils (supports element-wise weighting)
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction
        )


# Edge Loss: penalizes differences in high-frequency (edge) content
# Uses Laplacian pyramid to extract edges, then MSE on edge maps
class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(EdgeLoss, self).__init__()
        # 5x5 Gaussian kernel for smoothing (outer product of 1D kernel)
        k = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1).cuda()

        self.weight = loss_weight

    def conv_gauss(self, img):
        # Apply Gaussian blur using depthwise convolution
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        # Compute Laplacian: original - upsampled(downsampled(gaussian(original)))
        # This extracts high-frequency details (edges)
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]  # downsample 2x
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample with scaling
        filtered = self.conv_gauss(new_filter)  # smooth upsampled
        diff = current - filtered  # edge map = original - low-freq
        return diff

    def forward(self, x, y):
        # MSE between edge maps of prediction and target
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss * self.weight


# Perceptual Loss: compares VGG feature representations
# Measures high-level semantic similarity, not just pixel-wise
class PerceptualLoss(nn.Module):
    def __init__(
        self,
        layer_weights,  # dict: layer_name -> weight (e.g. {'conv3_4': 1.0})
        vgg_type="vgg19",
        use_input_norm=True,  # normalize to ImageNet stats
        range_norm=True,  # convert [-1,1] to [0,1]
        perceptual_weight=1.0,
        style_weight=0.0,  # Gram matrix loss weight (0 = disabled)
        criterion="l1",
    ):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        # VGG feature extractor (frozen, pretrained on ImageNet)
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == "mse":
            self.criterion = torch.nn.MSELoss(reduction="mean")
        elif self.criterion_type == "fro":
            self.criterion = None  # Frobenius norm, computed inline
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, x, gt):
        # Extract multi-layer VGG features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())  # detach GT to avoid backprop through it

        # Perceptual loss: compare feature activations
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    percep_loss += (
                        torch.norm(x_features[k] - gt_features[k], p="fro")
                        * self.layer_weights[k]
                    )
                else:
                    percep_loss += (
                        self.criterion(x_features[k], gt_features[k])
                        * self.layer_weights[k]
                    )
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # Style loss: compare Gram matrices (texture/style similarity)
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    style_loss += (
                        torch.norm(
                            self._gram_mat(x_features[k])
                            - self._gram_mat(gt_features[k]),
                            p="fro",
                        )
                        * self.layer_weights[k]
                    )
                else:
                    style_loss += (
                        self.criterion(
                            self._gram_mat(x_features[k]),
                            self._gram_mat(gt_features[k]),
                        )
                        * self.layer_weights[k]
                    )
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss  # returns tuple (percep, style)


# SSIM Loss: Structural Similarity Index as a loss function
# Returns (1 - SSIM) so minimizing loss = maximizing SSIM
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, weight=1.0):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)  # Gaussian window
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        # Recreate window if channel count changed or device mismatch
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        # Loss = (1 - SSIM) * weight
        return (
            1.0
            - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)
        ) * self.weight
