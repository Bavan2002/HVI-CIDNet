import torch
import torch.nn.functional as F
import functools
from math import exp
from torch.autograd import Variable


# Applies reduction mode to loss tensor (none/mean/sum)
def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


# Apply element-wise weight to loss and reduce
def weight_reduce_loss(loss, weight=None, reduction="mean"):
    # If weight provided, multiply element-wise
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # Apply reduction
    if weight is None or reduction == "sum":
        loss = reduce_loss(loss, reduction)
    elif reduction == "mean":
        # Weighted mean: sum(loss) / sum(weight)
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


# Decorator: wraps element-wise loss function to support weight and reduction
# Usage: @weighted_loss on a function that returns element-wise loss
def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", **kwargs):
        loss = loss_func(pred, target, **kwargs)  # get element-wise loss
        loss = weight_reduce_loss(loss, weight, reduction)  # apply weight + reduce
        return loss

    return wrapper


# Weighted L1 loss (decorated)
@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction="none")


# Weighted MSE loss (decorated)
@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="none")


# ===== SSIM Helper Functions =====


# 1D Gaussian kernel
def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / torch.sum(gauss)  # normalize to sum=1


# Create 2D Gaussian window for SSIM computation
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # sigma=1.5
    _2D_window = (
        _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    )  # outer product
    # Expand to (channel, 1, window_size, window_size) for depthwise conv
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


# Compute SSIM map between two images
# SSIM = (2*mu1*mu2 + C1)(2*sigma12 + C2) / (mu1^2 + mu2^2 + C1)(sigma1^2 + sigma2^2 + C2)
def map_ssim(img1, img2, window, window_size, channel, size_average=True):
    # Local means via Gaussian-weighted average
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance: E[X^2] - E[X]^2
    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    # Stability constants (avoid division by zero)
    C1 = 0.01**2  # for luminance
    C2 = 0.03**2  # for contrast

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()  # scalar
    else:
        return ssim_map.mean(1).mean(1).mean(1)  # per-batch
