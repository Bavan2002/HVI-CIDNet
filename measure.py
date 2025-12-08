# ===== Image Quality Metrics Computation =====
# Measures PSNR, SSIM, and LPIPS between enhanced outputs and ground truth
# Usage: python measure.py --lol  (measure LOLv1 results)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
import torch
import glob
import cv2
import lpips  # Learned Perceptual Image Patch Similarity
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import platform

# ===== Argument Parser =====
mea_parser = argparse.ArgumentParser(description="Measure")
mea_parser.add_argument(
    "--use_GT_mean",
    action="store_true",
    help="Use the mean of GT to rectify the output of the model",
)
mea_parser.add_argument("--lol", action="store_true", help="measure lolv1 dataset")
mea_parser.add_argument(
    "--lol_v2_real", action="store_true", help="measure lol_v2_real dataset"
)
mea_parser.add_argument(
    "--lol_v2_syn", action="store_true", help="measure lol_v2_syn dataset"
)
mea_parser.add_argument(
    "--SICE_grad", action="store_true", help="measure SICE_grad dataset"
)
mea_parser.add_argument(
    "--SICE_mix", action="store_true", help="measure SICE_mix dataset"
)
mea_parser.add_argument("--fivek", action="store_true", help="measure fivek dataset")
mea = mea_parser.parse_args()


# ===== SSIM Computation (Single Channel) =====
def ssim(prediction, target):
    """Compute SSIM for single-channel images using 11x11 Gaussian window."""
    C1 = (0.01 * 255) ** 2  # Stability constant for luminance
    C2 = (0.03 * 255) ** 2  # Stability constant for contrast
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)

    # Create 11x11 Gaussian kernel with sigma=1.5
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    # Compute local means (crop 5 pixels from each edge)
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    # SSIM formula: (2*mu1*mu2 + C1)(2*sigma12 + C2) / ((mu1^2+mu2^2+C1)(sigma1^2+sigma2^2+C2))
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


# ===== SSIM Computation (Multi-Channel) =====
def calculate_ssim(target, ref):
    """Calculate SSIM - same output as MATLAB. Images should be [0, 255]."""
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)  # Grayscale
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            # RGB: average SSIM across channels
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


# ===== PSNR Computation =====
def calculate_psnr(target, ref):
    """Compute Peak Signal-to-Noise Ratio in dB."""
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    # PSNR = 10 * log10(MAX^2 / MSE), MAX=255 for 8-bit images
    psnr = 10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8))
    return psnr


# ===== Main Metrics Function =====
def metrics(im_dir, label_dir, use_GT_mean):
    """Compute average PSNR, SSIM, LPIPS across all images in directory."""
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    n = 0

    # Initialize LPIPS model (AlexNet backbone)
    loss_fn = lpips.LPIPS(net="alex")
    loss_fn.cuda()

    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1

        im1 = Image.open(item).convert("RGB")  # Enhanced image

        # Extract filename (handle both Windows and Linux paths)
        os_name = platform.system()
        if os_name.lower() == "windows":
            name = item.split("\\")[-1]
        elif os_name.lower() == "linux":
            name = item.split("/")[-1]
        else:
            name = item.split("/")[-1]

        im2 = Image.open(label_dir + name).convert("RGB")  # Ground truth
        (h, w) = im2.size
        im1 = im1.resize((h, w))  # Resize to match GT dimensions
        im1 = np.array(im1)
        im2 = np.array(im2)

        # Optional: Normalize brightness to match GT mean
        if use_GT_mean:
            mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
            mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
            im1 = np.clip(im1 * (mean_target / mean_restored), 0, 255)

        # Compute metrics
        score_psnr = calculate_psnr(im1, im2)
        score_ssim = calculate_ssim(im1, im2)

        # LPIPS requires tensor input in [-1, 1] range
        ex_p0 = lpips.im2tensor(im1).cuda()
        ex_ref = lpips.im2tensor(im2).cuda()
        score_lpips = loss_fn.forward(ex_ref, ex_p0)

        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips.item()
        torch.cuda.empty_cache()

    # Compute averages
    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_lpips = avg_lpips / n
    return avg_psnr, avg_ssim, avg_lpips


# ===== Main Entry Point =====
if __name__ == "__main__":
    # Dataset paths: output images vs ground truth
    if mea.lol:
        im_dir = "./output/LOLv1/*.png"
        label_dir = "./datasets/LOLdataset/eval15/high/"
    if mea.lol_v2_real:
        im_dir = "./output/LOLv2_real/*.png"
        label_dir = "./datasets/LOLv2/Real_captured/Test/Normal/"
    if mea.lol_v2_syn:
        im_dir = "./output/LOLv2_syn/*.png"
        label_dir = "./datasets/LOLv2/Synthetic/Test/Normal/"
    if mea.SICE_grad:
        im_dir = "./output/SICE_grad/*.png"
        label_dir = "./datasets/SICE/SICE_Reshape/"
    if mea.SICE_mix:
        im_dir = "./output/SICE_mix/*.png"
        label_dir = "./datasets/SICE/SICE_Reshape/"
    if mea.fivek:
        im_dir = "./output/fivek/*.jpg"
        label_dir = "./datasets/FiveK/test/target/"

    # Compute and display metrics
    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, mea.use_GT_mean)
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
    print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))  # Lower is better
