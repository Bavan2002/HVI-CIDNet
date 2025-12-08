import cv2
import math
import numpy as np
from scipy.ndimage import convolve
from scipy.special import gamma
import torch

# ===== Image Resize Utilities (MATLAB-compatible bicubic) =====


def cubic(x):
    # Cubic interpolation kernel (Keys kernel)
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * (((absx > 1) * (absx <= 2)).type_as(absx))


def calculate_weights_indices(
    in_length, out_length, scale, kernel, kernel_width, antialiasing
):
    # Calculate interpolation weights and input pixel indices for resizing
    # Returns weights matrix and index mapping for each output pixel

    if (scale < 1) and antialiasing:
        # Larger kernel for anti-aliasing when downsampling
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Map output coords to input coords (0.5 -> 0.5, 0.5+scale -> 1.5)
    u = x / scale + 0.5 * (1 - 1 / scale)

    # Left-most contributing input pixel
    left = torch.floor(u - kernel_width / 2)

    # Max pixels involved in computing one output pixel
    p = math.ceil(kernel_width) + 2

    # Build index matrix: indices[i] = input pixels for output pixel i
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(
        0, p - 1, p
    ).view(1, p).expand(out_length, p)

    # Distance from each input pixel to output position
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

    # Apply cubic kernel to get weights
    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)

    # Normalize so each row sums to 1
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # Remove zero-weight columns at edges
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1  # symmetric padding needed at start
    sym_len_e = indices.max() - in_length  # symmetric padding needed at end
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # MATLAB-compatible bicubic image resize
    # Input: (C,H,W) tensor or (H,W,C) numpy, range [0,1]
    # Output: same format, resized
    squeeze_flag = False
    if type(img).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if img.ndim == 2:
            img = img[:, :, None]
            squeeze_flag = True
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if img.ndim == 2:
            img = img.unsqueeze(0)
            squeeze_flag = True

    in_c, in_h, in_w = img.size()
    out_h, out_w = math.ceil(in_h * scale), math.ceil(in_w * scale)
    kernel_width = 4
    kernel = "cubic"

    # Get interpolation weights for H and W dimensions
    weights_h, indices_h, sym_len_hs, sym_len_he = calculate_weights_indices(
        in_h, out_h, scale, kernel, kernel_width, antialiasing
    )
    weights_w, indices_w, sym_len_ws, sym_len_we = calculate_weights_indices(
        in_w, out_w, scale, kernel, kernel_width, antialiasing
    )
    # Process H dimension with symmetric boundary padding
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(img)

    # Mirror padding at boundaries
    sym_patch = img[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    # Apply H interpolation
    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = (
                img_aug[j, idx : idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])
            )

    # Process W dimension (same process)
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    # Apply W interpolation
    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx : idx + kernel_width].mv(weights_w[i])

    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2


# ===== Color Space Conversion Utilities =====


def _convert_input_type_range(img):
    # Convert input to float32 [0,1] range
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.0
    else:
        raise TypeError(
            f"The img type should be np.float32 or np.uint8, but got {img_type}"
        )
    return img


def _convert_output_type_range(img, dst_type):
    # Convert output to desired type/range (uint8 [0,255] or float32 [0,1])
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(
            f"The dst_type should be np.float32 or np.uint8, but got {dst_type}"
        )
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.0
    return img.astype(dst_type)


def rgb2ycbcr(img, y_only=False):
    # RGB to YCbCr (ITU-R BT.601, MATLAB-compatible)
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img,
            [
                [65.481, -37.797, 112.0],
                [128.553, -74.203, -93.786],
                [24.966, 112.0, -18.214],
            ],
        ) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img, y_only=False):
    # BGR to YCbCr (OpenCV format input)
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img,
            [
                [24.966, 112.0, -18.214],
                [128.553, -74.203, -93.786],
                [65.481, -37.797, 112.0],
            ],
        ) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2rgb(img):
    # YCbCr to RGB (MATLAB-compatible)
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(
        img,
        [
            [0.00456621, 0.00456621, 0.00456621],
            [0, -0.00153632, 0.00791071],
            [0.00625893, -0.00318811, 0],
        ],
    ) * 255.0 + [-222.921, 135.576, -276.836]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def to_y_channel(img):
    # Extract Y channel from BGR image (for PSNR/SSIM on luminance)
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.0


def reorder_image(img, input_order="HWC"):
    # Reorder image dimensions to HWC format
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(
            f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'"
        )
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == "CHW":
        img = img.transpose(1, 2, 0)
    return img


def rgb2ycbcr_pt(img, y_only=False):
    # PyTorch version of RGB to YCbCr (for batch processing)
    # Input: (N,3,H,W) tensor in [0,1]
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = (
            torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
        )
    else:
        weight = torch.tensor(
            [
                [65.481, -37.797, 112.0],
                [128.553, -74.203, -93.786],
                [24.966, 112.0, -18.214],
            ]
        ).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = (
            torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
        )

    out_img = out_img / 255.0
    return


def tensor2img(tensor):
    # Convert tensor to uint8 numpy image
    im = (255.0 * tensor).data.cpu().numpy()
    im[im > 255] = 255
    im[im < 0] = 0
    im = im.astype(np.uint8)
    return im


def img2tensor(img):
    # Convert numpy image to tensor (adds batch dim)
    img = (img / 255.0).astype("float32")
    if img.ndim == 2:
        img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))  # C, H, W
        img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    tensor = torch.from_numpy(img)
    return tensor


# ===== NIQE (Natural Image Quality Evaluator) =====


def estimate_aggd_param(block):
    # Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters
    # Used in NIQE to characterize natural image statistics
    # Returns: alpha (shape), beta_l (left scale), beta_r (right scale)
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (
        gamma(gam_reciprocal) * gamma(gam_reciprocal * 3)
    )

    # Left and right standard deviations (asymmetric)
    left_std = np.sqrt(np.mean(block[block < 0] ** 2))
    right_std = np.sqrt(np.mean(block[block > 0] ** 2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block))) ** 2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1) ** 2)
    array_position = np.argmin((r_gam - rhatnorm) ** 2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)


def compute_feature(block):
    # Compute 18-dim NIQE feature vector for an image block
    # Features capture deviations from natural image statistics
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    # Compute pairwise product features in 4 directions (H, V, diagonals)
    # Captures structural regularity disturbances
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(block, shifts[i], axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def niqe(
    img,
    mu_pris_param,
    cov_pris_param,
    gaussian_window,
    block_size_h=96,
    block_size_w=96,
):
    # Core NIQE computation
    # Compares distorted image statistics to pristine image statistics
    # Lower score = better quality
    assert img.ndim == 2, (
        "Input image must be a gray or Y (of YCbCr) image with shape (h, w)."
    )

    # Crop to integer number of blocks
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0 : num_block_h * block_size_h, 0 : num_block_w * block_size_w]

    distparam = []
    for scale in (1, 2):  # multi-scale analysis
        # Local mean and std via Gaussian filtering
        mu = convolve(img, gaussian_window, mode="nearest")
        sigma = np.sqrt(
            np.abs(
                convolve(np.square(img), gaussian_window, mode="nearest")
                - np.square(mu)
            )
        )
        # MSCN (Mean Subtracted Contrast Normalized) coefficients
        img_nomalized = (img - mu) / (sigma + 1)

        # Extract features from each block
        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                block = img_nomalized[
                    idx_h * block_size_h // scale : (idx_h + 1) * block_size_h // scale,
                    idx_w * block_size_w // scale : (idx_w + 1) * block_size_w // scale,
                ]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))

        if scale == 1:
            img = imresize(img / 255.0, scale=0.5, antialiasing=True)
            img = img * 255.0

    distparam = np.concatenate(distparam, axis=1)  # concatenate multi-scale features

    # Fit multivariate Gaussian to distorted image features
    mu_distparam = np.nanmean(distparam, axis=0)
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    # NIQE quality = Mahalanobis distance between pristine and distorted distributions
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param),
        np.transpose((mu_pris_param - mu_distparam)),
    )

    quality = np.sqrt(quality)
    quality = float(np.squeeze(quality))
    return quality


def calculate_niqe(img, crop_border=0, input_order="HWC", convert_to="y", **kwargs):
    # Main NIQE interface
    # Loads pristine model params and computes NIQE score
    # Lower = better quality
    niqe_pris_params = np.load("./loss/niqe_pris_params.npz")
    mu_pris_param = niqe_pris_params["mu_pris_param"]
    cov_pris_param = niqe_pris_params["cov_pris_param"]
    gaussian_window = niqe_pris_params["gaussian_window"]

    img = img.astype(np.float32)
    if input_order != "HW":
        img = reorder_image(img, input_order=input_order)
        if convert_to == "y":
            img = to_y_channel(img)  # compute on Y channel
        elif convert_to == "gray":
            img = cv2.cvtColor(img / 255.0, cv2.COLOR_BGR2GRAY) * 255.0
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    img = img.round()  # MATLAB compatibility

    niqe_result = niqe(img, mu_pris_param, cov_pris_param, gaussian_window)

    return niqe_result
