# ===== Gradio Web Demo for HVI-CIDNet =====
# Interactive web interface for low-light image enhancement
# Usage: python app.py (GPU) or python app.py --cpu (CPU-only)

import numpy as np
import torch
import gradio as gr
from PIL import Image
from net.CIDNet import CIDNet
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import imquality.brisque as brisque  # Blind/Referenceless Image Spatial Quality Evaluator
from loss.niqe_utils import *  # Natural Image Quality Evaluator utilities
import platform
import argparse

# ===== CLI Arguments =====
opt_parser = argparse.ArgumentParser(description="App")
opt_parser.add_argument("--cpu", action="store_true", help="CPU-Only")
opt = opt_parser.parse_args()

# ===== Model Initialization =====
if opt.cpu:
    eval_net = CIDNet().cpu()
else:
    eval_net = CIDNet().cuda()

# Enable both gating modes for maximum flexibility
eval_net.trans.gated = True  # LOLv1-style gating
eval_net.trans.gated2 = True  # Alpha-based scaling


# ===== Main Image Processing Function =====
def process_image(input_img, score, model_path, gamma, alpha_s=1.0, alpha_i=1.0):
    """
    Enhance a low-light image using HVI-CIDNet.

    Args:
        input_img: PIL image to enhance
        score: 'Yes' to compute NIQE/BRISQUE quality metrics
        model_path: Path to model weights (relative to weights/)
        gamma: Input gamma correction (lower=lighter, range [0.5, 2.5])
        alpha_s: Saturation scaling factor (higher=more saturated)
        alpha_i: Intensity scaling factor (higher=lighter output)
    """
    torch.set_grad_enabled(False)  # Disable gradients for inference

    # Load selected model weights
    eval_net.load_state_dict(
        torch.load(
            os.path.join(directory, model_path),
            map_location=lambda storage, loc: storage,
        )
    )
    eval_net.eval()

    # Convert PIL to tensor
    pil2tensor = transforms.Compose([transforms.ToTensor()])
    input = pil2tensor(input_img)

    # Pad to multiple of 8 (required by U-Net architecture)
    factor = 8
    h, w = input.shape[1], input.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input = F.pad(input.unsqueeze(0), (0, padw, 0, padh), "reflect")  # Reflect padding

    with torch.no_grad():
        # Set transform parameters
        eval_net.trans.alpha_s = alpha_s  # Saturation control
        eval_net.trans.alpha = alpha_i  # Intensity control

        if opt.cpu:
            output = eval_net(input**gamma)  # Apply gamma before enhancement
        else:
            output = eval_net(input.cuda() ** gamma)

    # Clamp output to valid range
    if opt.cpu:
        output = torch.clamp(output, 0, 1)
    else:
        output = torch.clamp(output.cuda(), 0, 1).cuda()

    output = output[:, :, :h, :w]  # Remove padding
    enhanced_img = transforms.ToPILImage()(output.squeeze(0))

    # Optionally compute no-reference quality metrics
    if score == "Yes":
        im1 = enhanced_img.convert("RGB")
        score_brisque = brisque.score(im1)  # Lower is better
        im1 = np.array(im1)
        score_niqe = calculate_niqe(im1)  # Lower is better
        return enhanced_img, score_niqe, score_brisque
    else:
        return enhanced_img, 0, 0


# ===== Utility Functions for Weight Discovery =====
def find_pth_files(directory):
    """Recursively find all .pth weight files, excluding train/ subdirs."""
    pth_files = []
    for root, dirs, files in os.walk(directory):
        if "train" in root.split(os.sep):
            continue  # Skip training checkpoints
        for file in files:
            if file.endswith(".pth"):
                pth_files.append(os.path.join(root, file))
    return pth_files


def remove_weights_prefix(paths):
    """Remove 'weights/' prefix from paths for cleaner display."""
    os_name = platform.system()
    if os_name.lower() == "windows":
        cleaned_paths = [path.replace("weights\\", "") for path in paths]
    elif os_name.lower() == "linux":
        cleaned_paths = [path.replace("weights/", "") for path in paths]
    return cleaned_paths


# ===== Discover Available Model Weights =====
directory = "weights"
pth_files = find_pth_files(directory)
pth_files2 = remove_weights_prefix(pth_files)

# ===== Gradio Interface Definition =====
interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(label="Low-light Image", type="pil"),
        gr.Radio(
            choices=["Yes", "No"],
            label="Image Score",
            info='Calculate NIQE and BRISQUE, default is "No".',
        ),
        gr.Radio(
            choices=pth_files2,
            label="Model Weights",
            info='Choose your model. The best models are "SICE.pth" and "generalization.pth".',
        ),
        gr.Slider(
            0.1,
            5,
            label="gamma curve",
            step=0.01,
            value=1.0,
            info="Lower is lighter, and best range is [0.5,2.5].",
        ),
        gr.Slider(
            0,
            2,
            label="Alpha-s",
            step=0.01,
            value=1.0,
            info="Higher is more saturated.",
        ),
        gr.Slider(
            0.1, 2, label="Alpha-i", step=0.01, value=1.0, info="Higher is lighter."
        ),
    ],
    outputs=[
        gr.Image(label="Result", type="pil"),
        gr.Textbox(label="NIQE", info="Lower is better."),
        gr.Textbox(label="BRISQUE", info="Lower is better."),
    ],
    title="HVI-CIDNet (Low-Light Image Enhancement)",
    allow_flagging="never",
)

# Launch on port 7862
interface.launch(server_port=7862)
