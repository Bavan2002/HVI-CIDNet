# ===== HuggingFace Model Evaluation Script =====
# Load pretrained HVI-CIDNet from HuggingFace Hub and enhance a single image
# Usage: python eval_hf.py --path Fediory/HVI-CIDNet-LOLv1-wperc --input_img path/to/image.jpg

from net.CIDNet import CIDNet
import os
import json
import safetensors.torch as sf
from huggingface_hub import hf_hub_download
import argparse
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import platform
from PIL import Image

# ===== Argument Parser =====
eval_parser = argparse.ArgumentParser(description="EvalHF")
eval_parser.add_argument(
    "--path",
    type=str,
    default="Fediory/HVI-CIDNet-LOLv1-wperc",
    help="HuggingFace model path. See: https://huggingface.co/papers/2502.20272",
)
eval_parser.add_argument(
    "--input_img",
    type=str,
    default="../datasets/DICM/01.jpg",
    help="Path to input low-light image",
)
eval_parser.add_argument(
    "--alpha_s", type=float, default=1.0, help="Saturation scaling factor"
)
eval_parser.add_argument(
    "--alpha_i", type=float, default=1.0, help="Intensity scaling factor"
)
eval_parser.add_argument(
    "--gamma", type=float, default=1.0, help="Input gamma correction"
)
el = eval_parser.parse_args()


# ===== HuggingFace Model Loader =====
def from_pretrained(cls, pretrained_model_name_or_path: str):
    """Load pretrained weights from HuggingFace Hub.

    Args:
        cls: Model instance to load weights into
        pretrained_model_name_or_path: HuggingFace repo ID (e.g., "Fediory/HVI-CIDNet-LOLv1-wperc")

    Returns:
        Model with loaded weights
    """
    model_id = str(pretrained_model_name_or_path)

    # Download and load config (optional)
    config_file = hf_hub_download(
        repo_id=model_id, filename="config.json", repo_type="model"
    )
    config = None
    if config_file is not None:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Download and load safetensors weights
    model_file = hf_hub_download(
        repo_id=model_id, filename="model.safetensors", repo_type="model"
    )
    state_dict = sf.load_file(model_file)
    cls.load_state_dict(state_dict, strict=False)  # strict=False allows partial loading
    return cls


# ===== Initialize Model =====
model = CIDNet().cuda()
model = from_pretrained(cls=model, pretrained_model_name_or_path=el.path)
model.eval()

# ===== Load and Preprocess Input Image =====
pil2tensor = transforms.Compose([transforms.ToTensor()])
img = Image.open(el.input_img).convert("RGB")
input = pil2tensor(img)

# Pad to multiple of 8 (required by U-Net architecture)
factor = 8
h, w = input.shape[1], input.shape[2]
H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
padh = H - h if h % factor != 0 else 0
padw = W - w if w % factor != 0 else 0
input = F.pad(input.unsqueeze(0), (0, padw, 0, padh), "reflect")

# ===== Run Enhancement =====
with torch.no_grad():
    # Configure HVI transform parameters
    model.trans.alpha_s = el.alpha_s  # Saturation control
    model.trans.alpha = el.alpha_i  # Intensity control
    model.trans.gated = True  # Enable LOLv1-style gating
    model.trans.gated2 = True  # Enable alpha-based scaling
    output = model(input.cuda() ** el.gamma)  # Apply gamma before enhancement

# ===== Postprocess and Save =====
output = torch.clamp(output.cuda(), 0, 1).cuda()
output = output[:, :, :h, :w]  # Remove padding
enhanced_img = transforms.ToPILImage()(output.squeeze(0))

# Save to output directory
output_folder = "./output_hf"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
item = el.input_img
name = item.split("/")[-1]
enhanced_img.save(output_folder + "/" + name)
