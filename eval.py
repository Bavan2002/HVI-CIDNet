# ===== Evaluation Script for HVI-CIDNet =====
# Runs trained model on test datasets and saves enhanced images
# Usage: python eval.py --lol --perc  (evaluate LOLv1 with perceptual loss weights)

import os
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.CIDNet import CIDNet

# ===== Argument Parser Setup =====
eval_parser = argparse.ArgumentParser(description="Eval")
eval_parser.add_argument(
    "--perc", action="store_true", help="trained with perceptual loss"
)
eval_parser.add_argument("--lol", action="store_true", help="output lolv1 dataset")
eval_parser.add_argument(
    "--lol_v2_real", action="store_true", help="output lol_v2_real dataset"
)
eval_parser.add_argument(
    "--lol_v2_syn", action="store_true", help="output lol_v2_syn dataset"
)
eval_parser.add_argument(
    "--SICE_grad", action="store_true", help="output SICE_grad dataset"
)
eval_parser.add_argument(
    "--SICE_mix", action="store_true", help="output SICE_mix dataset"
)
eval_parser.add_argument("--fivek", action="store_true", help="output FiveK dataset")

eval_parser.add_argument(
    "--best_GT_mean",
    action="store_true",
    help="output lol_v2_real dataset best_GT_mean",
)
eval_parser.add_argument(
    "--best_PSNR", action="store_true", help="output lol_v2_real dataset best_PSNR"
)
eval_parser.add_argument(
    "--best_SSIM", action="store_true", help="output lol_v2_real dataset best_SSIM"
)

eval_parser.add_argument(
    "--custome", action="store_true", help="output custome dataset"
)
eval_parser.add_argument("--custome_path", type=str, default="./YOLO")
eval_parser.add_argument(
    "--unpaired", action="store_true", help="output unpaired dataset"
)
eval_parser.add_argument("--DICM", action="store_true", help="output DICM dataset")
eval_parser.add_argument("--LIME", action="store_true", help="output LIME dataset")
eval_parser.add_argument("--MEF", action="store_true", help="output MEF dataset")
eval_parser.add_argument("--NPE", action="store_true", help="output NPE dataset")
eval_parser.add_argument("--VV", action="store_true", help="output VV dataset")
eval_parser.add_argument("--alpha", type=float, default=1.0)
eval_parser.add_argument("--gamma", type=float, default=1.0)
eval_parser.add_argument(
    "--unpaired_weights", type=str, default="./weights/LOLv2_syn/w_perc.pth"
)
eval_parser.add_argument("--cpu", action="store_true", help="run on CPU only")

ep = eval_parser.parse_args()  # Parse all CLI arguments

# ===== Device Configuration =====
if ep.cpu:
    device = torch.device("cpu")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Main Evaluation Function =====
def eval(
    model,
    testing_data_loader,
    model_path,
    output_folder,
    device,
    norm_size=True,  # If True, images already resized; if False, restore original size
    LOL=False,  # LOLv1 dataset flag - uses gated transform
    v2=False,  # LOLv2-real dataset flag - uses alpha scaling
    unpaired=False,  # Unpaired dataset flag (DICM, LIME, etc.)
    alpha=1.0,  # Intensity scaling factor for v2/unpaired
    gamma=1.0,  # Input gamma correction
):
    torch.set_grad_enabled(False)  # Disable gradients for inference

    # Load pretrained weights
    model.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage)
    )
    print("Pre-trained model is loaded.")
    model.eval()  # Set to evaluation mode
    print("Evaluation:")

    # Configure HVI transform based on dataset type
    if LOL:
        model.trans.gated = True  # Use learned gating for LOLv1
    elif v2:
        model.trans.gated2 = True  # Use alpha-based scaling for LOLv2
        model.trans.alpha = alpha
    elif unpaired:
        model.trans.gated2 = True  # Unpaired uses same config as v2
        model.trans.alpha = alpha

    # Process each image in the test set
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]  # Fixed-size images
            else:
                input, name, h, w = (
                    batch[0],
                    batch[1],
                    batch[2],
                    batch[3],
                )  # Variable size with original dims

            input = input.to(device)
            output = model(input**gamma)  # Apply gamma before enhancement

        # Create output directory if needed
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        output = torch.clamp(output, 0, 1)  # Clamp to valid range
        if not norm_size:
            output = output[:, :, :h, :w]  # Crop to original size

        # Save enhanced image
        output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
        output_img.save(output_folder + name[0])
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("===> End evaluation")

    # Reset transform flags
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)


if __name__ == "__main__":
    # Validate CUDA availability
    if not ep.cpu and not torch.cuda.is_available():
        raise Exception("No GPU found. Use --cpu flag to run on CPU")

    # Create output directory
    if not os.path.exists("./output"):
        os.mkdir("./output")

    norm_size = True  # Default: images are pre-resized
    num_workers = 1
    alpha = None  # Intensity scaling (only used for v2/unpaired)

    # ===== Dataset Configuration =====
    # Each dataset has specific: data path, output folder, weight path

    if ep.lol:
        # LOLv1 dataset - 15 test images
        eval_data = DataLoader(
            dataset=get_eval_set("./datasets/LOLdataset/eval15/low"),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = "./output/LOLv1/"
        if ep.perc:
            weight_path = "./weights/LOLv1/w_perc.pth"  # With perceptual loss
        else:
            weight_path = "./weights/LOLv1/wo_perc.pth"  # Without perceptual loss

    elif ep.lol_v2_real:
        # LOLv2-Real dataset - real captured low-light images
        eval_data = DataLoader(
            dataset=get_eval_set("./datasets/LOLv2/Real_captured/Test/Low"),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = "./output/LOLv2_real/"
        # Different alpha values optimized for different metrics
        if ep.best_GT_mean:
            weight_path = "./weights/LOLv2_real/w_perc.pth"
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = "./weights/LOLv2_real/best_PSNR.pth"
            alpha = 0.8
        elif ep.best_SSIM:
            weight_path = "./weights/LOLv2_real/best_SSIM.pth"
            alpha = 0.82

    elif ep.lol_v2_syn:
        # LOLv2-Synthetic dataset - synthetically darkened images
        eval_data = DataLoader(
            dataset=get_eval_set("./datasets/LOLv2/Synthetic/Test/Low"),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = "./output/LOLv2_syn/"
        if ep.perc:
            weight_path = "./weights/LOLv2_syn/w_perc.pth"
        else:
            weight_path = "./weights/LOLv2_syn/wo_perc.pth"

    elif ep.SICE_grad:
        # SICE Gradient subset - variable exposure images
        eval_data = DataLoader(
            dataset=get_SICE_eval_set("./datasets/SICE/SICE_Grad"),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = "./output/SICE_grad/"
        weight_path = "./weights/SICE.pth"
        norm_size = False  # Variable image sizes

    elif ep.SICE_mix:
        # SICE Mixed subset
        eval_data = DataLoader(
            dataset=get_SICE_eval_set("./datasets/SICE/SICE_Mix"),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = "./output/SICE_mix/"
        weight_path = "./weights/SICE.pth"
        norm_size = False

    elif ep.fivek:
        # MIT-Adobe FiveK dataset
        eval_data = DataLoader(
            dataset=get_SICE_eval_set("./datasets/FiveK/test/input"),
            num_workers=num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = "./output/fivek/"
        weight_path = "./weights/fivek.pth"
        norm_size = False

    elif ep.unpaired:
        # ===== Unpaired Datasets (no ground truth) =====
        # Used for qualitative evaluation only
        if ep.DICM:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("./datasets/DICM"),
                num_workers=num_workers,
                batch_size=1,
                shuffle=False,
            )
            output_folder = "./output/DICM/"
        elif ep.LIME:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("./datasets/LIME"),
                num_workers=num_workers,
                batch_size=1,
                shuffle=False,
            )
            output_folder = "./output/LIME/"
        elif ep.MEF:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("./datasets/MEF"),
                num_workers=num_workers,
                batch_size=1,
                shuffle=False,
            )
            output_folder = "./output/MEF/"
        elif ep.NPE:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("./datasets/NPE"),
                num_workers=num_workers,
                batch_size=1,
                shuffle=False,
            )
            output_folder = "./output/NPE/"
        elif ep.VV:
            eval_data = DataLoader(
                dataset=get_SICE_eval_set("./datasets/VV"),
                num_workers=num_workers,
                batch_size=1,
                shuffle=False,
            )
            output_folder = "./output/VV/"
        elif ep.custome:
            # Custom user-provided dataset
            eval_data = DataLoader(
                dataset=get_SICE_eval_set(ep.custome_path),
                num_workers=num_workers,
                batch_size=1,
                shuffle=False,
            )
            output_folder = "./output/custome/"
        alpha = ep.alpha
        norm_size = False
        weight_path = ep.unpaired_weights

    # ===== Run Evaluation =====
    eval_net = CIDNet().to(device)
    eval(
        eval_net,
        eval_data,
        weight_path,
        output_folder,
        device,
        norm_size=norm_size,
        LOL=ep.lol,
        v2=ep.lol_v2_real,
        unpaired=ep.unpaired,
        alpha=alpha,
        gamma=ep.gamma,
    )
