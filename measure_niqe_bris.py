# ===== No-Reference Quality Metrics (NIQE & BRISQUE) =====
# Measures image quality without ground truth reference
# Used for unpaired datasets: DICM, LIME, MEF, NPE, VV
# Usage: python measure_niqe_bris.py --DICM

import glob
from tqdm import tqdm
from PIL import Image
from brisque import BRISQUE  # Blind/Referenceless Image Spatial Quality Evaluator
from loss.niqe_utils import *  # Natural Image Quality Evaluator
import argparse

# ===== Argument Parser =====
eval_parser = argparse.ArgumentParser(description="Eval")
eval_parser.add_argument(
    "--DICM", action="store_true", help="Measure DICM dataset results"
)
eval_parser.add_argument(
    "--LIME", action="store_true", help="Measure LIME dataset results"
)
eval_parser.add_argument(
    "--MEF", action="store_true", help="Measure MEF dataset results"
)
eval_parser.add_argument(
    "--NPE", action="store_true", help="Measure NPE dataset results"
)
eval_parser.add_argument("--VV", action="store_true", help="Measure VV dataset results")
ep = eval_parser.parse_args()

# Initialize BRISQUE model (url=False uses local model)
brisque_obj = BRISQUE(url=False)


# ===== Main Metrics Function =====
def metrics(im_dir):
    """Compute average NIQE and BRISQUE across all images in directory.

    Note: Both metrics are no-reference (don't need ground truth).
    Lower scores = better quality for both metrics.
    """
    avg_niqe = 0
    n = 0
    avg_brisque = 0

    for item in tqdm(sorted(glob.glob(im_dir))):
        n += 1

        pil_img = Image.open(item).convert("RGB")
        im1 = np.array(pil_img, dtype=np.float64)

        # Ensure image has 3 channels and is contiguous (required by NIQE)
        if len(im1.shape) == 2:
            im1 = np.stack([im1, im1, im1], axis=-1)  # Grayscale to RGB
        elif im1.shape[-1] != 3:
            im1 = np.stack([im1[:, :, 0]] * 3, axis=-1)

        im1 = np.ascontiguousarray(im1)

        # Compute no-reference quality scores
        score_brisque = brisque_obj.score(pil_img)  # BRISQUE expects PIL image
        score_niqe = calculate_niqe(im1)  # NIQE expects numpy array

        avg_brisque += score_brisque
        avg_niqe += score_niqe

        torch.cuda.empty_cache()

    # Compute averages
    avg_brisque = avg_brisque / n
    avg_niqe = avg_niqe / n
    return avg_niqe, avg_brisque


# ===== Main Entry Point =====
if __name__ == "__main__":
    # Set image directory based on dataset selection
    # Note: Different datasets use different image formats
    if ep.DICM:
        im_dir = "./output/DICM/*.jpg"
    elif ep.LIME:
        im_dir = "./output/LIME/*.bmp"
    elif ep.MEF:
        im_dir = "./output/MEF/*.png"
    elif ep.NPE:
        im_dir = "./output/NPE/*.jpg"
    elif ep.VV:
        im_dir = "./output/VV/*.jpg"

    # Compute and display metrics (lower is better for both)
    avg_niqe, avg_brisque = metrics(im_dir)
    print(f"Avg NIQE: {avg_niqe:.4f}")  # Lower is better
    print(f"Avg BRISQUE: {avg_brisque:.4f}")  # Lower is better
