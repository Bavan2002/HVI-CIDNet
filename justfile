# HVI-CIDNet Justfile

# Default- shows all available commands
default:
    @just --list

# =============================================================================
# GRADIO DEMO
# =============================================================================

# Run Gradio demo with GPU (opens browser at http://127.0.0.1:7862)
demo:
    python3 app.py

# Run Gradio demo with CPU only
demo-cpu:
    python3 app.py --cpu

# =============================================================================
# HUGGING FACE INFERENCE
# =============================================================================

# Run inference using Hugging Face model (default: LOLv1-wperc)
# Usage: just hf-infer <input_image> [alpha_s] [alpha_i] [gamma]
hf-infer input_img alpha_s="1.0" alpha_i="1.0" gamma="1.0":
    python3 eval_hf.py --path fediory/HVI-CIDNet-LOLv1-wperc --input_img {{input_img}} --alpha_s {{alpha_s}} --alpha_i {{alpha_i}} --gamma {{gamma}}

# Run inference with LOLv2-syn generalization model (best for diverse images)
hf-infer-general input_img alpha_s="1.0" alpha_i="1.0" gamma="1.0":
    python3 eval_hf.py --path fediory/HVI-CIDNet-LOLv2-syn-generalization --input_img {{input_img}} --alpha_s {{alpha_s}} --alpha_i {{alpha_i}} --gamma {{gamma}}

# =============================================================================
# CUSTOM IMAGE INFERENCE (Using local weights)
# =============================================================================

# Infer single/multiple custom images with default generalization weights
# Usage: just infer-custom <image_path_or_folder> [alpha] [gamma]
infer-custom path alpha="1.0" gamma="1.0":
    python3 eval.py --unpaired --custome --custome_path {{path}} --unpaired_weights ./weights/LOLv2_syn/generalization.pth --alpha {{alpha}} --gamma {{gamma}}

# Infer with LOLv1 weights (with perceptual loss)
infer-custom-lolv1 path alpha="1.0" gamma="1.0":
    python3 eval.py --unpaired --custome --custome_path {{path}} --unpaired_weights ./weights/LOLv1/w_perc.pth --alpha {{alpha}} --gamma {{gamma}}

# Infer with LOLv2-syn weights (with perceptual loss)
infer-custom-lolv2syn path alpha="1.0" gamma="1.0":
    python3 eval.py --unpaired --custome --custome_path {{path}} --unpaired_weights ./weights/LOLv2_syn/w_perc.pth --alpha {{alpha}} --gamma {{gamma}}

# Infer with LOLv2-real weights (best PSNR)
infer-custom-lolv2real path alpha="1.0" gamma="1.0":
    python3 eval.py --unpaired --custome --custome_path {{path}} --unpaired_weights ./weights/LOLv2_real/best_PSNR.pth --alpha {{alpha}} --gamma {{gamma}}

# =============================================================================
# PAIRED DATASETS EVALUATION
# =============================================================================

# Evaluate on LOLv1 dataset with perceptual loss weights
eval-lolv1-perc:
    python3 eval.py --lol --perc

# Evaluate on LOLv1 dataset without perceptual loss weights
eval-lolv1:
    python3 eval.py --lol

# Evaluate on LOLv2-real dataset with best GT mean weights
eval-lolv2real-gtmean:
    python3 eval.py --lol_v2_real --best_GT_mean

# Evaluate on LOLv2-real dataset with best PSNR weights
eval-lolv2real-psnr:
    python3 eval.py --lol_v2_real --best_PSNR

# Evaluate on LOLv2-real dataset with best SSIM weights
eval-lolv2real-ssim:
    python3 eval.py --lol_v2_real --best_SSIM

# Evaluate on LOLv2-synthetic dataset with perceptual loss weights
eval-lolv2syn-perc:
    python3 eval.py --lol_v2_syn --perc

# Evaluate on LOLv2-synthetic dataset without perceptual loss weights
eval-lolv2syn:
    python3 eval.py --lol_v2_syn

# Evaluate on SICE-Grad dataset
eval-sice-grad:
    python3 eval.py --SICE_grad

# Evaluate on SICE-Mix dataset
eval-sice-mix:
    python3 eval.py --SICE_mix

# Evaluate on FiveK dataset
eval-fivek:
    python3 eval.py --fivek

# Evaluate on Sony-Total-Dark (SID) dataset
eval-sid:
    python3 eval_SID_blur.py --SID

# Evaluate on LOL-Blur dataset
eval-blur:
    python3 eval_SID_blur.py --Blur

# =============================================================================
# UNPAIRED DATASETS EVALUATION
# =============================================================================

# Evaluate on DICM dataset
# Usage: just eval-dicm [alpha] [gamma]
eval-dicm alpha="1.0" gamma="1.0":
    python3 eval.py --unpaired --DICM --unpaired_weights ./weights/LOLv2_syn/generalization.pth --alpha {{alpha}} --gamma {{gamma}}

# Evaluate on LIME dataset
eval-lime alpha="1.0" gamma="1.0":
    python3 eval.py --unpaired --LIME --unpaired_weights ./weights/LOLv2_syn/generalization.pth --alpha {{alpha}} --gamma {{gamma}}

# Evaluate on MEF dataset
eval-mef alpha="1.0" gamma="1.0":
    python3 eval.py --unpaired --MEF --unpaired_weights ./weights/LOLv2_syn/generalization.pth --alpha {{alpha}} --gamma {{gamma}}

# Evaluate on NPE dataset
eval-npe alpha="1.0" gamma="1.0":
    python3 eval.py --unpaired --NPE --unpaired_weights ./weights/LOLv2_syn/generalization.pth --alpha {{alpha}} --gamma {{gamma}}

# Evaluate on VV dataset
eval-vv alpha="1.0" gamma="1.0":
    python3 eval.py --unpaired --VV --unpaired_weights ./weights/LOLv2_syn/generalization.pth --alpha {{alpha}} --gamma {{gamma}}

# =============================================================================
# METRICS MEASUREMENT
# =============================================================================

# Measure metrics on LOLv1 dataset
measure-lolv1:
    python3 measure.py --lol

# Measure metrics on LOLv1 dataset with GT mean adjustment
measure-lolv1-gtmean:
    python3 measure.py --lol --use_GT_mean

# Measure metrics on LOLv2-real dataset
measure-lolv2real:
    python3 measure.py --lol_v2_real

# Measure metrics on LOLv2-real dataset with GT mean adjustment
measure-lolv2real-gtmean:
    python3 measure.py --lol_v2_real --use_GT_mean

# Measure metrics on LOLv2-synthetic dataset
measure-lolv2syn:
    python3 measure.py --lol_v2_syn

# Measure metrics on LOLv2-synthetic dataset with GT mean adjustment
measure-lolv2syn-gtmean:
    python3 measure.py --lol_v2_syn --use_GT_mean

# Measure metrics on Sony-Total-Dark (SID) dataset
measure-sid:
    python3 measure_SID_blur.py --SID

# Measure metrics on LOL-Blur dataset
measure-blur:
    python3 measure_SID_blur.py --Blur

# Measure metrics on SICE-Grad dataset
measure-sice-grad:
    python3 measure.py --SICE_grad

# Measure metrics on SICE-Grad dataset with GT mean adjustment
measure-sice-grad-gtmean:
    python3 measure.py --SICE_grad --use_GT_mean

# Measure metrics on SICE-Mix dataset
measure-sice-mix:
    python3 measure.py --SICE_mix

# Measure metrics on SICE-Mix dataset with GT mean adjustment
measure-sice-mix-gtmean:
    python3 measure.py --SICE_mix --use_GT_mean

# Measure metrics on FiveK dataset
measure-fivek:
    python3 measure.py --fivek

# Measure NIQE and BRISQUE on DICM dataset
measure-dicm:
    python3 measure_niqe_bris.py --DICM

# Measure NIQE and BRISQUE on LIME dataset
measure-lime:
    python3 measure_niqe_bris.py --LIME

# Measure NIQE and BRISQUE on MEF dataset
measure-mef:
    python3 measure_niqe_bris.py --MEF

# Measure NIQE and BRISQUE on NPE dataset
measure-npe:
    python3 measure_niqe_bris.py --NPE

# Measure NIQE and BRISQUE on VV dataset
measure-vv:
    python3 measure_niqe_bris.py --VV

# =============================================================================
# TRAINING
# =============================================================================

# Start training (configure parameters in data/options.py)
train:
    python3 train.py

# =============================================================================
# NETWORK TESTING
# =============================================================================

# Test network parameters, FLOPs, and running time
net-test:
    python3 net_test.py

# =============================================================================
# BATCH OPERATIONS
# =============================================================================

# Evaluate all paired datasets
eval-all-paired:
    @echo "Evaluating LOLv1..."
    @just eval-lolv1-perc
    @echo "Evaluating LOLv2-real..."
    @just eval-lolv2real-gtmean
    @echo "Evaluating LOLv2-syn..."
    @just eval-lolv2syn-perc
    @echo "Evaluating SICE-Grad..."
    @just eval-sice-grad
    @echo "Evaluating SICE-Mix..."
    @just eval-sice-mix
    @echo "Evaluating FiveK..."
    @just eval-fivek
    @echo "Evaluating SID..."
    @just eval-sid
    @echo "Evaluating LOL-Blur..."
    @just eval-blur

# Evaluate all unpaired datasets
eval-all-unpaired alpha="1.0" gamma="1.0":
    @echo "Evaluating DICM..."
    @just eval-dicm {{alpha}} {{gamma}}
    @echo "Evaluating LIME..."
    @just eval-lime {{alpha}} {{gamma}}
    @echo "Evaluating MEF..."
    @just eval-mef {{alpha}} {{gamma}}
    @echo "Evaluating NPE..."
    @just eval-npe {{alpha}} {{gamma}}
    @echo "Evaluating VV..."
    @just eval-vv {{alpha}} {{gamma}}

# Measure metrics on all paired datasets
measure-all-paired:
    @echo "Measuring LOLv1..."
    @just measure-lolv1
    @echo "Measuring LOLv2-real..."
    @just measure-lolv2real
    @echo "Measuring LOLv2-syn..."
    @just measure-lolv2syn
    @echo "Measuring SICE-Grad..."
    @just measure-sice-grad
    @echo "Measuring SICE-Mix..."
    @just measure-sice-mix
    @echo "Measuring FiveK..."
    @just measure-fivek
    @echo "Measuring SID..."
    @just measure-sid
    @echo "Measuring LOL-Blur..."
    @just measure-blur

# Measure metrics on all unpaired datasets
measure-all-unpaired:
    @echo "Measuring DICM..."
    @just measure-dicm
    @echo "Measuring LIME..."
    @just measure-lime
    @echo "Measuring MEF..."
    @just measure-mef
    @echo "Measuring NPE..."
    @just measure-npe
    @echo "Measuring VV..."
    @just measure-vv

# =============================================================================
# QUICK START EXAMPLES
# =============================================================================

# Quick example: Enhance a single image with default settings
example-single:
    @echo "Example: Enhancing a single image..."
    @echo "Usage: just hf-infer ./path/to/your/image.jpg"
    @echo "Output will be in ./output_hf/"

# Quick example: Enhance images with custom brightness
example-brighter:
    @echo "Example: Make images brighter (alpha=1.2)"
    @echo "Usage: just infer-custom ./your/images/ 1.2 1.0"

# Quick example: Enhance images with more contrast
example-contrast:
    @echo "Example: Increase contrast (gamma=1.2)"
    @echo "Usage: just infer-custom ./your/images/ 1.0 1.2"

# =============================================================================
# UTILITIES
# =============================================================================

# Check if required dependencies are installed
check-deps:
    @echo "Checking python3..."
    @python3 --version
    @echo "Checking PyTorch..."
    @python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
    @echo "Checking CUDA availability..."
    @python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Show output directories
show-outputs:
    @echo "Output directories:"
    @echo "  - Gradio/eval.py outputs: ./output/"
    @echo "  - Hugging Face outputs: ./output_hf/"
    @echo "  - Training outputs: ./results/training/"
    @echo "  - Training weights: ./weights/train/"

# Clean output directories
clean-outputs:
    @echo "Cleaning output directories..."
    rm -rf ./output/*
    rm -rf ./output_hf/*
    @echo "Done!"

# Show help for parameters
help-params:
    @echo "Parameter Guide:"
    @echo ""
    @echo "alpha / alpha_s / alpha_i:"
    @echo "  - Controls illumination/brightness enhancement"
    @echo "  - Higher values (>1.0) = brighter output"
    @echo "  - Lower values (<1.0) = darker output"
    @echo "  - Recommended range: 0.8 - 1.2"
    @echo ""
    @echo "gamma:"
    @echo "  - Controls enhancement curve (contrast)"
    @echo "  - Higher values = more contrast"
    @echo "  - Lower values = less contrast"
    @echo "  - Recommended range: 0.8 - 1.2"
    @echo ""
    @echo "Recommended weights for custom images:"
    @echo "  - Best generalization: ./weights/LOLv2_syn/generalization.pth"
    @echo "  - Natural images: ./weights/LOLv2_syn/w_perc.pth or ./weights/LOLv1/w_perc.pth"
