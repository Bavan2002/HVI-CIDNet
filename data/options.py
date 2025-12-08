# ===== Training Configuration Options =====
# Central configuration file for all training hyperparameters
# Usage: Imported by train.py to configure training runs

import argparse


def option():
    """Define all training configuration options via argparse."""
    parser = argparse.ArgumentParser(description="CIDNet")

    # ===== Training Hyperparameters =====
    parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
    parser.add_argument(
        "--cropSize", type=int, default=400, help="image crop size (patch size)"
    )
    parser.add_argument(
        "--nEpochs", type=int, default=1500, help="number of epochs to train for end"
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="number of epochs to start, >0 is retrained a pre-trained pth",
    )
    parser.add_argument(
        "--snapshots", type=int, default=10, help="Snapshots for save checkpoints pth"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--gpu_mode", type=bool, default=True)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="number of threads for dataloader to use",
    )

    # ===== Learning Rate Scheduler =====
    # Only one should be True at a time
    parser.add_argument(
        "--cos_restart_cyclic", type=bool, default=False
    )  # Cyclic cosine annealing
    parser.add_argument(
        "--cos_restart", type=bool, default=True
    )  # Cosine annealing with restarts

    # ===== Warmup Training =====
    parser.add_argument("--warmup_epochs", type=int, default=3, help="warmup_epochs")
    parser.add_argument(
        "--start_warmup",
        type=bool,
        default=True,
        help="turn False to train without warmup",
    )

    # ===== Training Dataset Paths =====
    parser.add_argument(
        "--data_train_lol_blur", type=str, default="./datasets/LOL_blur/train"
    )
    parser.add_argument(
        "--data_train_lol_v1", type=str, default="./datasets/LOLdataset/our485"
    )
    parser.add_argument(
        "--data_train_lolv2_real",
        type=str,
        default="./datasets/LOLv2/Real_captured/Train",
    )
    parser.add_argument(
        "--data_train_lolv2_syn", type=str, default="./datasets/LOLv2/Synthetic/Train"
    )
    parser.add_argument(
        "--data_train_SID", type=str, default="./datasets/Sony_total_dark/train"
    )
    parser.add_argument(
        "--data_train_SICE", type=str, default="./datasets/SICE/Dataset/train"
    )
    parser.add_argument(
        "--data_train_fivek", type=str, default="./datasets/FiveK/train"
    )

    # ===== Validation Input Paths (Low-light Images) =====
    parser.add_argument(
        "--data_val_lol_blur", type=str, default="./datasets/LOL_blur/eval/low_blur"
    )
    parser.add_argument(
        "--data_val_lol_v1", type=str, default="./datasets/LOLdataset/eval15/low"
    )
    parser.add_argument(
        "--data_val_lolv2_real",
        type=str,
        default="./datasets/LOLv2/Real_captured/Test/Low",
    )
    parser.add_argument(
        "--data_val_lolv2_syn", type=str, default="./datasets/LOLv2/Synthetic/Test/Low"
    )
    parser.add_argument(
        "--data_val_SID", type=str, default="./datasets/Sony_total_dark/eval/short"
    )
    parser.add_argument(
        "--data_val_SICE_mix", type=str, default="./datasets/SICE/Dataset/eval/test"
    )
    parser.add_argument(
        "--data_val_SICE_grad", type=str, default="./datasets/SICE/Dataset/eval/test"
    )
    parser.add_argument(
        "--data_test_fivek", type=str, default="./datasets/FiveK/test/input"
    )

    # ===== Validation Ground Truth Paths (Normal-light Images) =====
    parser.add_argument(
        "--data_valgt_lol_blur",
        type=str,
        default="./datasets/LOL_blur/eval/high_sharp_scaled/",
    )
    parser.add_argument(
        "--data_valgt_lol_v1", type=str, default="./datasets/LOLdataset/eval15/high/"
    )
    parser.add_argument(
        "--data_valgt_lolv2_real",
        type=str,
        default="./datasets/LOLv2/Real_captured/Test/Normal/",
    )
    parser.add_argument(
        "--data_valgt_lolv2_syn",
        type=str,
        default="./datasets/LOLv2/Synthetic/Test/Normal/",
    )
    parser.add_argument(
        "--data_valgt_SID", type=str, default="./datasets/Sony_total_dark/eval/long/"
    )
    parser.add_argument(
        "--data_valgt_SICE_mix",
        type=str,
        default="./datasets/SICE/Dataset/eval/target/",
    )
    parser.add_argument(
        "--data_valgt_SICE_grad",
        type=str,
        default="./datasets/SICE/Dataset/eval/target/",
    )
    parser.add_argument(
        "--data_valgt_fivek", type=str, default="./datasets/FiveK/test/target/"
    )

    parser.add_argument(
        "--val_folder",
        default="./results/",
        help="Location to save validation datasets",
    )

    # ===== Loss Function Weights =====
    parser.add_argument(
        "--HVI_weight", type=float, default=1.0
    )  # HVI space loss weight
    parser.add_argument(
        "--L1_weight", type=float, default=1.0
    )  # L1 reconstruction loss
    parser.add_argument("--D_weight", type=float, default=0.5)  # SSIM loss (D = DSSIM)
    parser.add_argument("--E_weight", type=float, default=50.0)  # Edge loss weight
    parser.add_argument(
        "--P_weight", type=float, default=1e-2
    )  # Perceptual (VGG) loss weight

    # ===== Gamma Augmentation =====
    # Random gamma correction for improved generalization
    parser.add_argument(
        "--gamma", type=bool, default=False
    )  # Enable gamma augmentation
    parser.add_argument(
        "--start_gamma", type=int, default=60
    )  # Epoch to start gamma aug
    parser.add_argument("--end_gamma", type=int, default=120)  # Epoch to end gamma aug

    # ===== Gradient Control =====
    parser.add_argument(
        "--grad_detect",
        type=bool,
        default=False,
        help="if gradient explosion occurs, turn-on it",
    )
    parser.add_argument(
        "--grad_clip",
        type=bool,
        default=True,
        help="if gradient fluctuates too much, turn-on it",
    )

    # ===== Dataset Selection =====
    # Set exactly ONE to True to select training dataset
    parser.add_argument(
        "--lol_v1", type=bool, default=True
    )  # LOLv1 dataset (485 pairs)
    parser.add_argument(
        "--lolv2_real", type=bool, default=False
    )  # LOLv2-Real (685 pairs)
    parser.add_argument(
        "--lolv2_syn", type=bool, default=False
    )  # LOLv2-Synthetic (900 pairs)
    parser.add_argument(
        "--lol_blur", type=bool, default=False
    )  # LOL-Blur (10200 pairs)
    parser.add_argument("--SID", type=bool, default=False)  # Sony SID (2099 pairs)
    parser.add_argument("--SICE_mix", type=bool, default=False)  # SICE Mixed subset
    parser.add_argument("--SICE_grad", type=bool, default=False)  # SICE Gradient subset
    parser.add_argument(
        "--fivek", type=bool, default=False
    )  # MIT-Adobe FiveK (4500 pairs)
    return parser
