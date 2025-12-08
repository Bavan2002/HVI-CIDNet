"""
train.py - Main training script for HVI-CIDNet

Training pipeline for low-light image enhancement with dual-space loss (RGB + HVI).
Supports multiple datasets (LOL, LOLv2, SICE, SID, FiveK) and learning rate schedulers.
"""

import os
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/Colab
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime

# ===== Training History Tracking =====
epoch_list = []  # epoch numbers
rgb_losses = []  # RGB space loss per epoch
hvi_losses = []  # HVI space loss per epoch
total_losses = []  # combined loss per epoch
learning_rates = []  # LR per epoch

# Validation metrics (recorded at snapshot intervals)
metrics_epochs = []
psnr_list = []
ssim_list = []
lpips_list = []

opt = option().parse_args()


def seed_torch():
    """Set random seeds for reproducibility across all libraries."""
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_init():
    """Initialize training environment: seeds, CUDA, cudnn."""
    seed_torch()
    cudnn.benchmark = True  # optimize convolution algorithms
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")


def train(epoch):
    """
    Run one training epoch.

    Returns: (total_loss, rgb_loss, hvi_loss, num_samples)
    """
    model.train()
    loss_print = 0
    loss_rgb_print = 0
    loss_hvi_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    pic_last_10 = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)  # debug gradient issues

    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()  # low-light input
        im2 = im2.cuda()  # ground truth

        # Optional: apply random gamma curve for data augmentation
        if opt.gamma:
            gamma = random.randint(opt.start_gamma, opt.end_gamma) / 100.0
            output_rgb = model(im1**gamma)
        else:
            output_rgb = model(im1)

        gt_rgb = im2

        # Convert to HVI space for dual-space loss
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)

        # HVI space loss: L1 + SSIM + Edge + Perceptual
        loss_hvi = (
            L1_loss(output_hvi, gt_hvi)
            + D_loss(output_hvi, gt_hvi)
            + E_loss(output_hvi, gt_hvi)
            + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        )

        # RGB space loss: L1 + SSIM + Edge + Perceptual
        loss_rgb = (
            L1_loss(output_rgb, gt_rgb)
            + D_loss(output_rgb, gt_rgb)
            + E_loss(output_rgb, gt_rgb)
            + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        )

        # Total loss: RGB + weighted HVI
        loss = loss_rgb + opt.HVI_weight * loss_hvi
        iter += 1

        # Gradient clipping to prevent exploding gradients
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses for logging
        loss_print = loss_print + loss.item()
        loss_rgb_print = loss_rgb_print + loss_rgb.item()
        loss_hvi_print = loss_hvi_print + loss_hvi.item()
        loss_last_10 = loss_last_10 + loss.item()
        pic_cnt += 1
        pic_last_10 += 1

        # Print stats at end of epoch
        if iter == train_len:
            print(
                "===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={}.".format(
                    epoch, loss_last_10 / pic_last_10, optimizer.param_groups[0]["lr"]
                )
            )
            loss_last_10 = 0
            pic_last_10 = 0

            # Save sample output for visual inspection
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder + "training"):
                os.mkdir(opt.val_folder + "training")
            output_img.save(opt.val_folder + "training/test.png")
            gt_img.save(opt.val_folder + "training/gt.png")

    return loss_print, loss_rgb_print, loss_hvi_print, pic_cnt


def checkpoint(epoch):
    """
    Save training checkpoint including model, optimizer, scheduler, and history.

    Saves two files:
    - epoch_N.pth: model weights only (for inference)
    - checkpoint_epoch_N.pth: full state (for resuming training)
    """
    if not os.path.exists("./weights"):
        os.mkdir("./weights")
    if not os.path.exists("./weights/train"):
        os.mkdir("./weights/train")

    # Model weights only (inference)
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

    # Full checkpoint (resume training)
    full_checkpoint_path = "./weights/train/checkpoint_epoch_{}.pth".format(epoch)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            # Loss history
            "epoch_list": epoch_list,
            "rgb_losses": rgb_losses,
            "hvi_losses": hvi_losses,
            "total_losses": total_losses,
            "learning_rates": learning_rates,
            # Metrics history
            "metrics_epochs": metrics_epochs,
            "psnr_list": psnr_list,
            "ssim_list": ssim_list,
            "lpips_list": lpips_list,
        },
        full_checkpoint_path,
    )

    print("Checkpoint saved to {}".format(model_out_path))
    print("Full checkpoint saved to {}".format(full_checkpoint_path))
    return model_out_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load full training state from checkpoint for resuming training."""
    global epoch_list, rgb_losses, hvi_losses, total_losses, learning_rates
    global metrics_epochs, psnr_list, ssim_list, lpips_list

    print("Loading checkpoint from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore loss history
    epoch_list = checkpoint["epoch_list"]
    rgb_losses = checkpoint["rgb_losses"]
    hvi_losses = checkpoint["hvi_losses"]
    total_losses = checkpoint["total_losses"]
    learning_rates = checkpoint["learning_rates"]

    # Restore metrics history
    metrics_epochs = checkpoint["metrics_epochs"]
    psnr_list = checkpoint["psnr_list"]
    ssim_list = checkpoint["ssim_list"]
    lpips_list = checkpoint["lpips_list"]

    print("Checkpoint loaded. Resuming from epoch {}".format(checkpoint["epoch"]))
    return checkpoint["epoch"]


def plot_loss(
    epochs, rgb_loss, hvi_loss, total_loss, save_path="./weights/train/loss_graph.png"
):
    """Plot RGB, HVI, and total loss curves over training."""
    if not os.path.exists("./weights/train"):
        os.mkdir("./weights/train")
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rgb_loss, "b-", linewidth=2, label="RGB Loss")
    plt.plot(epochs, hvi_loss, "orange", linewidth=2, label="HVI Loss")
    plt.plot(epochs, total_loss, "g-", linewidth=2, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Loss graph saved to {}".format(save_path))


def plot_lr(epochs, lrs, save_path="./weights/train/lr_graph.png"):
    """Plot learning rate schedule (log scale)."""
    if not os.path.exists("./weights/train"):
        os.mkdir("./weights/train")
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs, "r-", linewidth=2, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Learning rate graph saved to {}".format(save_path))


def plot_metrics(
    epochs,
    psnr_list,
    ssim_list,
    lpips_list,
    save_path="./weights/train/metrics_graph.png",
):
    """Plot PSNR, SSIM, LPIPS validation metrics (dual y-axis)."""
    if not os.path.exists("./weights/train"):
        os.mkdir("./weights/train")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # PSNR on left y-axis
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PSNR (dB)", color="blue")
    ax1.plot(epochs, psnr_list, "b-", linewidth=2, label="PSNR")
    ax1.tick_params(axis="y", labelcolor="blue")

    # SSIM and LPIPS on right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("SSIM / LPIPS", color="green")
    ax2.plot(epochs, ssim_list, "g-", linewidth=2, label="SSIM")
    ax2.plot(epochs, lpips_list, "r-", linewidth=2, label="LPIPS")
    ax2.tick_params(axis="y", labelcolor="green")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Validation Metrics Over Time")
    plt.grid(True)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Metrics graph saved to {}".format(save_path))


def load_datasets():
    """
    Load training and validation datasets based on command line options.

    Supports: LOLv1, LOLv2-real, LOLv2-syn, LOL-blur, SID, SICE, FiveK
    """
    print("===> Loading datasets")
    if (
        opt.lol_v1
        or opt.lol_blur
        or opt.lolv2_real
        or opt.lolv2_syn
        or opt.SID
        or opt.SICE_mix
        or opt.SICE_grad
        or opt.fivek
    ):
        # LOLv1 dataset (485 training pairs)
        if opt.lol_v1:
            train_set = get_lol_training_set(opt.data_train_lol_v1, size=opt.cropSize)
            training_data_loader = DataLoader(
                dataset=train_set,
                num_workers=opt.threads,
                batch_size=opt.batchSize,
                shuffle=opt.shuffle,
            )
            test_set = get_eval_set(opt.data_val_lol_v1)
            testing_data_loader = DataLoader(
                dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False
            )

        # LOL-blur dataset (motion blur + low-light)
        if opt.lol_blur:
            train_set = get_training_set_blur(
                opt.data_train_lol_blur, size=opt.cropSize
            )
            training_data_loader = DataLoader(
                dataset=train_set,
                num_workers=opt.threads,
                batch_size=opt.batchSize,
                shuffle=opt.shuffle,
            )
            test_set = get_eval_set(opt.data_val_lol_blur)
            testing_data_loader = DataLoader(
                dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False
            )

        # LOLv2-real dataset (685 real captured pairs)
        if opt.lolv2_real:
            train_set = get_lol_v2_training_set(
                opt.data_train_lolv2_real, size=opt.cropSize
            )
            training_data_loader = DataLoader(
                dataset=train_set,
                num_workers=opt.threads,
                batch_size=opt.batchSize,
                shuffle=opt.shuffle,
            )
            test_set = get_eval_set(opt.data_val_lolv2_real)
            testing_data_loader = DataLoader(
                dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False
            )

        # LOLv2-synthetic dataset (900 synthetic pairs)
        if opt.lolv2_syn:
            train_set = get_lol_v2_syn_training_set(
                opt.data_train_lolv2_syn, size=opt.cropSize
            )
            training_data_loader = DataLoader(
                dataset=train_set,
                num_workers=opt.threads,
                batch_size=opt.batchSize,
                shuffle=opt.shuffle,
            )
            test_set = get_eval_set(opt.data_val_lolv2_syn)
            testing_data_loader = DataLoader(
                dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False
            )

        # SID (Sony) dataset - extreme low-light RAW
        if opt.SID:
            train_set = get_SID_training_set(opt.data_train_SID, size=opt.cropSize)
            training_data_loader = DataLoader(
                dataset=train_set,
                num_workers=opt.threads,
                batch_size=opt.batchSize,
                shuffle=opt.shuffle,
            )
            test_set = get_eval_set(opt.data_val_SID)
            testing_data_loader = DataLoader(
                dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False
            )

        # SICE dataset - multi-exposure
        if opt.SICE_mix:
            train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
            training_data_loader = DataLoader(
                dataset=train_set,
                num_workers=opt.threads,
                batch_size=opt.batchSize,
                shuffle=opt.shuffle,
            )
            test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
            testing_data_loader = DataLoader(
                dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False
            )

        if opt.SICE_grad:
            train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
            training_data_loader = DataLoader(
                dataset=train_set,
                num_workers=opt.threads,
                batch_size=opt.batchSize,
                shuffle=opt.shuffle,
            )
            test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
            testing_data_loader = DataLoader(
                dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False
            )

        # MIT-Adobe FiveK dataset
        if opt.fivek:
            train_set = get_fivek_training_set(opt.data_train_fivek, size=opt.cropSize)
            training_data_loader = DataLoader(
                dataset=train_set,
                num_workers=opt.threads,
                batch_size=opt.batchSize,
                shuffle=opt.shuffle,
            )
            test_set = get_fivek_eval_set(opt.data_val_fivek)
            testing_data_loader = DataLoader(
                dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False
            )
    else:
        raise Exception("should choose a dataset")
    return training_data_loader, testing_data_loader


def build_model():
    """Initialize CIDNet model on GPU."""
    print("===> Building model ")
    model = CIDNet().cuda()
    return model


def make_scheduler():
    """
    Create optimizer and learning rate scheduler.

    Supports:
    - CosineAnnealingRestartCyclicLR: two-phase cosine with different min LRs
    - CosineAnnealingRestartLR: single cosine annealing
    - Optional warmup phase
    """
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    if opt.cos_restart_cyclic:
        # Two-phase: fast decay then slow decay
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[
                    (opt.nEpochs // 4) - opt.warmup_epochs,
                    (opt.nEpochs * 3) // 4,
                ],
                restart_weights=[1, 1],
                eta_mins=[0.0002, 0.0000001],
            )
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=opt.warmup_epochs,
                after_scheduler=scheduler_step,
            )
        else:
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[opt.nEpochs // 4, (opt.nEpochs * 3) // 4],
                restart_weights=[1, 1],
                eta_mins=[0.0002, 0.0000001],
            )
    elif opt.cos_restart:
        # Single cosine annealing
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[opt.nEpochs - opt.warmup_epochs],
                restart_weights=[1],
                eta_min=1e-7,
            )
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=opt.warmup_epochs,
                after_scheduler=scheduler_step,
            )
        else:
            scheduler = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[opt.nEpochs],
                restart_weights=[1],
                eta_min=1e-7,
            )
    else:
        raise Exception("should choose a scheduler")
    return optimizer, scheduler


def init_loss():
    """
    Initialize loss functions with configured weights.

    Loss components:
    - L1: pixel-wise absolute difference (weight=1.0)
    - D (SSIM): structural similarity (weight=0.5)
    - E (Edge): Laplacian edge loss (weight=50.0)
    - P (Perceptual): VGG feature loss (weight=0.01)
    """
    L1_weight = opt.L1_weight
    D_weight = opt.D_weight
    E_weight = opt.E_weight
    P_weight = 1.0

    L1_loss = L1Loss(loss_weight=L1_weight, reduction="mean").cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss(
        {"conv1_2": 1, "conv2_2": 1, "conv3_4": 1, "conv4_4": 1},  # VGG layers
        perceptual_weight=P_weight,
        criterion="mse",
    ).cuda()
    return L1_loss, P_loss, E_loss, D_loss


# ===== Main Training Loop =====
if __name__ == "__main__":
    # Initialize
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer, scheduler = make_scheduler()
    L1_loss, P_loss, E_loss, D_loss = init_loss()

    # Metrics tracking
    psnr = []
    ssim = []
    lpips = []
    start_epoch = 0

    # Resume from checkpoint if specified
    if opt.start_epoch > 0:
        checkpoint_path = f"./weights/train/checkpoint_epoch_{opt.start_epoch}.pth"
        if os.path.exists(checkpoint_path):
            start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
            psnr = psnr_list.copy()
            ssim = ssim_list.copy()
            lpips = lpips_list.copy()
        else:
            # Fallback: load model weights only (old checkpoint format)
            print("Full checkpoint not found, loading model weights only...")
            pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
            model.load_state_dict(
                torch.load(pth, map_location=lambda storage, loc: storage)
            )
            start_epoch = opt.start_epoch
            # Advance scheduler to match resumed epoch
            print(f"Advancing scheduler to epoch {start_epoch}...")
            for _ in range(start_epoch):
                scheduler.step()
            print(
                f"Scheduler advanced. Current LR: {optimizer.param_groups[0]['lr']:.6e}"
            )

    if not os.path.exists(opt.val_folder):
        os.mkdir(opt.val_folder)

    # Training loop
    for epoch in range(start_epoch + 1, opt.nEpochs + 1):
        epoch_loss, epoch_rgb_loss, epoch_hvi_loss, pic_num = train(epoch)
        scheduler.step()

        # Track losses and learning rate
        epoch_list.append(epoch)
        total_losses.append(epoch_loss / pic_num)
        rgb_losses.append(epoch_rgb_loss / pic_num)
        hvi_losses.append(epoch_hvi_loss / pic_num)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        # Periodic validation and checkpointing
        if epoch % opt.snapshots == 0:
            plot_loss(epoch_list, rgb_losses, hvi_losses, total_losses)
            plot_lr(epoch_list, learning_rates)
            model_out_path = checkpoint(epoch)
            norm_size = True

            # Set output folder and GT path based on dataset
            if opt.lol_v1:
                output_folder = "LOLv1/"
                label_dir = opt.data_valgt_lol_v1
            if opt.lolv2_real:
                output_folder = "LOLv2_real/"
                label_dir = opt.data_valgt_lolv2_real
            if opt.lolv2_syn:
                output_folder = "LOLv2_syn/"
                label_dir = opt.data_valgt_lolv2_syn
            if opt.lol_blur:
                output_folder = "LOL_blur/"
                label_dir = opt.data_valgt_lol_blur
            if opt.SID:
                output_folder = "SID/"
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.SICE_mix:
                output_folder = "SICE_mix/"
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.SICE_grad:
                output_folder = "SICE_grad/"
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
            if opt.fivek:
                output_folder = "fivek/"
                label_dir = opt.data_valgt_fivek
                norm_size = False

            # Run evaluation
            im_dir = opt.val_folder + output_folder + "*.png"
            eval(
                model,
                testing_data_loader,
                model_out_path,
                opt.val_folder + output_folder,
                norm_size=norm_size,
                LOL=opt.lol_v1,
                v2=opt.lolv2_real,
                alpha=0.8,
            )

            # Compute metrics
            avg_psnr, avg_ssim, avg_lpips = metrics(
                im_dir, label_dir, use_GT_mean=False
            )
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)

            # Track metrics for plotting
            metrics_epochs.append(epoch)
            psnr_list.append(avg_psnr)
            ssim_list.append(avg_ssim)
            lpips_list.append(avg_lpips)
            plot_metrics(metrics_epochs, psnr_list, ssim_list, lpips_list)

            print(psnr)
            print(ssim)
            print(lpips)
        torch.cuda.empty_cache()

    # Save final training report
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    with open(f"./results/training/metrics{now}.md", "w") as f:
        f.write("dataset: " + output_folder + "\n")
        f.write(f"lr: {opt.lr}\n")
        f.write(f"batch size: {opt.batchSize}\n")
        f.write(f"crop size: {opt.cropSize}\n")
        f.write(f"HVI_weight: {opt.HVI_weight}\n")
        f.write(f"L1_weight: {opt.L1_weight}\n")
        f.write(f"D_weight: {opt.D_weight}\n")
        f.write(f"E_weight: {opt.E_weight}\n")
        f.write(f"P_weight: {opt.P_weight}\n")
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")
        f.write(
            "|----------------------|----------------------|----------------------|----------------------|\n"
        )
        for i in range(len(psnr)):
            f.write(
                f"| {opt.start_epoch + (i + 1) * opt.snapshots} | {psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} |\n"
            )
