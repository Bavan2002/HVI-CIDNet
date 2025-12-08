# ===== Data Loading Utilities =====
# Factory functions for creating training and evaluation datasets
# Each dataset returns: (low_light_img, ground_truth_img, filename1, filename2)

from torchvision.transforms import (
    Compose,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from data.LOLdataset import *
from data.eval_sets import *
from data.SICE_blur_SID import *
from data.fivek import *


# ===== Transform Pipelines =====
def transform1(size=256):
    """Training transform: random crop + augmentation + to tensor."""
    return Compose(
        [
            RandomCrop((size, size)),  # Extract random patch
            RandomHorizontalFlip(),  # 50% horizontal flip
            RandomVerticalFlip(),  # 50% vertical flip
            ToTensor(),  # Convert to [0,1] tensor
        ]
    )


def transform2():
    """Evaluation transform: just convert to tensor (no augmentation)."""
    return Compose([ToTensor()])


# ===== Training Dataset Factories =====
def get_lol_training_set(data_dir, size):
    """LOLv1 training set - 485 low/high pairs."""
    return LOLDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_training_set(data_dir, size):
    """LOLv2-Real training set - 685 Low/Normal pairs."""
    return LOLv2DatasetFromFolder(data_dir, transform=transform1(size))


def get_training_set_blur(data_dir, size):
    """LOL-Blur training set - 10200 low_blur/high_sharp pairs."""
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size))


def get_lol_v2_syn_training_set(data_dir, size):
    """LOLv2-Synthetic training set - 900 Low/Normal pairs."""
    return LOLv2SynDatasetFromFolder(data_dir, transform=transform1(size))


def get_SID_training_set(data_dir, size):
    """Sony SID training set - 2099 short/long exposure pairs."""
    return SIDDatasetFromFolder(data_dir, transform=transform1(size))


def get_SICE_training_set(data_dir, size):
    """SICE training set - 4803 multi-exposure images with labels."""
    return SICEDatasetFromFolder(data_dir, transform=transform1(size))


def get_fivek_training_set(data_dir, size):
    """MIT-Adobe FiveK training set - 4500 input/target pairs."""
    return FiveKDatasetFromFolder(data_dir, transform=transform1(size))


# ===== Evaluation Dataset Factories =====
def get_eval_set(data_dir):
    """Standard evaluation set - fixed size, no padding needed."""
    return DatasetFromFolderEval(data_dir, transform=transform2())


def get_SICE_eval_set(data_dir):
    """SICE/variable-size evaluation set - auto-pads to multiple of 8."""
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())


def get_fivek_eval_set(data_dir):
    """FiveK evaluation set - uses SICE loader for variable sizes."""
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())
