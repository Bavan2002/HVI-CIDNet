# ===== MIT-Adobe FiveK Dataset Loader =====
# PyTorch Dataset class for FiveK dataset (following Retinexformer structure)
# Reference: https://github.com/caiyuanhao1998/Retinexformer

import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *


class FiveKDatasetFromFolder(data.Dataset):
    """MIT-Adobe FiveK dataset - 4500 input/target image pairs for training.
    Structure: data_dir/input/*.jpg and data_dir/target/*.jpg
    Input images are raw/underexposed, targets are expert-retouched.
    """

    def __init__(self, data_dir, transform=None):
        super(FiveKDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        # Build paths to input and target directories
        folder = self.data_dir + "/input"
        folder2 = self.data_dir + "/target"
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [
            join(folder2, x) for x in listdir(folder2) if is_image_file(x)
        ]

        # Load image pair
        im1 = load_img(data_filenames[index])  # Input (raw/underexposed)
        im2 = load_img(data_filenames2[index])  # Target (expert-retouched)
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])

        # Synchronized random augmentation for both images
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)  # Make seed with numpy generator
        if self.transform:
            random.seed(seed)  # Set seed for reproducible transforms
            torch.manual_seed(seed)  # Needed for torchvision 0.7+
            im1 = self.transform(im1)
            random.seed(seed)  # Reset seed for identical transform
            torch.manual_seed(seed)
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 4500  # Training set size
