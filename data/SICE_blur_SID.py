# ===== LOL-Blur, SID, and SICE Dataset Loaders =====
# PyTorch Dataset classes for specialized low-light datasets
# Each returns: (low_img, high_img, low_filename, high_filename)

import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from data.util import *
from torchvision import transforms as t
import torch.nn.functional as F


# ===== LOL-Blur Dataset =====
class LOLBlurDatasetFromFolder(data.Dataset):
    """LOL-Blur dataset - 10200 low_blur/high_sharp_scaled image pairs.
    Structure: data_dir/low_blur/XXXX/*.png and data_dir/high_sharp_scaled/XXXX/*.png
    Randomly samples from 260 scenes, each with multiple blur levels.
    """

    def __init__(self, data_dir, transform=None):
        super(LOLBlurDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        # Random scene selection (0-259)
        while True:
            seed = random.randint(1, 1000000)
            random.seed(seed)
            index = random.randint(0, 259)
            fill_index = str(index + 1).zfill(
                4
            )  # Zero-pad to 4 digits: 0001, 0002, etc.
            folder = join(self.data_dir + "/low_blur", fill_index)
            folder2 = join(self.data_dir + "/high_sharp_scaled", fill_index)
            if not os.path.exists(folder):
                continue
            data_filenames = [
                join(folder, x) for x in listdir(folder) if is_image_file(x)
            ]
            data_filenames2 = [
                join(folder2, x) for x in listdir(folder2) if is_image_file(x)
            ]
            num = len(data_filenames)
            if num != 0:
                break

        # Random image within scene
        index1 = random.randint(1, num)

        im1 = load_img(data_filenames[index1 - 1])  # Blurry low-light
        im2 = load_img(data_filenames2[index1 - 1])  # Sharp normal-light

        # Synchronized augmentation
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
        return im1, im2, data_filenames[index1 - 1], data_filenames2[index1 - 1]

    def __len__(self):
        return 10200  # Total image pairs across all scenes


# ===== Sony SID Dataset =====
class SIDDatasetFromFolder(data.Dataset):
    """Sony SID (See-in-the-Dark) dataset - 2099 short/long exposure pairs.
    Structure: data_dir/short/XXXXX/*.png and data_dir/long/XXXXX/*.png
    Each scene has multiple short exposures but one long exposure reference.
    """

    def __init__(self, data_dir, transform=None):
        super(SIDDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        # Random scene selection (0-233)
        while True:
            seed = random.randint(1, 1000000)
            random.seed(seed)
            index = random.randint(0, 233)
            fill_index = str(index + 1).zfill(5)  # Zero-pad to 5 digits
            folder = join(self.data_dir + "/short", fill_index)
            folder2 = join(self.data_dir + "/long", fill_index)
            if os.path.exists(folder):
                data_filenames = [
                    join(folder, x) for x in listdir(folder) if is_image_file(x)
                ]
                data_filenames2 = [
                    join(folder2, x) for x in listdir(folder2) if is_image_file(x)
                ]
                num = len(data_filenames)
                break
            else:
                continue

        # Random short exposure, but always use first long exposure as GT
        index1 = random.randint(1, num)

        im1 = load_img(data_filenames[index1 - 1])  # Short exposure (dark)
        im2 = load_img(data_filenames2[0])  # Long exposure (GT) - always first
        _, file1 = os.path.split(data_filenames[index1 - 1])
        _, file2 = os.path.split(data_filenames2[0])

        # Synchronized augmentation
        seed = np.random.randint(random.randint(1, 1000000))
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 2099


# ===== SICE Dataset =====
class SICEDatasetFromFolder(data.Dataset):
    """SICE (Single Image Contrast Enhancement) dataset - 4803 multi-exposure images.
    Structure: data_dir/train/XXX/*.JPG and data_dir/label/XXX.JPG
    Each scene folder contains multiple exposures; label is the reference.
    """

    def __init__(self, data_dir, transform=None):
        super(SICEDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        # Random scene selection (0-590)
        while True:
            seed = random.randint(1, 1000000)
            random.seed(seed)
            index = random.randint(0, 590)
            fill_index = str(index + 1)  # Scene number (no padding)
            train, tail = os.path.split(self.data_dir)
            folder = join(self.data_dir, fill_index)
            data_gt = join(train + "/label", fill_index + ".JPG")  # Ground truth label
            if os.path.exists(folder):
                data_filenames = [
                    join(folder, x) for x in listdir(folder) if is_image_file(x)
                ]
                num = len(data_filenames)
                break
            else:
                continue

        # Random exposure level within scene
        index1 = random.randint(1, num)

        im1 = load_img(data_filenames[index1 - 1])  # Random exposure
        im2 = load_img(data_gt)  # Reference label
        _, file1 = os.path.split(data_filenames[index1 - 1])
        _, file2 = os.path.split(data_gt)

        # Synchronized augmentation
        seed = np.random.randint(random.randint(1, 1000000))
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 4803  # Total images across all scenes
