# ===== LOL Dataset Loaders =====
# PyTorch Dataset classes for LOLv1, LOLv2-Real, and LOLv2-Synthetic datasets
# Each dataset returns: (low_img, high_img, low_filename, high_filename)

import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *
from torchvision import transforms as t


# ===== LOLv1 Dataset =====
class LOLDatasetFromFolder(data.Dataset):
    """LOLv1 dataset - 485 low/high image pairs.
    Structure: data_dir/low/*.png and data_dir/high/*.png
    """

    def __init__(self, data_dir, transform=None):
        super(LOLDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        # Build paths to low and high directories
        folder = self.data_dir + "/low"
        folder2 = self.data_dir + "/high"
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [
            join(folder2, x) for x in listdir(folder2) if is_image_file(x)
        ]
        num = len(data_filenames)

        # Load image pair
        im1 = load_img(data_filenames[index])  # Low-light image
        im2 = load_img(data_filenames2[index])  # Ground truth (high-light)
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])

        # Apply same random transform to both images (synchronized augmentation)
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
        return 485  # Fixed dataset size


# ===== LOLv2-Real Dataset =====
class LOLv2DatasetFromFolder(data.Dataset):
    """LOLv2-Real dataset - 685 Low/Normal image pairs.
    Structure: data_dir/Low/*.png and data_dir/Normal/*.png
    """

    def __init__(self, data_dir, transform=None):
        super(LOLv2DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        folder = self.data_dir + "/Low"
        folder2 = self.data_dir + "/Normal"
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [
            join(folder2, x) for x in listdir(folder2) if is_image_file(x)
        ]

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])

        # Synchronized random augmentation
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 685  # Fixed dataset size


# ===== LOLv2-Synthetic Dataset =====
class LOLv2SynDatasetFromFolder(data.Dataset):
    """LOLv2-Synthetic dataset - 900 synthetically darkened Low/Normal pairs.
    Structure: data_dir/Low/*.png and data_dir/Normal/*.png
    """

    def __init__(self, data_dir, transform=None):
        super(LOLv2SynDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        folder = self.data_dir + "/Low"
        folder2 = self.data_dir + "/Normal"
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [
            join(folder2, x) for x in listdir(folder2) if is_image_file(x)
        ]

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])

        # Synchronized random augmentation
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 900  # Fixed dataset size


class LOLDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        folder = self.data_dir + "/low"
        folder2 = self.data_dir + "/high"
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [
            join(folder2, x) for x in listdir(folder2) if is_image_file(x)
        ]
        num = len(data_filenames)

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)  # make a seed with numpy generator
        if self.transform:
            random.seed(seed)  # apply this seed to img tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 485


class LOLv2DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLv2DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        folder = self.data_dir + "/Low"
        folder2 = self.data_dir + "/Normal"
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [
            join(folder2, x) for x in listdir(folder2) if is_image_file(x)
        ]

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)  # make a seed with numpy generator
        if self.transform:
            random.seed(seed)  # apply this seed to img tranforms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)  # apply this seed to img tranforms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 685


class LOLv2SynDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLv2SynDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        folder = self.data_dir + "/Low"
        folder2 = self.data_dir + "/Normal"
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [
            join(folder2, x) for x in listdir(folder2) if is_image_file(x)
        ]

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)  # make a seed with numpy generator
        if self.transform:
            random.seed(seed)  # apply this seed to img tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return 900
