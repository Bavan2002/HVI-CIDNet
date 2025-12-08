# ===== Evaluation Dataset Loaders =====
# PyTorch Dataset classes for evaluation/inference
# Returns: (input_tensor, filename) or (input_tensor, filename, h, w) for variable sizes

import os
import torch.utils.data as data
from os import listdir
from os.path import join
from data.util import *
import torch.nn.functional as F


# ===== Variable-Size Evaluation Dataset =====
class SICEDatasetFromFolderEval(data.Dataset):
    """Evaluation dataset for variable-size images (SICE, FiveK, unpaired).
    Auto-pads images to multiple of 8 for U-Net compatibility.
    Returns: (padded_input, filename, original_h, original_w)
    """

    def __init__(self, data_dir, transform=None):
        super(SICEDatasetFromFolderEval, self).__init__()
        data_filenames = [
            join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)
        ]
        data_filenames.sort()  # Ensure consistent ordering
        self.data_filenames = data_filenames
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.transform:
            input = self.transform(input)

            # Pad to multiple of 8 (required by U-Net encoder-decoder)
            factor = 8
            h, w = input.shape[1], input.shape[2]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input = F.pad(input.unsqueeze(0), (0, padw, 0, padh), "reflect").squeeze(0)

        return input, file, h, w  # Return original dims for cropping output

    def __len__(self):
        return len(self.data_filenames)


# ===== Fixed-Size Evaluation Dataset =====
class DatasetFromFolderEval(data.Dataset):
    """Evaluation dataset for fixed-size images (LOLv1, LOLv2).
    No padding needed - images are already properly sized.
    Returns: (input_tensor, filename)
    """

    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [
            join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)
        ]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.transform:
            input = self.transform(input)
        return input, file

    def __len__(self):
        return len(self.data_filenames)
