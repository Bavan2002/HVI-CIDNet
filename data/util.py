# ===== Image Loading Utilities =====
# Helper functions for dataset classes

from PIL import Image


def is_image_file(filename):
    """Check if a file is an image based on extension."""
    return any(
        filename.endswith(extension)
        for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"]
    )


def load_img(filepath):
    """Load an image as RGB PIL Image."""
    img = Image.open(filepath).convert("RGB")
    return img
