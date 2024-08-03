##########################################################################
# Description: Contains functions for image processing
# Notes:
##########################################################################

from glob import glob
from typing import Union, Tuple, List
import numpy as np
from PIL import Image


def save_image(img: Union[np.ndarray, Image.Image], path: str) -> None:
    """Save an image to a file."""
    _conv_to_pil(img).save(path)

def load_image(path: str, as_float32: bool = True, channels_first: bool = False) -> np.ndarray:
    """Load an image from a file and convert it to a numpy array."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    if as_float32:
        arr = arr.astype(np.float32) / 255.0

    if channels_first:
        arr = np.moveaxis(arr, -1, 0)

    return arr

def show_image(img: Union[np.ndarray, Image.Image, str]) -> None:
    """Display an image."""
    if isinstance(img, str):
        img = load_image(img)

    _conv_to_pil(img).show()

def resize_img(img: Union[np.ndarray, Image.Image], size: Tuple[int, int]) -> np.ndarray:
    """Resize an image to the specified dimensions."""
    pil_img = _conv_to_pil(img)
    resized = pil_img.resize(size, Image.ANTIALIAS)

    if isinstance(img, Image.Image):
        return resized
    else:
        return _from_pil(resized, img.dtype == np.float32)


def _conv_to_float32(arr: np.ndarray) -> np.ndarray:
    """Convert image array to float32 format."""
    return arr.astype('float32') / 255.0 if arr.dtype != np.float32 else arr


def _conv_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert image array to uint8 format."""
    return (arr * 255).astype('uint8') if arr.dtype != np.uint8 else arr


def _conv_to_pil(img: Union[np.ndarray, Image.Image]) -> Image.Image:
    """Convert image to PIL Image format."""
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(_conv_to_uint8(img))

def list_images(directory: str) -> List[str]:
    """List all image files in a directory."""
    return sorted(filter(is_image, glob(f"{directory}/*.*")))

def is_image(filepath: str) -> bool:
    """Check if a file is an image based on its extension."""
    return filepath.lower().endswith(('.jpg', '.jpeg', '.png'))

def _from_pil(img: Union[np.ndarray, Image.Image], as_float32: bool = True) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    arr = np.array(img)
    return _conv_to_float32(arr) if as_float32 else arr

