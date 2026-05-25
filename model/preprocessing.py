"""Image preprocessing utilities shared by training and prediction scripts."""

from pathlib import Path

import cv2
import numpy as np


def preprocess_image_array(image_array, image_size=(28, 28), invert_if_needed=False):
    """
    Convert an image array into a normalized 2D grayscale image.

    Args:
        image_array (np.ndarray): Input image (gray or BGR).
        image_size (tuple): Output size as (width, height).
        invert_if_needed (bool): Invert image if background is bright.

    Returns:
        np.ndarray: Normalized image with shape (height, width), values in [0, 1].
    """
    if image_array is None:
        raise ValueError("Input image array is empty.")

    # Convert to grayscale if the input image has color channels.
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array

    # If image looks like dark text on white paper, invert to match MNIST-like style.
    if invert_if_needed and float(np.mean(gray)) > 127.0:
        gray = cv2.bitwise_not(gray)

    resized = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    normalized = denoised.astype("float32") / 255.0
    return normalized


def preprocess_image_file(image_path, image_size=(28, 28), invert_if_needed=True):
    """
    Read an image file and return a normalized 2D image.

    Args:
        image_path (str | Path): Path to image file.
        image_size (tuple): Output size as (width, height).
        invert_if_needed (bool): Invert image if background is bright.

    Returns:
        np.ndarray: Normalized image with shape (height, width), values in [0, 1].
    """
    image_path = Path(image_path)
    image_array = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_array is None:
        raise ValueError(f"Could not read image file: {image_path}")
    return preprocess_image_array(
        image_array=image_array,
        image_size=image_size,
        invert_if_needed=invert_if_needed,
    )


def add_channel_dimension(image_2d):
    """Convert (H, W) image to (H, W, 1) format expected by CNNs."""
    return np.expand_dims(image_2d, axis=-1)


def prepare_single_image_for_model(image_2d):
    """Convert one 2D image into model batch shape: (1, H, W, 1)."""
    image_3d = add_channel_dimension(image_2d)
    return np.expand_dims(image_3d, axis=0)

