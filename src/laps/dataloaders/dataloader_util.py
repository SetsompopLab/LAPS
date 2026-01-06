"""
Dataset-related utilities with torchvision transforms.

This module provides utilities for data loading and augmentation,
migrated from MONAI to torchvision transforms where possible.
"""

import math
from typing import List, Sequence, Union

import numpy as np
import torch
import torchvision.transforms as T
from torch import nn

__all__ = [
    "PadToSquare",
    "get_augmentation_transforms",
    "LoadComplexImage",
    "RandomTranspose",
]


class PadToSquare(nn.Module):
    """Pad image to square shape to prevent distortion during resizing."""

    def __init__(self, fill: Union[int, float] = 0, mode: str = "reflect"):
        super().__init__()
        self.fill = int(fill)
        self.mode = mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape (C, H, W)

        Returns:
            Padded tensor of shape (C, max(H,W), max(H,W))
        """
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3D tensor (C, H, W), got shape {image.shape}")

        height, width = image.shape[1:]
        max_size = max(width, height)

        left_padding = (max_size - width) // 2
        right_padding = max_size - width - left_padding
        top_padding = (max_size - height) // 2
        bottom_padding = max_size - height - top_padding

        # torchvision Pad expects (left, top, right, bottom)
        padding = (left_padding, top_padding, right_padding, bottom_padding)
        pad_transform = T.Pad(padding, fill=self.fill, padding_mode=self.mode)

        return pad_transform(image)


class RandomTranspose(nn.Module):
    """Randomly transpose spatial dimensions with given probability."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: Input tensor of shape (C, H, W)

        Returns:
            Transposed tensor of shape (C, W, H) or original tensor
        """
        if torch.rand(1) < self.p:
            return tensor.transpose(-2, -1)  # Transpose last two dimensions
        return tensor


def get_augmentation_transforms(
    random_flip: bool = False,
    rotation_degrees: float = 10.0,
    p_hflip: float = 0.2,
    p_vflip: float = 0.2,
    p_transpose: float = 0.2,
    p_rotation: float = 0.2,
) -> List[nn.Module]:
    """
    Create list of augmentation transforms using torchvision.

    Args:
        random_flip: Whether to include random flips
        rotation_degrees: Maximum rotation angle in degrees
        p_flip: Probability for flip transforms
        p_transpose: Probability for transpose transform
        p_rotation: Probability for applying rotation (to avoid interpolation artifacts)

    Returns:
        List of transform modules
    """
    transforms = []

    if random_flip:
        transforms.extend(
            [
                T.RandomVerticalFlip(p=p_vflip),
                T.RandomHorizontalFlip(p=p_hflip),
                RandomTranspose(p=p_transpose),
            ]
        )

    if rotation_degrees > 0:
        # Wrap rotation in RandomApply to make it probabilistic
        rotation_transform = T.RandomRotation(
            degrees=(-rotation_degrees, rotation_degrees),
            interpolation=T.InterpolationMode.BILINEAR,
        )
        transforms.append(T.RandomApply([rotation_transform], p=p_rotation))

    return transforms


class LoadComplexImage(nn.Module):
    """
    Load and preprocess complex-valued images from numpy arrays.

    This transform loads complex data from .npy files and handles:
    - Normalization by maximum absolute value
    - Optional global phase modulation for augmentation
    - Complex dropout (returning magnitude only)
    - Conversion to real/imaginary channel representation
    """

    def __init__(
        self,
        complex_dropout_frac: float = 0.0,
        complex_global_phase_modulation: bool = False,
    ):
        """
        Args:
            complex_dropout_frac: Fraction of time to return magnitude only [0, 1]
            complex_global_phase_modulation: Whether to apply random global phase shifts
        """
        super().__init__()
        self.complex_dropout_frac = complex_dropout_frac
        self.complex_global_phase_modulation = complex_global_phase_modulation

    def forward(self, path: str) -> torch.Tensor:
        """
        Load complex image from file path.

        Args:
            path: Path to .npy file containing complex data

        Returns:
            Tensor of shape (2, H, W) with real and imaginary parts

        Raises:
            ValueError: If data is not complex-valued
            FileNotFoundError: If file doesn't exist
        """
        try:
            data = np.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not load file: {path}")

        if not np.iscomplexobj(data):
            raise ValueError(f"Data in {path} is not complex-valued.")

        # Normalize by maximum absolute value
        max_abs = np.max(np.abs(data))
        if max_abs > 0:
            data = data / max_abs

        # Apply global phase modulation for augmentation
        if self.complex_global_phase_modulation:
            phase = np.random.rand() * 2 * np.pi
            data = data * np.exp(1j * phase)

        # Decide whether to keep complex data or use magnitude only
        use_complex = np.random.rand() > self.complex_dropout_frac

        if use_complex:
            # Stack real and imaginary parts
            real_part = data.real[np.newaxis, ...]
            imag_part = data.imag[np.newaxis, ...]
            stacked_data = np.concatenate([real_part, imag_part], axis=0)
        else:
            # Use magnitude only, set imaginary part to zero
            magnitude = np.abs(data)[np.newaxis, ...]
            zeros = np.zeros_like(magnitude)
            stacked_data = np.concatenate([magnitude, zeros], axis=0)

        return torch.from_numpy(stacked_data.astype(np.float32))


def convert_real_to_complex(
    img: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert single-channel real image to complex representation with random smooth phase.

    Args:
        img: Real-valued tensor of shape (1, H, W)

    Returns:
        Complex representation tensor of shape (2, H, W) where:
        - Channel 0: real part (original magnitude)
        - Channel 1: imaginary part (magnitude * sin(phase))
    """
    if img.shape[0] != 1:
        return img

    assert img.min() >= 0, "Image must be non-negative"

    # Generate random smooth phase
    phase = generate_random_smooth_phase(img.shape[1], img.shape[2], dtype=dtype)

    # Create complex representation: magnitude * exp(i * phase)
    magnitude = img[0]  # Shape (H, W)
    real_part = magnitude * torch.cos(phase)
    imag_part = magnitude * torch.sin(phase)

    # Stack to create 2-channel complex representation
    complex_img = torch.stack([real_part, imag_part], dim=0)  # Shape (2, H, W)

    return complex_img


def generate_random_smooth_phase(
    height: int, width: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate a random smooth phase with linear and optional second-order components.

    Ensures less than 1 phase wrap (< 2π) across the image.
    80% of the time uses only linear gradient, 20% adds second-order term.

    Args:
        height: Image height
        width: Image width
        dtype: Output tensor dtype

    Returns:
        Phase tensor of shape (height, width) with values in [-π, π]
    """
    # Create coordinate grids
    y_coords = torch.linspace(-1, 1, height, dtype=dtype)
    x_coords = torch.linspace(-1, 1, width, dtype=dtype)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

    center_y = torch.rand(1, dtype=dtype) * 1.5 - 0.75  # [-0.75, 0.75]
    center_x = torch.rand(1, dtype=dtype) * 1.5 - 0.75  # [-0.75, 0.75]

    Y_shifted = Y - center_y
    X_shifted = X - center_x

    max_gradient = np.pi * 0.5  # 60% of π to be safe
    grad_y = (torch.rand(1, dtype=dtype) * 2 - 1) * max_gradient  # [-0.6π, 0.6π]
    grad_x = (torch.rand(1, dtype=dtype) * 2 - 1) * max_gradient  # [-0.6π, 0.6π]

    # Linear phase component
    phase = grad_y * Y_shifted + grad_x * X_shifted

    # 20% of the time, add second-order term
    if torch.rand(1) < 0.2:
        max_second_order = np.pi * 0.2  # Smaller coefficient for quadratic terms
        coeff_yy = (torch.rand(1, dtype=dtype) * 2 - 1) * max_second_order
        coeff_xx = (torch.rand(1, dtype=dtype) * 2 - 1) * max_second_order
        coeff_xy = (torch.rand(1, dtype=dtype) * 2 - 1) * max_second_order

        phase += (
            coeff_yy * Y_shifted**2
            + coeff_xx * X_shifted**2
            + coeff_xy * Y_shifted * X_shifted
        )

    return phase
