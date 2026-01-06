from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

import loguru
import numpy as np
import torch
import torchvision.transforms as T
from torch import nn

from .dataloader_util import (
    LoadComplexImage,
    convert_real_to_complex,
    get_augmentation_transforms,
)

__all__ = [
    "LoaderType",
    "get_loader",
    "BaseLoader",
    "SlamDicomLoader",
    "SlamComplexLoader",
    "slam_dicom_load",
    "slam_complex_load",
]


class LoaderType(Enum):
    """Constants for different loader types."""
    SLAM_DICOM = "slam_dicom"
    SLAM = "slam"

class LoadNumpyImage(nn.Module):
    """Load numpy images (.npy files) and convert to tensor."""

    def __init__(self, complexify: bool = True):
        super().__init__()
        self.complexify = complexify

    def forward(self, path: str) -> torch.Tensor:
        """Load numpy image and convert to tensor."""
        try:
            data = np.load(path)
            tensor = torch.from_numpy(data).float()

            # Ensure channel dimension
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0)  # Add channel dimension (C, H, W)
            elif len(tensor.shape) == 3 and tensor.shape[0] not in [1, 2, 3, 4]:
                # If first dimension is not a typical channel count, move it to the end
                tensor = tensor.permute(2, 0, 1)

            if self.complexify and not torch.is_complex(tensor):
                norm = tensor.max() - tensor.min()
                tensor = tensor - tensor.min()
                if norm > 0:
                    tensor = tensor / norm

                tensor = convert_real_to_complex(tensor)

            return tensor
        except Exception as e:
            raise RuntimeError(f"Failed to load numpy image {path}: {e}")


class EnsureChannelFirst(nn.Module):
    """Ensure tensor has channel dimension first."""

    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has shape (C, H, W)."""
        if len(tensor.shape) == 2:
            return tensor.unsqueeze(0)
        elif len(tensor.shape) == 3:
            return tensor
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tensor.shape}")


class ConditionalCrop(nn.Module):
    """Apply center crop or random crop based on flag."""

    def __init__(self, size: Sequence[int], center_crop: bool = True):
        super().__init__()
        self.size = size
        if center_crop:
            self.crop = T.CenterCrop(size)
        else:
            self.crop = T.RandomCrop(size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.crop(tensor)


class PadToSize(nn.Module):
    """Pad tensor to specified size using torchvision Pad."""

    def __init__(
        self, spatial_size: Sequence[int], mode: str = "reflect", value: float = 0
    ):
        super().__init__()
        self.spatial_size = tuple(spatial_size)
        self.padding_mode = mode
        self.fill = value

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad tensor to spatial size using torchvision Pad."""
        if len(tensor.shape) != 3:
            raise ValueError(f"Expected 3D tensor (C, H, W), got shape {tensor.shape}")

        _, h, w = tensor.shape
        target_h, target_w = self.spatial_size

        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        if pad_h == 0 and pad_w == 0:
            return tensor

        # Calculate padding for each side
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        pad_transform = T.Pad(
            padding, fill=int(self.fill), padding_mode=self.padding_mode
        )

        return pad_transform(tensor)


class BaseLoader(ABC):
    """
    Abstract base class for all data loaders.

    Provides common functionality and enforces consistent interface
    across different loader implementations.
    """

    def __init__(
        self,
        image_size: Sequence[int] = (256, 256),
        dtype: torch.dtype = torch.float32,
        num_channels: int = 2,
        complex_output: bool = True,
    ):
        """
        Initialize base loader.

        Args:
            image_size: Target image size (H, W)
            dtype: Output tensor dtype
            num_channels: Number of output channels
            complex_output: Whether to output complex data representation
        """
        self.image_size = tuple(image_size)
        self.dtype = dtype
        self.num_channels = num_channels
        self.complex_output = complex_output
        self.output_shape = (num_channels, *self.image_size)
        self.logger = loguru.logger

        # Initialize transforms
        self._setup_transforms()

    def set_logger(self, logger):
        self.logger = logger

    def _scale_intensity(self, img: torch.Tensor) -> torch.Tensor:
        """Scale intensity to [0, 1] range."""
        for c in range(img.shape[0]):
            channel = img[c]
            min_val, max_val = channel.min(), channel.max()
            if max_val > min_val:
                img[c] = (channel - min_val) / (max_val - min_val)
        return img

    @abstractmethod
    def _setup_transforms(self):
        """Setup the image transformation pipeline."""
        pass

    @abstractmethod
    def _load_and_process(self, path: str) -> torch.Tensor:
        """Load and process image from path."""
        pass

    def load(self, path: str) -> torch.Tensor:
        """
        Safe load wrapper with error handling.

        Args:
            path: Path to image file

        Returns:
            Processed tensor of shape (num_channels, H, W)
        """
        try:
            return self._load_and_process(path)
        except Exception as e:
            self.logger.error(f"Error loading {path}: {e}")
            return torch.zeros(self.output_shape, dtype=self.dtype)

    def _prepare_output_channels(self, img: torch.Tensor) -> torch.Tensor:
        """
        Prepare output tensor with correct number of channels.

        Args:
            img: Input tensor, typically 2-channel complex representation

        Returns:
            Tensor with self.num_channels channels
        """
        if self.complex_output and self.num_channels == 2:
            # Standard complex representation: [real, imag]
            return img[:2]  # Ensure only 2 channels
        elif self.complex_output and self.num_channels == 1:
            # Magnitude only
            if img.shape[0] >= 2:
                magnitude = torch.abs(img[0] + 1j * img[1])
                return magnitude[None, ...]
            else:
                return img[:1]
        elif self.complex_output and self.num_channels > 2:
            # Mixed: real, imag, magnitude, ...
            full_tensor = torch.zeros(
                self.num_channels, *self.image_size, dtype=self.dtype
            )
            if img.shape[0] >= 2:
                full_tensor[:2] = img[:2]  # Real and imaginary
                if self.num_channels > 2:
                    magnitude = torch.abs(img[0] + 1j * img[1])
                    full_tensor[2:] = magnitude[None, ...].repeat(
                        self.num_channels - 2, 1, 1
                    )
            return full_tensor
        else:
            # Magnitude-only output, rescale to [-1, 1]
            if img.shape[0] >= 2:
                magnitude = torch.abs(img[0] + 1j * img[1])
            else:
                magnitude = img[0]

            magnitude_normalized = magnitude * 2 - 1  # [0, 1] -> [-1, 1]
            return magnitude_normalized[None, ...].repeat(self.num_channels, 1, 1)


class SlamDicomLoader(BaseLoader):
    """
    Loader for Slam DICOM dataset (stored as .npy files).

    Handles medical images with standardized preprocessing.
    """

    def __init__(
        self,
        image_size: Sequence[int] = (256, 256),
        dtype: torch.dtype = torch.float32,
        num_channels: int = 2,
        random_flip: bool = False,
        rot_degree: float = 0,
        complex_output: bool = True,
        **kwargs,
    ):
        self.random_flip = random_flip
        self.rot_degree = rot_degree

        super().__init__(image_size, dtype, num_channels, complex_output)

    def _setup_transforms(self):
        """Setup transforms for numpy-stored medical images."""
        transforms = [
            LoadNumpyImage(complexify=self.complex_output),
            EnsureChannelFirst(),
            T.Resize(
                size=self.image_size,
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
        ]

        # Add augmentations
        if self.random_flip or self.rot_degree > 0:
            augment_transforms = get_augmentation_transforms(
                random_flip=self.random_flip, rotation_degrees=self.rot_degree
            )
            transforms.extend(augment_transforms)

        self.img_transforms = T.Compose(transforms)

    def _load_and_process(self, path: str) -> torch.Tensor:
        """Load and process medical image."""
        img = self.img_transforms(path)
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)
        img = img.to(dtype=self.dtype)

        return self._prepare_output_channels(img)


class SlamComplexLoader(BaseLoader):
    """
    Loader for Slam complex-valued dataset.

    Handles complex MRI data with phase information preservation.
    """

    def __init__(
        self,
        image_size: Sequence[int] = (256, 256),
        dtype: torch.dtype = torch.float32,
        num_channels: int = 2,
        random_flip: bool = False,
        rot_degree: float = 0,
        complex_dropout_frac: float = 0.0,
        complex_global_phase_modulation: bool = False,
        complex_output: bool = True,
        **kwargs,
    ):
        self.random_flip = random_flip
        self.rot_degree = rot_degree
        self.complex_dropout_frac = complex_dropout_frac
        self.complex_global_phase_modulation = complex_global_phase_modulation

        super().__init__(image_size, dtype, num_channels, complex_output)

    def _setup_transforms(self):
        """Setup complex image transforms."""
        transforms = [
            LoadComplexImage(
                complex_dropout_frac=self.complex_dropout_frac,
                complex_global_phase_modulation=self.complex_global_phase_modulation,
            ),
            T.Resize(
                size=self.image_size,
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
        ]

        # Add augmentations
        if self.random_flip or self.rot_degree > 0:
            augment_transforms = get_augmentation_transforms(
                random_flip=self.random_flip, rotation_degrees=self.rot_degree
            )
            transforms.extend(augment_transforms)

        self.img_transforms = T.Compose(transforms)

    def _load_and_process(self, path: str) -> torch.Tensor:
        """Load and process complex image."""
        img = self.img_transforms(path)
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)
        img = img.to(dtype=self.dtype)
        return self._prepare_output_channels(img)


def get_loader(
    loader_type: LoaderType,
    image_size: Sequence[int] = (256, 256),
    dtype: torch.dtype = torch.float32,
    num_channels: int = 2,
    random_flip: bool = False,
    rot_degree: float = 0,
    complex_dropout_frac: float = 0.0,
    complex_global_phase_modulation: bool = True,
    complex_output: bool = True,
    logger=None,
):
    """
    Factory function to create appropriate loader based on type.

    Args:
        loader_type: Type of loader to create
        image_size: Target image size (H, W)
        dtype: Output tensor dtype
        num_channels: Number of output channels
        random_flip: Whether to apply random flips
        rot_degree: Maximum rotation angle in degrees (applied probabilistically)
        complex_dropout_frac: Fraction of time to use magnitude only [0, 1]
        complex_global_phase_modulation: Whether to apply random global phase
        complex_output: Whether to output complex representation
        logger: Optional logger instance

    Returns:
        Configured loader's load method

    Raises:
        ValueError: If loader_type is not recognized
    """
    loader_classes = {
        LoaderType.SLAM_DICOM: SlamDicomLoader,
        LoaderType.SLAM: SlamComplexLoader,
    }

    if loader_type not in loader_classes:
        available_types = list(loader_classes.keys())
        raise ValueError(
            f"Unknown loader type '{loader_type}'. Available: {available_types}"
        )

    loader_cls = loader_classes[loader_type]

    # Create loader with appropriate parameters
    loader_kwargs = {
        "image_size": image_size,
        "dtype": dtype,
        "num_channels": num_channels,
        "random_flip": random_flip,
        "rot_degree": rot_degree,
        "complex_output": complex_output,
    }

    # Add loader-specific parameters
    if loader_type in [
        LoaderType.SLAM,
    ]:
        loader_kwargs.update(
            {
                "complex_dropout_frac": complex_dropout_frac,
                "complex_global_phase_modulation": complex_global_phase_modulation,
            }
        )

    loader = loader_cls(**loader_kwargs)

    if logger is not None:
        loader.set_logger(logger)

    return loader.load


def slam_dicom_load(
    path: str,
    image_size: Sequence[int] = (256, 256),
    dtype: torch.dtype = torch.float32,
    num_channels: int = 2,
    random_flip: bool = False,
    rot_degree: float = 0,
    complex_dropout_frac: float = 0.0,
    complex_output: bool = True,
):
    return SlamDicomLoader(
        image_size=image_size,
        dtype=dtype,
        num_channels=num_channels,
        random_flip=random_flip,
        rot_degree=rot_degree,
        complex_dropout_frac=complex_dropout_frac,
        complex_output=complex_output,
    ).load(path)


def slam_complex_load(
    path: str,
    image_size: Sequence[int] = (256, 256),
    dtype: torch.dtype = torch.float32,
    num_channels: int = 2,
    random_flip: bool = False,
    rot_degree: float = 0,
    complex_dropout_frac: float = 0.0,
    complex_output: bool = True,
    complex_global_phase_modulation: bool = True,
):
    return SlamComplexLoader(
        image_size=image_size,
        dtype=dtype,
        num_channels=num_channels,
        random_flip=random_flip,
        rot_degree=rot_degree,
        complex_dropout_frac=complex_dropout_frac,
        complex_global_phase_modulation=complex_global_phase_modulation,
        complex_output=complex_output,
    ).load(path)
