from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    Orientation,
    ScaleIntensity,
    CropForeground,
    ScaleIntensityRange,
    RandSpatialCrop,
    Lambda,
    Resize,
    ScaleIntensityRangePercentiles,
    CenterSpatialCrop,
    SpatialPad,
    RandFlip,
    RandLambda,
    RandRotate,
    Transpose,
)
import torch
import torch.nn.functional as F
from monai.transforms import Transform
import torchvision
from PIL import Image
import polars as pl
import numpy as np
from typing import Sequence


class MonaiNormalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return torchvision.transforms.Normalize(self.mean, self.std)(img)


class MonaiPad(Transform):
    def __init__(self, size, mode="constant", value=0):
        self.size = size
        self.mode = mode
        self.value = value

    def __call__(self, img):
        padding = []
        for i in range(len(img.shape) - 1, 0, -1):  # Start from the last dimension
            total_pad = max(self.size[i - 1] - img.shape[i], 0)
            padding.extend([total_pad // 2, total_pad - total_pad // 2])
        padded_img = F.pad(img, padding, mode=self.mode, value=self.value)
        return padded_img


class MonaiImageOpen(Transform):
    def __init__(self):
        pass

    def __call__(self, path):
        return Image.open(path)


"""
Custom transform to normalize and pad 2D images
@input: path to image (str)
@Output: padding (np.array)
"""


def augmenting_transforms(random_flip, rot_degree) -> Sequence[Transform]:
    augment_list = []
    if random_flip:
        augment_list.extend(
            [
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandLambda(Transpose((0, 2, 1)), prob=0.5),
            ],
        )
    if rot_degree > 0:
        augment_list.append(RandRotate(range_x=np.deg2rad(rot_degree)))
    return augment_list


def load_2d(
    path: str,
    merge_channels: bool = False,
    dtype: torch.dtype = torch.float32,
    **kwargs,
):
    img_transforms = Compose(
        transforms=[
            MonaiImageOpen(),
            torchvision.transforms.ToTensor(),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
        ],
        lazy=True,
    )

    try:
        img = img_transforms(path).as_tensor()

        if merge_channels:
            img = img.mean(0, keepdim=True)

        return img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 384, 384))


def load_2d_finetune(
    path: str, dtype: torch.dtype = torch.float32, merge_channels: bool = True, **kwargs
):
    img_transforms = Compose(
        transforms=[
            MonaiImageOpen(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((384, 384), interpolation=3, antialias=True),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
        ],
        lazy=True,
    )

    try:
        img = img_transforms(path).as_tensor()
        if merge_channels:
            img = img.mean(0, keepdim=True)
        return img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 384, 384))


def load_2d_four_channel(
    path: str, dtype: torch.dtype = torch.float32, merge_channels: bool = True, **kwargs
):
    img_transforms = Compose(
        transforms=[
            MonaiImageOpen(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((384, 384), interpolation=3, antialias=True),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
        ],
        lazy=True,
    )

    try:
        img = img_transforms(path).as_tensor()
        if merge_channels:
            img = img.mean(0, keepdim=True)
        return img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 384, 384))


def load_oasis(
    path: str,
    dtype: torch.dtype = torch.float32,
    return_2_ch: bool = True,
    random_flip=False,
    rot_degree=0,
    **kwargs,
):
    specific_crop = Lambda(func=lambda x: x[..., 15:180, 30:220])
    crop_transform = CenterSpatialCrop(roi_size=(192, 272))
    transforms_list = [
        LoadImage(),
        crop_transform,
        SpatialPad(spatial_size=(192, 272), mode="edge"),
        specific_crop,
        SpatialPad(spatial_size=(190, 190), mode="edge"),
        Resize(
            spatial_size=(256, 256), mode="bilinear"
        ),  # NOTE: this distorts the image unless its already square
        ScaleIntensity(channel_wise=True, minv=0, maxv=1),
        *augmenting_transforms(random_flip, rot_degree),
        # MonaiNormalize(mean=[0.5], std=[0.5]),
    ]
    img_transforms = Compose(
        transforms=transforms_list,
        lazy=True,
    )

    try:
        img = img_transforms(path).as_tensor()
        if return_2_ch:
            full_im = torch.zeros((2, 256, 256), dtype=img.dtype)
            full_im[0, :, :] = img[0, :, :]
            return full_im
        else:
            return img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((2, 256, 256))


def load_slam_dicoms(
    path: str,
    dtype: torch.dtype = torch.float32,
    return_2_ch: bool = True,
    random_flip=False,
    rot_degree=0,
    **kwargs,
):
    transforms_list = [
        LoadImage(),
        EnsureChannelFirst(channel_dim="no_channel"),
        Resize(spatial_size=(256, 256), mode="bilinear"),
        ScaleIntensity(channel_wise=True, minv=0, maxv=1),
        *augmenting_transforms(random_flip, rot_degree),
        # MonaiNormalize(mean=[0.5], std=[0.5]),
    ]
    img_transforms = Compose(
        transforms=transforms_list,
        lazy=True,
    )

    try:
        img = img_transforms(path).as_tensor()
        if return_2_ch:
            full_im = torch.zeros((2, 256, 256), dtype=img.dtype)
            full_im[0, :, :] = img[0, :, :]
            return full_im
        return img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((2, 256, 256))


class LoadComplexImage(Transform):
    """
    Custom transform to load an image while preserving its complex data.

    Specify a % chance of returning just the magnitude too
    """

    def __init__(
        self,
        complex_dropout_frac: float = 0.0,
        complex_global_phase_modulation: bool = False,
    ):
        super().__init__()
        self.complex_dropout_frac = complex_dropout_frac
        self.complex_global_phase_modulation = complex_global_phase_modulation

    def __call__(self, path: str) -> torch.Tensor:
        data = np.load(path)
        if not np.iscomplexobj(data):
            raise ValueError(f"Data in {path} is not complex.")
        data = data / np.max(np.abs(data))

        # Apply global phase modulation if desired
        # This is a random phase shift applied to the whole image for data augmentation
        if self.complex_global_phase_modulation:
            phase = np.random.rand() * 2 * np.pi
            data = data * np.exp(1j * phase)

        is_complex = np.random.rand() > self.complex_dropout_frac
        if is_complex:
            data = np.concatenate(
                (data.real[np.newaxis], data.imag[np.newaxis]), axis=0
            )
        else:
            zeros = np.zeros_like(data.real)
            data = np.concatenate((np.abs(data)[np.newaxis], zeros[np.newaxis]), axis=0)

        data = torch.from_numpy(data)
        return data


def load_bruno(
    path: str,
    dtype: torch.dtype = torch.float32,
    random_flip=False,
    rot_degree=0,
    rand_phase=False,
    **kwargs,
):
    transforms_list = [
        LoadComplexImage(
            complex_dropout_frac=0, complex_global_phase_modulation=rand_phase
        ),
        Resize(
            spatial_size=(256, 256), mode="bilinear"
        ),  # may distort image if not square
        *augmenting_transforms(random_flip, rot_degree),
    ]

    img_transforms = Compose(
        transforms=transforms_list,
        lazy=True,
    )

    try:
        img = img_transforms(path).as_tensor().type(dtype)

        return img
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((2, 256, 256), dtype=dtype)


"""
Custom transform to normalize, crop, and pad 3D volumes
@input: path to image (str)
@Output: padding (np.array)
"""


def load_mri_3d(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    mri_transforms = Compose(
        transforms=[
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
            CropForeground(k_divisible=[16, 16, 16]),
        ],
        lazy=True,
    )

    try:
        mr_augmented = mri_transforms(path).as_tensor()
        return mr_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 128, 128, 128))


"""
Custom transform to normalize, crop, and pad 3D CT volumes
@input: path to image (str)
@Output: padding (np.array)
"""


def load_ct_3d(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    ct_transforms = Compose(
        transforms=[
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            MonaiNormalize(mean=[0.5], std=[0.5]),
            CropForeground(k_divisible=[16, 16, 16]),
        ],
        lazy=True,
    )

    try:
        ct_augmented = ct_transforms(path).as_tensor()
        return ct_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 256, 256, 256))


"""
Custom transform to normalize, crop, and pad 3D volumes
@input: path to image (str)
@Output: padding (np.array)
"""


def load_mri_3d_finetune(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    mri_transforms = Compose(
        transforms=[
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensity(channel_wise=True, minv=0, maxv=1),
            MonaiNormalize(mean=[0.5], std=[0.5]),
            MonaiPad(size=[64, 64, 64], value=-1),
            RandSpatialCrop(roi_size=[64, 64, 64]),
        ],
        lazy=True,
    )

    try:
        mr_augmented = mri_transforms(path).as_tensor()
        return mr_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 64, 64, 64))


"""
Custom transform to normalize, crop, and pad ct 3D volumes
@input: path to image (str)
@Output: padding (np.array)
"""


def load_ct_3d_finetune(path: str, dtype: torch.dtype = torch.float32, **kwargs):
    ct_transforms = Compose(
        transforms=[
            LoadImage(),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            MonaiNormalize(mean=[0.5], std=[0.5]),
            MonaiPad(size=[64, 64, 64], value=-1),
            RandSpatialCrop(roi_size=[64, 64, 64]),
        ],
        lazy=True,
    )

    try:
        ct_augmented = ct_transforms(path).as_tensor()
        return ct_augmented
    except Exception as e:
        print(f"Error in loading {path} with error: {e}")
        return torch.zeros((1, 64, 64, 64))


def load_labels(
    df: pl.DataFrame,
    dtype: np.dtype = None,
    # fill_null=None,
    fill_nan: float = None,
    squeeze: int = None,
) -> torch.Tensor:
    """Load the labels from a dataframe."""
    # BUG: Polars hangs when trying to convert to numpy in a DataLoader
    x = df.to_pandas().to_numpy()
    if dtype is not None:
        x = x.astype(dtype)

    if isinstance(squeeze, int):
        out = torch.from_numpy(x).squeeze(dim=squeeze)
    else:
        out = torch.from_numpy(x).squeeze()

    if isinstance(fill_nan, float):
        out = torch.where(out.isnan(), fill_nan, out)

    return out
