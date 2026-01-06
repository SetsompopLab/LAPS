"""
List of datasets available in LAPS.

See `README.md` for how to pull the relevant SLAM data as desired.
"""

import os
from typing import Optional, Sequence, Union, Callable

import loguru
import torch
from torchvision.transforms import Compose

from laps import DATA_DIR
from laps.dataloaders import GenericDataset, LoaderType, get_loader
from laps.dataloaders.paired_dataloader import (
    AccelerationType,
    Dimension,
    PairedDataset,
    ScanPlane,
    ScanType,
)

# great global dataset counter
DATASET_ID_COUNTER = 0


class DatasetWrapper:
    def __init__(
        self,
        split_path: Union[str, os.PathLike],
        data_dir: Union[str, os.PathLike],
        loader_type: LoaderType,
        paired: bool = False,
    ):
        self.split_path = split_path
        self.data_dir = data_dir
        self.loader_type = loader_type
        self.paired = paired

    def get_dataset(
        self,
        split: str,
        image_size: Sequence[int] = (256, 256),
        dtype: torch.dtype = torch.float32,
        num_channels: int = 2,
        randcrop: bool = False,
        random_flip: bool = False,
        rot_degree: float = 0,
        complex_dropout_frac: float = 0.0,
        complex_global_phase_modulation: bool = False,
        complex_output: bool = True,
        logger=loguru.logger,
    ):
        """
        Returns the dataset path for the given split.
        """
        global DATASET_ID_COUNTER
        DATASET_ID_COUNTER += 1

        if randcrop == True:
            logger.warning(
                "randcrop is deprecated and will be removed in a future release. "
                "Use random_flip and rot_degree instead."
            )

        assert split in [
            "train",
            "test",
            "val",
        ], "split must be one of ['train', 'test', 'val']"

        loader_transform = get_loader(
            loader_type=self.loader_type,
            image_size=image_size,
            dtype=dtype,
            num_channels=num_channels,
            random_flip=random_flip,
            rot_degree=rot_degree,
            complex_dropout_frac=complex_dropout_frac,
            complex_global_phase_modulation=complex_global_phase_modulation,
            complex_output=complex_output,
            logger=logger,
        )
        return GenericDataset(
            split_path=self.split_path,
            data_dir=self.data_dir,
            dataset_id=(DATASET_ID_COUNTER + 0),
            split_name=split,
            split_column="split",
            quality_column="quality",
            img_column="image_uuid",
            prior_column="image_uuid_prior" if self.paired else None,
            img_suffix=".npy",
            img_transform=Compose(transforms=[loader_transform]),
            logger=logger,
        )

    def __repr__(self):
        return f"DatasetWrapper(split_path={self.split_path}, data_dir={self.data_dir})"


class PairedDatasetWrapper:
    def __init__(
        self,
        csv_path: Union[str, os.PathLike],
        data_dir: Union[str, os.PathLike],
    ):
        """
        Paired dataset wrapper for loading paired datasets.
        """
        self.csv_path = csv_path
        self.data_dir = data_dir

    def get_dataset(
        self,
        acceleration_filter: AccelerationType = AccelerationType.ALL,
        scan_plane_filter: ScanPlane = ScanPlane.ALL,
        scan_type_filter: ScanType = ScanType.ALL,
        minimum_quality: Optional[int] = None,
        load_dimension: Dimension = Dimension.DIM_2D,
        return_metadata: bool = False,
        verbose: bool = True,
        shuffle: bool = False,
        max_samples: Optional[int] = None,
        mod_jump: int = 1,
        n_slices: Optional[int] = None,
        custom_filt: Optional[Callable] = None,
    ):

        return PairedDataset(
            csv_path=self.csv_path,
            data_dir=self.data_dir,
            registered=True,
            acceleration_filter=acceleration_filter,
            scan_plane_filter=scan_plane_filter,
            scan_type_filter=scan_type_filter,
            minimum_quality=minimum_quality,
            load_dimension=load_dimension,
            return_metadata=return_metadata,
            verbose=verbose,
            shuffle=shuffle,
            max_samples=max_samples,
            mod_jump=mod_jump,
            n_slices=n_slices,
            custom_filt=custom_filt,
        )

# Image datasets for training 
LAPS_DATASETS = {
    LoaderType.SLAM: DatasetWrapper(
        split_path=DATA_DIR / "slam-train.csv",
        data_dir=DATA_DIR / "slam-train",
        loader_type=LoaderType.SLAM,
    ),
    LoaderType.SLAM_DICOM: DatasetWrapper(
        split_path=DATA_DIR / "slam-train-dicom.csv",
        data_dir=DATA_DIR / "slam-train-dicom",
        loader_type=LoaderType.SLAM_DICOM,
    ),
}

# Paired datasets for inference
LAPS_PAIRED_DATASETS = {
    "slam-test": PairedDatasetWrapper(
        csv_path=DATA_DIR / "slam-test.csv",
        data_dir=DATA_DIR / "slam-test",
    ),
}