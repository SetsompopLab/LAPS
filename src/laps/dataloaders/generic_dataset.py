import os
from typing import Any, Callable, List, Optional, Union

import loguru
import polars as pl
from torch.utils.data import Dataset

from laps.globals import QUALITY_MAX

from .labels import IMAGE_KEY, LABEL_KEY, MASK_KEY, PRIOR_KEY, TEXT_KEY

__all__ = ["GenericDataset"]


class GenericDataset(Dataset):
    def __init__(
        self,
        split_path: Union[os.PathLike, str],
        data_dir: Union[os.PathLike, str],  # DEPRECATED
        dataset_id: int,
        *,
        split_column: Optional[str] = None,
        split_name: Optional[str] = None,
        img_column: Optional[str] = None,
        img_dir: Optional[str] = None,
        img_suffix: Optional[str] = None,
        img_transform: Optional[Callable] = None,
        prior_column: Optional[str] = None,
        lbl_columns: Optional[List[str]] = None,
        lbl_transform: Optional[Callable] = None,
        msk_column: Optional[str] = None,
        msk_dir: Optional[str] = None,
        msk_suffix: Optional[str] = None,
        msk_transform: Optional[Callable] = None,
        txt_column: Optional[str] = None,
        txt_dir: Optional[str] = None,
        txt_suffix: Optional[str] = None,
        txt_transform: Optional[Callable] = None,
        com_transform: Optional[Callable] = None,
        quality_column: Optional[str] = None,
        logger: Any = loguru.logger,
        **kwargs,
    ):
        """A Generic Dataset implementation.

        The generic dataset can be used to create any dataset from a given CSV "split" and a data
        directory. The dataset aims to be as flexible as possible, allowing the user to specify
        the columns to use for images, labels, and text. The user can also specify the transforms
        to apply to each of these components.

        Args:
            split_path (Union[os.PathLike, str]): Path to the split file.
            data_dir (Union[os.PathLike, str]): Path to the data directory.
            img_column (str, optional): Image column. Defaults to "image_uuid".
            img_suffix (str, optional): Image suffix. Defaults to ".npy".
            img_transform (Callable, optional): Image transform. Defaults to None.
            lbl_columns (List[str], optional): Label columns. Defaults to None.
            lbl_transform (Callable, optional): Label transform. Defaults to None.
            txt_column (str, optional): Text column. Defaults to None.
            txt_transform (Callable, optional): Text transform. Defaults to None.
            com_transform (Callable, optional): Composite transform. Defaults to None.
            dataset_id (int, optional): Dataset ID. Defaults to None.
        """

        # Store arguments
        self.split_path = split_path
        self.split_column = split_column
        self.split_name = split_name
        self.dataset_id = dataset_id
        self.img_dir = img_dir if img_dir is not None else data_dir
        self.img_column = img_column
        self.img_suffix = img_suffix
        self.img_transform = img_transform
        self.lbl_columns = lbl_columns
        self.lbl_transform = lbl_transform
        self.msk_column = msk_column
        self.msk_dir = msk_dir
        self.msk_suffix = msk_suffix
        self.msk_transform = msk_transform
        self.txt_column = txt_column
        self.txt_dir = txt_dir
        self.txt_suffix = txt_suffix
        self.txt_transform = txt_transform
        self.com_transform = com_transform
        self.quality_column = quality_column
        self.kwargs = kwargs
        self.logger = logger

        self.samples = {}

        # Create the samples for the images, labels, and text
        if not os.path.exists(split_path):
            raise ValueError(f"Split path {split_path} does not exist.")

        self.df = pl.read_csv(os.path.abspath(split_path))
        if isinstance(split_column, str) and isinstance(split_name, str):
            self.df = self.df.filter(pl.col(split_column) == split_name)

        # Generate image paths
        if img_column is not None:
            self.samples[IMAGE_KEY] = []
            for x in self.df.get_column(img_column).to_list():
                file_name = x
                if img_suffix and not x.endswith(img_suffix):
                    file_name = f"{x}{img_suffix}"
                self.samples[IMAGE_KEY].append(os.path.join(self.img_dir, file_name))

        if quality_column is not None and quality_column in self.df.columns:
            self.samples["quality"] = self.df.get_column(quality_column).to_list()
        else:
            # If no quality column, set all to the second best quality
            self.samples["quality"] = [QUALITY_MAX - 1] * len(self.df)

        if prior_column is not None:
            self.samples["prior"] = []
            for x in self.df.get_column(prior_column).to_list():
                file_name = x
                if img_suffix and not x.endswith(img_suffix):
                    file_name = f"{x}{img_suffix}"
                self.samples["prior"].append(os.path.join(self.img_dir, file_name))

        # Extract the columns with labels
        if lbl_columns is not None:
            self.samples[LABEL_KEY] = self.df.select(lbl_columns)

        # Extract the column with text or a path to a text file
        if txt_column is not None:
            self.samples[TEXT_KEY] = self.df.get_column(txt_column).to_list()

        # Extract the column with masks
        if msk_column is not None:
            self.samples[MASK_KEY] = []
            for x in self.df.get_column(msk_column).to_list():
                file_name = x
                if msk_suffix and not x.endswith(msk_suffix):
                    file_name = f"{x}{msk_suffix}"
                if self.msk_dir is None:
                    raise ValueError(
                        "Mask directory must be specified if mask column is provided."
                    )
                path = os.path.join(self.msk_dir, file_name)
                self.samples[MASK_KEY].append(path)

        self.print_stats()

    def __getitem__(self, idx: int):
        """Return a dictionary with the requested sample."""
        sample = {"group_id": self.dataset_id}

        # Image
        if IMAGE_KEY in self.samples:
            sample[IMAGE_KEY] = self.samples[IMAGE_KEY][idx]
            if callable(self.img_transform):
                sample[IMAGE_KEY] = self.img_transform(sample[IMAGE_KEY])

        # Quality
        if "quality" in self.samples:
            sample["quality"] = self.samples["quality"][idx]

        # Prior Image
        if PRIOR_KEY in self.samples:
            sample[PRIOR_KEY] = self.samples[PRIOR_KEY][idx]
            if callable(self.img_transform):
                sample[PRIOR_KEY] = self.img_transform(sample[PRIOR_KEY])

        # Labels
        if LABEL_KEY in self.samples:
            sample[LABEL_KEY] = self.samples[LABEL_KEY][idx]
            if callable(self.lbl_transform):
                sample[LABEL_KEY] = self.lbl_transform(sample[LABEL_KEY])

        # Mask
        if MASK_KEY in self.samples:
            sample[MASK_KEY] = self.samples[MASK_KEY][idx]
            if callable(self.msk_transform):
                sample[MASK_KEY] = self.msk_transform(sample[MASK_KEY])

        # Text
        if TEXT_KEY in self.samples:
            sample[TEXT_KEY] = self.samples[TEXT_KEY][idx]
            if callable(self.txt_transform):
                sample[TEXT_KEY] = self.txt_transform(sample[TEXT_KEY])

        # Common transform applied on sample level
        if callable(self.com_transform):
            return self.com_transform(sample)

        return sample

    def __len__(self):
        return len(self.df)

    def print_stats(self):

        print_str = f"""=== Dataset stats for split={self.split_name or "full"} ===
        CSV file: {self.split_path}
        Data directory: {self.img_dir}
        Number of samples: {len(self.df)}
        """
        self.logger.info(print_str)

    def get_labels(self):
        return self.samples[LABEL_KEY].to_numpy().flatten()
