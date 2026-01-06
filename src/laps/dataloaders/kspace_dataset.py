"""
Generic Dataset which contains k-space data (for MoDL)
"""

import os
from typing import Any, Callable, List, Optional, Union

import loguru
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from laps.globals import QUALITY_MAX, QUALITY_MIN

from .labels import IMAGE_KEY, KSP_KEY, MASK_KEY, MPS_KEY, QUALITY_KEY

__all__ = ["KspaceDataset"]


class KspaceDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[
            os.PathLike, str, pd.DataFrame
        ],  # contains recon_path for imgs, and data_path for kspace
        data_dir: Union[os.PathLike, str],
        dataset_id: int,
        *,
        minimum_quality: Optional[int] = int(QUALITY_MIN),
        logger: Any = loguru.logger,
        verbose: bool = True,
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
            dataset_id (int, optional): Dataset ID. Defaults to None.
        """

        # Store arguments
        if isinstance(csv_path, pd.DataFrame):
            self.dataset = csv_path.copy()
            self.csv_path = None
        else:
            self.csv_path = csv_path
            self.dataset = pd.read_csv(csv_path)

        self.data_dir = data_dir
        self.dataset_id = dataset_id
        self.kwargs = kwargs
        self.logger = logger
        self.verbose = verbose

        if minimum_quality is not None and ("quality" in self.dataset.columns):
            self.dataset = self.dataset[self.dataset["quality"] >= minimum_quality]
        elif minimum_quality is not None:
            self.logger.warning(
                "Minimum quality specified, but no quality column found in the dataset."
            )
            self.dataset["quality"] = (
                QUALITY_MAX - QUALITY_MIN
            ) // 2  # Default to middle quality

        self.dataset.reset_index(drop=True, inplace=True)

        if self.verbose:
            self.print_stats()

    def __getitem__(self, idx: int):
        """Return a dictionary with the requested sample."""

        row = self.dataset.iloc[idx]

        sample = dict()
        sample["group_id"] = self.dataset_id

        data = torch.load(os.path.join(self.data_dir, row["data_path"]))
        sample[IMAGE_KEY] = torch.from_numpy(
            np.load(os.path.join(self.data_dir, row["recon_path"]))
        )
        sample[KSP_KEY] = data["ksp"]
        sample[MASK_KEY] = data["mask"]
        sample[MPS_KEY] = data["mps"]
        sample[QUALITY_KEY] = torch.tensor([row["quality"]]).float()

        return sample

    def __len__(self):
        return len(self.dataset)

    def print_stats(self):

        print_str = f"""=== Dataset stats for split ===
        CSV file: {self.csv_path}
        Data directory: {self.data_dir}
        Number of samples: {len(self.dataset)}
        """
        self.logger.info(print_str)

    def get_labels(self):
        raise NotImplementedError("KspaceDataset does not have label.")
