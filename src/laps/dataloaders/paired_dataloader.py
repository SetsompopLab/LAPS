import os
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from .labels import IMAGE_KEY, KSP_KEY, MASK_KEY, MPS_KEY, PRIOR_KEY

class AccelerationType(Enum):
    FS = 0
    R_1D = 1
    R_2D = 2
    ALL = "all"
    R_1D_FS = 3  # 1D and FS
    R_2D_FS = 4  # 2D and FS


class ScanPlane(Enum):
    AXIAL = "ax"
    CORONAL = "cor"
    SAGITTAL = "sag"
    REG = "reg" # for misreg experiment
    ALL = "all"


class ScanType(Enum):
    T1 = "T1"
    T2 = "T2"
    T2_FLAIR = "T2_FLAIR"
    ALL = "all"


class Dimension(Enum):
    DIM_2D = 2
    DIM_3D = 3
    MIDDLE_SLICE = "middle_slice"


class PairedDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[str, os.PathLike],
        data_dir: Union[str, os.PathLike],
        registered: bool = True,
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
        """
        Notes:
        - load_dimension: affects what kind of data is loaded.
            - 2D: Loads individual 2D slices.
            - 3D: Loads full 3D volumes at a time.
            - middle_slice: Loads only the middle slice of 3D volumes (still 2D data, just a reduced set)
        """
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.registered = registered
        self.return_metadata = return_metadata

        self.dataset = pd.read_csv(self.csv_path)

        # only keep with prior
        self.dataset = self.dataset[
            self.dataset["prior_path"].notna()
        ]

        if custom_filt is not None:
            self.dataset = custom_filt(self.dataset)

        self.acceleration_filter = acceleration_filter
        self.scan_plane_filter = scan_plane_filter
        self.scan_type_filter = scan_type_filter
        self.minimum_quality = minimum_quality
        self.load_dimension = load_dimension

        # Filter dataset based on provided criteria
        if acceleration_filter != AccelerationType.ALL:
            if acceleration_filter == AccelerationType.FS:
                self.dataset = self.dataset[self.dataset["AccelNumDim"] == 0]
            elif acceleration_filter == AccelerationType.R_1D:
                self.dataset = self.dataset[self.dataset["AccelNumDim"] == 1]
            elif acceleration_filter == AccelerationType.R_2D:
                self.dataset = self.dataset[self.dataset["AccelNumDim"] == 2]
            elif acceleration_filter == AccelerationType.R_1D_FS:
                self.dataset = self.dataset[
                    (self.dataset["AccelNumDim"] == 1)
                    | (self.dataset["AccelNumDim"] == 0)
                ]
            elif acceleration_filter == AccelerationType.R_2D_FS:
                self.dataset = self.dataset[
                    (self.dataset["AccelNumDim"] == 2)
                    | (self.dataset["AccelNumDim"] == 0)
                ]

        if scan_plane_filter != ScanPlane.ALL:
            if scan_plane_filter == ScanPlane.AXIAL:
                self.dataset = self.dataset[self.dataset["scan_plane"] == "ax"]
            elif scan_plane_filter == ScanPlane.CORONAL:
                self.dataset = self.dataset[self.dataset["scan_plane"] == "cor"]
            elif scan_plane_filter == ScanPlane.SAGITTAL:
                self.dataset = self.dataset[self.dataset["scan_plane"] == "sag"]
            elif scan_plane_filter == ScanPlane.REG:
                self.dataset = self.dataset[self.dataset["scan_plane"] == "reg"]

        if scan_type_filter != ScanType.ALL:
            if scan_type_filter == ScanType.T1:
                self.dataset = self.dataset[
                    self.dataset["scan_type"].str.contains("T1", case=False)
                ]
            elif scan_type_filter == ScanType.T2_FLAIR:
                self.dataset = self.dataset[
                    self.dataset["scan_type"].str.contains("T2_FLAIR", case=False)
                ]
            elif scan_type_filter == ScanType.T2:
                self.dataset = self.dataset[
                    self.dataset["scan_type"].str.contains("T2", case=False)
                ]
                self.dataset = self.dataset[
                    ~self.dataset["scan_type"].str.contains("FLAIR", case=False)
                ]

        if minimum_quality is not None:
            self.dataset = self.dataset[self.dataset["quality"] >= minimum_quality]

        if load_dimension == Dimension.MIDDLE_SLICE:
            # Filter to only include middle slices
            self.dataset = self.dataset[self.dataset["is_middle_slice"]]
            self.load_dimension = Dimension.DIM_2D
        else:
            self.load_dimension = load_dimension

        # create index for subj/scan
        self.dataset["vol_idx"] = (
            self.dataset["subj_index"].astype(str)
            + "_"
            + self.dataset["scan_index"].astype(str)
        )
        self.dataset["vol_idx"] = pd.Categorical(
            self.dataset["vol_idx"], categories=self.dataset["vol_idx"].unique()
        )
        self.dataset["vol_idx"] = self.dataset["vol_idx"].cat.codes

        # reset index of dataset
        self.dataset.reset_index(drop=True, inplace=True)

        # Subsample per-volume slices
        if n_slices is not None:
            keep_indices = []
            for vol, grp in self.dataset.groupby("vol_idx"):  # for each scan
                idxs = grp.index.values
                count = len(idxs)
                num = min(n_slices, count)
                # choose evenly spaced positions
                positions = np.linspace(0, count - 1, num=num, dtype=int)
                keep_indices.extend(idxs[positions])
            # Subset and reset index
            self.dataset = (
                self.dataset.loc[keep_indices]
                .sort_values(["vol_idx", "slice_index"])
                .reset_index(drop=True)
            )
        elif max_samples is not None and max_samples < len(self.dataset):
            max_samples = max_samples * mod_jump
            # Map each volume to its middle slice index
            mid_map = (
                self.dataset[
                    self.dataset["is_middle_slice"]
                ]  # take only middle slices rows
                .set_index("vol_idx")["slice_index"]  # map from vol_idx to slice_index
                .to_dict()
            )
            # Compute distance score to the middle slice
            self.dataset["slice_score"] = self.dataset.apply(
                lambda row: abs(row["slice_index"] - mid_map.get(row["vol_idx"], 0)),
                axis=1,
            )
            # Select top-scoring slices (closest to middle)
            self.dataset = (
                self.dataset.sort_values("slice_score")
                .head(max_samples)
                .drop(columns=["slice_score"])
                .reset_index(drop=True)
            )

            self.dataset = (
                self.dataset.sort_values(["vol_idx", "slice_index"])
                .iloc[::mod_jump]
                .reset_index(drop=True)
            )
        elif mod_jump > 1:
            self.dataset = (
                self.dataset.sort_values(["vol_idx", "slice_index"])
                .iloc[::mod_jump]
                .reset_index(drop=True)
            )

        # create internal index
        if shuffle:
            self.INDEX = np.random.permutation(len(self))
        else:
            self.INDEX = np.arange(len(self))

        self.prior_key = "prior_path_reg" if self.registered else "prior_path"

        if len(self.dataset) == 0:
            raise ValueError(
                "No data found after applying filters. Please adjust your filters."
            )

        if verbose:
            self.print_info()

    def print_info(self):
        """
        Print breakdown of dataset by filters applied.
        """
        printfunc = logger.info

        printfunc(f"Dataset loaded from: {self.csv_path}")
        printfunc(f"Data directory: {self.data_dir}")
        printfunc(f"Using Registered = {self.registered}")
        printfunc(f"Number of samples: {len(self.dataset)}")

        accel_counts = self.dataset["AccelNumDim"].value_counts()
        printfunc("Acceleration counts:")
        for key, value in accel_counts.items():
            if key == 0:
                printfunc(f"  FS: {value}")
            elif key == 1:
                printfunc(f"  1D: {value}")
            elif key == 2:
                printfunc(f"  2D: {value}")

        scan_plane_counts = self.dataset["scan_plane"].value_counts()
        printfunc("Scan plane counts:")
        for key, value in scan_plane_counts.items():
            if key == "ax":
                printfunc(f"  Axial: {value}")
            elif key == "cor":
                printfunc(f"  Coronal: {value}")
            elif key == "sag":
                printfunc(f"  Sagittal: {value}")

        scan_type_counts = self.dataset["scan_type"].value_counts()
        printfunc("Scan type counts:")
        for key, value in scan_type_counts.items():
            printfunc(f"  {key}: {value}")

        quality_counts = self.dataset["quality"].value_counts()
        printfunc("Quality counts:")
        for key, value in quality_counts.items():
            printfunc(f"  Quality {key}: {value}")

    def __get_row(self, row, return_metadata=False):
        sample = dict()
        data = torch.load(os.path.join(self.data_dir, row["data_path"]))
        sample[IMAGE_KEY] = torch.from_numpy(
            np.load(os.path.join(self.data_dir, row["recon_path"]))
        )
        sample[PRIOR_KEY] = torch.from_numpy(
            np.load(os.path.join(self.data_dir, row[self.prior_key]))
        )
        sample[KSP_KEY] = data["ksp"]
        sample[MASK_KEY] = data["mask"]
        sample[MPS_KEY] = data["mps"]
        if return_metadata:
            sample["metadata"] = {
                "change_extent": row["change_extent"],
                "scan_type": row["scan_type"],
                "scan_plane": row["scan_plane"],
                "quality": row["quality"],
                "Rro": row["Rro"],
                "Rpe": row["Rpe"],
                "AccelNumDim": row["AccelNumDim"],
            }
        return sample

    def __get_2d(self, idx):
        return self.__get_row(self.dataset.iloc[idx], self.return_metadata)

    def __get_3d(self, idx):
        ds_vol = self.dataset[self.dataset["vol_idx"] == idx]

        samp = self.__get_row(ds_vol.iloc[0], self.return_metadata)

        out = {}
        if self.return_metadata:
            out["metadata"] = samp["metadata"]

        Nslc = ds_vol.shape[0]
        Nc, Kx, Ky = samp[KSP_KEY].shape
        Nx, Ny = samp[IMAGE_KEY].shape

        ksp = torch.zeros((Nc, Kx, Ky, Nslc), dtype=samp[KSP_KEY].dtype)
        mask = torch.zeros((Kx, Ky, Nslc), dtype=samp[MASK_KEY].dtype)
        mps = torch.zeros((Nc, Kx, Ky, Nslc), dtype=samp[MPS_KEY].dtype)
        image = torch.zeros((Nx, Ny, Nslc), dtype=samp[IMAGE_KEY].dtype)
        prior = torch.zeros((Nx, Ny, Nslc), dtype=samp[PRIOR_KEY].dtype)

        ksp[..., 0] = samp[KSP_KEY]
        mask[..., 0] = samp[MASK_KEY]
        mps[..., 0] = samp[MPS_KEY]
        image[..., 0] = samp[IMAGE_KEY]
        prior[..., 0] = samp[PRIOR_KEY]

        for i in range(1, Nslc):
            samp = self.__get_row(ds_vol.iloc[i], return_metadata=False)
            ksp[..., i] = samp[KSP_KEY]
            mask[..., i] = samp[MASK_KEY]
            mps[..., i] = samp[MPS_KEY]
            image[..., i] = samp[IMAGE_KEY]
            prior[..., i] = samp[PRIOR_KEY]

        out[IMAGE_KEY] = image
        out[PRIOR_KEY] = prior
        out[KSP_KEY] = ksp
        out[MASK_KEY] = mask
        out[MPS_KEY] = mps

        return out

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Will return an item which is a torch dict with the following keys:
        -  KSP_KEY: k-space (C, X, Y, [Z])
        -  MASK_KEY: ksp_mask (X, Y, [Z])
        -  MPS_KEY: sensitivity maps (C, X, Y, [Z])
        -  IMAGE_KEY: gt recon (X, Y, [Z])
        -  PRIOR_KEY: prior image (X, Y, [Z])
        """

        # map to shuffled index
        if isinstance(idx, int):
            idx = self.INDEX[idx]

        if self.load_dimension == Dimension.DIM_3D:
            return self.__get_3d(idx)
        else:
            return self.__get_2d(idx)

    def __len__(self):
        if self.load_dimension == Dimension.DIM_3D:
            return len(self.dataset[self.dataset["is_middle_slice"]])
        else:
            return len(self.dataset)
