from laps import DATA_DIR
from pathlib import Path
import urllib.request
from typing import Literal
import pandas as pd
import requests
from tqdm import tqdm
import re
import numpy as np
import torch

SLAM_SDR_URL = "https://stacks.stanford.edu/file/rq296rb2765"
SLAM_DIR = DATA_DIR / "slam"
SLAM_TRAIN_DIR = DATA_DIR / "slam-train" # directory for saving slice-by-slice complex follow-up data for training
SLAM_TRAIN_DICOM_DIR = DATA_DIR / "slam-train-dicom" # directory for saving slice-by-slice prior DICOM data for training
SLAM_TEST_DIR = DATA_DIR / "slam-test" # directory for saving slice-by-slice complex follow-up data for inference

def format_scan_type(s):
    """
    Standardize Scan Type
    """
    s = s.lower()
    if ("ax" in s) and ("3d" in s) and ("t1") in s:
        return "Ax_T1_3D"
    elif ("ax" in s) and ("t2" in s) and ("flair" in s):
        return "Ax_T2_FLAIR_2D"
    elif ("ax" in s) and ("t2" in s):
        return "Ax_T2_2D"
    elif ("cor_obl" in s or "corobl" in s) and ("t2" in s):
        return "Cor_Obl_T2_2D"
    elif "sag_t2_cube" in s:
        return "Sag_T2_Cube_3D"
    elif "sag_t2_flair" in s:
        return "Sag_T2_FLAIR_3D"
    elif "sag" in s and "t1" in s and ("cube" in s or "3d" in s):
        return "Sag_T1_3D"
    elif "sag" in s and "t1" in s and "flair" in s:
        return "Sag_T1_FLAIR_2D"
    elif "sag" in s and "t2" in s and "3d" in s:
        return "Sag_T2_FLAIR_3D"
    elif "sag" in s and "t2" in s and "cube" in s:
        return "Sag_T2_3D"
    else:
        return "Unknown"
    
def get_filesize_bytes(url: str) -> int:
    r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, allow_redirects=True, timeout=30)
    cr = r.headers.get("Content-Range", "")
    m = re.search(r"/(\d+)$", cr)
    if not m:
        return 0
    return int(m.group(1))

def pull_slam_metadata(minimal=False, override=False, verbose=True) -> None:
    """
    Downloads SLAM metadata CSV file into the appropriate data directory.
    """
    files = [
        "README.md",
        "example.py",
        "test.csv",
        "train.csv",
    ]
    SLAM_DIR.mkdir(parents=True, exist_ok=True)    
    for f in files:
        if override or not (SLAM_DIR / f).exists():
            url = f"{SLAM_SDR_URL}/{f}"
            if verbose:
                print(f"Downloading {url} to {SLAM_DIR / f}...")
            urllib.request.urlretrieve(url, SLAM_DIR / f)    
    if minimal:
        # Shrink metadata files for minimal download 
        test_csv = SLAM_DIR / "test.csv"
        df_test = pd.read_csv(test_csv)
        df_test_min = df_test.iloc[[6]]
        df_test_min.reset_index(drop=True, inplace=True)
        df_test_min.loc[:, "index"] = range(len(df_test_min))   
        df_test_min.to_csv(test_csv, index=False)
        train_csv = SLAM_DIR / "train.csv"
        df_train = pd.read_csv(train_csv)
        df_train_min = df_train.iloc[0:5]
        df_train_min.reset_index(drop=True, inplace=True)
        df_train_min.loc[:, "index"] = range(len(df_train_min))
        df_train_min.to_csv(train_csv, index=False)

def pull_slam_volume_data(
        split: Literal["train", "test"],
        override: bool = False,
        load_ksp: bool = False,
) -> None:
    """
    Downloads SLAM dataset files into the appropriate data directory.
    Args:
        split: "train" or "test"
        minimal: If True, only downloads a minimal subset of the data for testing purposes.
        override: If True, re-downloads files even if they already exist.
        load_ksp: If True, downloads k-space data files; otherwise, downloads image data files.
    """
    assert split in ["train", "test"], "split must be 'train' or 'test'"
    assert (SLAM_DIR / f"{split}.csv").exists(), f"Metadata file for split '{split}' not found. Please download metadata first."

    load_keys = ["recon_path", "prior_path"]
    if load_ksp: 
        load_keys.append("ksp_path")
    
    base_url = f"{SLAM_SDR_URL}/recon/{split}"
    split_dir = SLAM_DIR / "recon" / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # build files to download
    df = pd.read_csv(SLAM_DIR / f"{split}.csv")
    files = []
    sizes = []
    for key in load_keys:
        for path in df[key].dropna().unique():
            files.append(path)
    for path in tqdm(files, desc="Preparing file list"):
        url = f"{base_url}/{path}"
        size_byte = get_filesize_bytes(url)
        sizes.append(size_byte)
    
    print(f"Total files to download for SLAM {split} split: {len(files)}")
    total_size = sum(sizes)
    print(f"Total download size: {total_size / (1024**2):.2f} MB")

    # build pbar for download progress in file size
    pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading SLAM {split} data")
    for path, size in zip(files, sizes):
        dest_path = split_dir / path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if override or not dest_path.exists():
            url = f"{base_url}/{path}"
            urllib.request.urlretrieve(url, dest_path)
        pbar.update(size)
    pbar.close()
    
def pull_slam_dataset_volumes(
        minimal: bool = False,
        override: bool = True,
        verbose: bool = True) -> None:
    """
    Downloads the SLAM dataset (metadata and data files) to ./data/ folder relative to the project root.
    Args:
        minimal: If True, only downloads a minimal subset of the data for testing purposes.
        override: If True, re-downloads files even if they already exist.
        verbose: If True, prints download progress.
    """
    pull_slam_metadata(override=override, minimal=minimal, verbose=verbose)
    for split, load_ksp in [("train", not minimal), ("test", True)]:
        pull_slam_volume_data(split=split, override=override, load_ksp=load_ksp)

def prepare_slam_train(dicom=False) -> None:
    """
    Save SLAM training dataset slice by slice for compatibility with training script.
    """
    root = SLAM_TRAIN_DICOM_DIR if dicom else SLAM_TRAIN_DIR
    key = "prior_path" if dicom else "recon_path"
    root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SLAM_DIR / "train.csv")
    df = df.dropna(subset=[key]).reset_index(drop=True)
    row_nr = []
    image_uuid = []
    quality = []
    split = []
    ii = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing SLAM training data"):
        vol_pth = row[key]
        vol_pth_full = SLAM_DIR / "recon" / "train" / row[key]
        q = row["quality"]
        vol = np.load(vol_pth_full)["arr_0"]  # shape: (H, W, S)

        if dicom:
            vol = np.abs(vol)
        else:
            vol = vol.astype(np.complex64)
        
        # pre-determined energy trimming
        start = row["slc_start_idx"]
        end = row["slc_end_idx"]
        vol = vol[:, :, start:end]

        for s in range(vol.shape[2]):
            slice_pth = str(vol_pth).replace(".npz", f"_{ii:06d}.npy")
            ii += 1
            slice_pth_full = root / slice_pth
            slice_pth_full.parent.mkdir(parents=True, exist_ok=True)
            np.save(slice_pth_full, vol[:, :, s])
            row_nr.append(idx)
            image_uuid.append(slice_pth)
            quality.append(q)
            split.append("train")

    df_slices = pd.DataFrame({
        "row_nr": row_nr,
        "image_uuid": image_uuid,
        "quality": quality,
        "split": split,
    })
    csv_file = str(root).rstrip("/") + ".csv"
    df_slices.to_csv(csv_file, index=False)
    print(f"Saved slice-by-slice SLAM training data to {root} and metadata to {csv_file}.")


def prepare_slam_test() -> None:
    """
    Save SLAM training dataset slice by slice for compatibility with inference script.
    """
    root = SLAM_TEST_DIR
    root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SLAM_DIR / "test.csv")
    df_out_keys = {
        "index": [],
        "subj_index": [],
        "scan_index": [],
        "slice_index": [],
        "recon_path": [],
        "prior_path": [],
        "prior_path_reg": [],
        "mask_path": [],
        "data_path": [],
        "is_middle_slice": [],
        "change_extent": [],
        "scan_plane": [],
        "scan_type": [],
        "quality": [],
        "Nc": [],
        "Kx": [],
        "Ky": [],
        "Nx": [],
        "Ny": [],
        "Nz": [],
        "Rro": [],
        "Rpe": [],
        "AccelNumDim": [],
    }
    ii = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing SLAM test data"):
        vol_pth = row["recon_path"].replace("/recon.npz", "")

        vol_pth_full = SLAM_DIR / "recon" / "test" / vol_pth
        vol_pth_full_out = root / vol_pth
        recon = np.load(vol_pth_full / "recon.npz")["arr_0"]  # shape: (H, W, S)
        prior = np.load(vol_pth_full / "prior.npz")["arr_0"]  # shape: (H, W, S)
        data = np.load(vol_pth_full / "data.npz")
        ksp = data["ksp"]
        mps = data["mps"]
        mask = data["mask"]

        # pre-determined energy trimming
        start = row["slc_start_idx"]
        end = row["slc_end_idx"]
        recon = recon[:, :, start:end]
        prior = prior[:, :, start:end]
        ksp = torch.from_numpy(ksp[..., start:end])
        mps = torch.from_numpy(mps[..., start:end])
        mask = torch.from_numpy(mask[:, :, start:end])
        N = recon.shape[2]
        middle_slice_idx = N // 2
        for s in range(recon.shape[2]):
            is_middle_slice = (s == middle_slice_idx)
            slice_folder = Path(str(vol_pth_full_out).rstrip("/") + f"_{s:06d}")
            slice_folder.mkdir(parents=True, exist_ok=True)
            recon_path = slice_folder / "recon.npy"
            prior_path = slice_folder / "prior.npy"
            data_path = slice_folder / "data.pt"
            np.save(recon_path, recon[..., s])
            np.save(prior_path, prior[..., s])
            torch.save(
                {
                    "ksp": ksp[..., s].clone(),
                    "mps": mps[..., s].clone(),
                    "mask": mask[..., s].clone(),
                },
                data_path,
            )
            df_out_keys["index"].append(ii)
            df_out_keys["subj_index"].append(row["subj_index"])
            df_out_keys["scan_index"].append(idx)
            df_out_keys["slice_index"].append(s)
            df_out_keys["recon_path"].append(str(recon_path.relative_to(root)))
            df_out_keys["prior_path"].append(str(prior_path.relative_to(root)))
            df_out_keys["prior_path_reg"].append(str(prior_path.relative_to(root)))
            df_out_keys["mask_path"].append("")
            df_out_keys["data_path"].append(str(data_path.relative_to(root)))
            df_out_keys["is_middle_slice"].append(is_middle_slice)
            df_out_keys["change_extent"].append(row["change_extent"])
            df_out_keys["scan_plane"].append(row["scan_plane"])
            df_out_keys["scan_type"].append(format_scan_type(row["scan_type"]))
            df_out_keys["quality"].append(row["quality"])
            df_out_keys["Nc"].append(row["Nc"])
            df_out_keys["Kx"].append(row["Kx"])
            df_out_keys["Ky"].append(row["Ky"])
            df_out_keys["Nx"].append(row["Nx"])
            df_out_keys["Ny"].append(row["Ny"])
            df_out_keys["Nz"].append(row["Nz"])
            df_out_keys["Rro"].append(row["Rro"])
            df_out_keys["Rpe"].append(row["Rpe"])
            df_out_keys["AccelNumDim"].append(row["AccelNumDim"])
            ii += 1

    df_out = pd.DataFrame(df_out_keys)
    csv_file = str(root).rstrip("/") + ".csv"
    df_out.to_csv(csv_file, index=False)
    print(f"Saved slice-by-slice SLAM test data to {root} and metadata to {csv_file}.")

