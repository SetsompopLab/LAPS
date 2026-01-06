import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


dicom_dataset = Path(
    "/local_mount/space/mayday/data/users/zachs/fast-mri-ldm/pre_processing/preprocess-bruno-2025-data"
)
dicom_dst_dir = Path(
    "/local_mount/space/mayday/data/datasets/ladyyy/datasets/med_vae_train//bruno_dicoms_v2"
)
os.makedirs(dicom_dst_dir, exist_ok=True)

df = pd.read_csv(dicom_dataset / "data" / "updated_dataset_w_dicoms_v2.csv")

save_df = []
global_cnt = 0

train_pct = 0.7
val_pct = 0.15
test_pct = 0.15

# Create a list of probabilities
probabilities = [train_pct, val_pct, test_pct]
splits = ["train", "val", "test"]

for j in tqdm(range(len(df))):
    row = df.iloc[j]
    # Randomly assign a split for this row (this split will be used for each saved slice)
    split = np.random.choice(splits, p=probabilities)

    # Process both "new" and "prior" scans.
    for status in ["new", "prior"]:
        # Get the scan path for the current status.
        scan_path = row.get(f"{status}_scan_path", None)
        if pd.isnull(scan_path) or not os.path.exists(scan_path):
            continue

        try:
            # Load the volume (assumed to be a 3D numpy array)
            volume = np.load(scan_path)
        except Exception as e:
            print(f"Error loading volume at {scan_path}: {e}")
            continue

        # Compute energy for each slice along the first dimension.
        # Energy is defined as the sum of absolute pixel values.
        energies = np.array(
            [np.sum(np.abs(volume[i, ...]) ** 2) for i in range(volume.shape[0])]
        )
        max_energy = energies.max()
        threshold = 0.3 * max_energy

        # Find indices of slices that meet the energy threshold.
        valid_indices = [i for i, e in enumerate(energies) if e >= threshold]
        if not valid_indices:
            print(f"No slices meet the energy threshold in volume at {scan_path}")
            continue

        # Iterate over valid slices.
        for i in valid_indices:
            slice_img = volume[i, ...]
            rand_id = random.randint(0, 1000000)
            filename = f"{row['exam_id']}_{row['scanner']}_{row['scan_type']}_{status}_{i}_{rand_id}.npy"
            filename = filename.replace(" ", "_").replace("/", "_")
            file_id = filename.split(".")[0]
            save_path = os.path.join(dicom_dst_dir, filename)
            try:
                np.save(save_path, slice_img)
            except Exception as e:
                print(f"Error saving slice {i} for {scan_path}: {e}")
                continue

            save_df.append(
                {
                    "row_nr": global_cnt,
                    "image_uuid": file_id,
                    "split": split,
                }
            )
            global_cnt += 1

save_df = pd.DataFrame(save_df)
# Save the DataFrame as a CSV file
save_df.to_csv(dicom_dst_dir.parent / "dicom_ds_v2.csv", index=False)
