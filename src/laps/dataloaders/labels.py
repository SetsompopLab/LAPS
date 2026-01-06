__all__ = [
    "IMAGE_KEY",
    "PRIOR_KEY",
    "KSP_KEY",
    "MPS_KEY",
    "LABEL_KEY",
    "MASK_KEY",
    "TEXT_KEY",
    "COLLATE_IMAGE_KEY",
    "COLLATE_TEXT_KEY",
    "QUALITY_KEY",
]

# Keys to use for torch.utils.Dataset
IMAGE_KEY = "img"
PRIOR_KEY = "prior"
KSP_KEY = "ksp"
MPS_KEY = "mps"
LABEL_KEY = "lbl"
MASK_KEY = "msk"
TEXT_KEY = "txt"
QUALITY_KEY = "quality"

# Keys to use for torch.utils.DataLoader
COLLATE_IMAGE_KEY = "pixel_values"
COLLATE_TEXT_KEY = "input_ids"
