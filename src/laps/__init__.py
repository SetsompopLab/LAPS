from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
WANDB_DIR = PROJECT_ROOT
TMP_DIR = PROJECT_ROOT / "tmp"