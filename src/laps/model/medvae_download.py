from huggingface_hub import HfApi, snapshot_download
import os
from laps import PROJECT_ROOT
MEDVAE_PATH = PROJECT_ROOT / "models" / "medvae_4"

def push_model_folder(
    repo_id: str = "zachary-shah/medvae_4",
    commit_message: str = "Upload model"
):
    """
    Push a local folder to Hugging Face as a model repository.
    """
    api = HfApi()

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True
    )

    api.upload_folder(
        folder_path=MEDVAE_PATH,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message
    )


def download_model_folder(
    repo_id: str = "zachary-shah/medvae_4",
    revision: str | None = None
):
    """
    Download a Hugging Face model repository into a local folder.
    """
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=MEDVAE_PATH,
        local_dir_use_symlinks=False
    )
