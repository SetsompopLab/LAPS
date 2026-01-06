import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

from laps import PROJECT_ROOT
from laps.dataloaders import LoaderType


@dataclass
class TrainingArgs:
    """
    I/0 Parameters
    """

    # Path to pretrained model or model identifier from huggingface.co/models, with revision and variant if applicable.
    pretrained_model_name_or_path: str = "yurman/uncond_sd2-base"
    model_revision: Optional[str] = None
    model_variant: Optional[str] = None
    # Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier
    # of the local or remote repository specified with --pretrained_model_name_or_path.
    non_ema_revision: Optional[str] = None
    # HuggingFace Hub syncing, and id if repository should keep in sync with the local `output_dir`.
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    # For tracking. supported platforms are "tensorboard" (default), "wandb", or "all".
    report_to: str = "tensorboard"
    tracker_project_name: str = "laps-complex-sd"
    # The directory where the downloaded models and datasets will be stored.
    cache_dir: Optional[str] = None
    # The output directory where the model predictions and checkpoints will be written.
    output_dir: str = "out/laps-complex-sd"
    # The name of the repository to keep in sync with the local `output_dir`.
    logging_dir: str = "logs"
    # Save a checkpoint of the training state every X updates.
    # These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`.
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = 3
    # Whether training should be resumed from a previous checkpoint.
    # Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.
    resume_from_checkpoint: Optional[str] = None

    """
    VAE Parameters
    """
    custom_vae_path: Optional[Union[str, os.PathLike]] = None
    vae_num_channels: int = 2
    downsampling_factor: int = 4
    # replace original VAE scaling factor. If None, the original value is used.
    vae_scaling_factor: Optional[float] = None
    update_scaling_factor: bool = False
    scaling_iters: int = 1000
    """
    Dataset Parameters
    """
    # dataset path(s). either list, or comma-separated string
    dataset: Sequence[LoaderType] = field(
        default_factory=lambda: [
            LoaderType.SLAM_DICOM,
            LoaderType.SLAM,
        ]
    )

    # The resolution for input images, all the images in the train/validation dataset will be resized to this resolution
    resolution: int = 256
    # Whether to center crop the input images to the resolution.
    center_crop: bool = False
    # adds horiz, vert flips, as well as spatial tranpose.
    random_flip: bool = True
    # Random rotation angle for data augmentation.
    rot_degree: float = 10

    # Choose whether to use complex dataset or stick to magnitude-weighted
    # Complex network:
    # - complex images scaled [-1, 1], magnitude images scaled [0,1]
    # - Channels [Re, Im, Mag] for 3-channel, or [Re Im] for 2-channel
    # Magnitude network:
    # - All images scaled [-1, 1], magnitude repeated over channels (previous method)
    train_complex_dataset: bool = True
    # dropout some complex images to magnitude
    complex_dropout_frac: float = 0
    # add global phase to complex images
    complex_global_phase_modulation: bool = True

    # Batch size (per device) for the training dataloader.
    train_batch_size: int = 64
    # Number of training epochs.
    num_train_epochs: int = 250
    # Total number of training steps to perform.  If provided, overrides num_train_epochs.
    max_train_steps: Optional[int] = 100000
    # For debugging purposes (overfitting) truncate the number of training examples to this value if set.
    max_train_samples: Optional[int] = None
    # Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    dataloader_num_workers: int = 0

    """
    Validation Parameters
    """
    # Num inference steps
    num_inference_steps: int = 50
    # Run validation every X global steps.
    validation_steps: int = 500
    # Number of validation samples to generate.
    num_validation_images: int = 4

    """
    General Diffusion Model Parameters
    """
    # The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`.
    # If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.
    prediction_type: Optional[str] = "v_prediction"
    # The scale of noise offset.
    noise_offset: float = 0.0
    # The scale of input perturbation.
    input_perturbation: float = 0.0
    # SNR weighting gamma to be used if rebalancing the loss.
    # Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.
    snr_gamma: float = 5.0
    # Dont rescale snr to terminal snr=0
    dont_rescale_snr_to_zero: bool = False

    """
    Optimization Parameters
    """
    # Number of updates steps to accumulate before performing a backward/update pass.
    gradient_accumulation_steps: int = 1
    # Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
    gradient_checkpointing: bool = False
    # Initial learning rate (after the potential warmup period) to use.
    learning_rate: float = 5e-5
    # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
    scale_lr: bool = False
    # The scheduler type to use.
    # Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    lr_scheduler: str = "constant"
    # Number of steps for the warmup in the lr scheduler.
    lr_warmup_steps: int = 500
    # Max gradient norm.
    max_grad_norm: float = 1.0
    # Adam Optimizer Parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08

    """
    Memory/Performance Parameters
    """
    # A seed for reproducible training.
    seed: Optional[int] = None
    # For distributed training: local_rank
    local_rank: int = -1
    # Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.
    mixed_precision: str = "no"
    # Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training.
    # For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    allow_tf32: bool = False

    """
    EMA Parameters
    """
    # Whether to use Exponential Moving Average for the final model weights.
    use_ema: bool = False

    """
    DEPECIATED: Conditional arguments
    """
    # Enable training of a conditional LDM instead of unconditional. Enabling conditional and turning
    # off complex training reduces to default way to train on stable diffusion
    use_conditional: bool = False
    # Initial learning rate for embeddings
    embeds_lr: float = 1e-4
    # Classifier-free guidance scaling factor
    cfg_scaling: float = 1
    # default text prompt
    default_text_prompt: str = (
        "An empty, flat black image with a MRI brain axial scan in the center"
    )

# Fine tuning configuration for unconditional SD2 base on complex MRI dataset
sd2_base_ft = TrainingArgs(
    pretrained_model_name_or_path="yurman/uncond-sd2-base-complex-4",
    push_to_hub=True,
    hub_model_id="uncond-sd2-ft",
    tracker_project_name="uncond-sd2-ft",
    output_dir="out/uncond-sd2-ft",
    checkpointing_steps=2000,
    checkpoints_total_limit=1,
    dataset=[
        LoaderType.SLAM,
        LoaderType.SLAM_DICOM,
    ],
    resolution=256,
    random_flip=True,
    rot_degree=10,
    train_batch_size=8,
    gradient_accumulation_steps=3,
    learning_rate=5e-5,
    max_train_steps=100000,
    dataloader_num_workers=8,
    num_inference_steps=50,
    validation_steps=250,
    num_validation_images=4,
    prediction_type="v_prediction",
    use_conditional=False,
    train_complex_dataset=True,
    complex_dropout_frac=0,
    complex_global_phase_modulation=True,
    dont_rescale_snr_to_zero=False,
    seed=42,
    use_ema=True,
    lr_scheduler="cosine",
    lr_warmup_steps=3000,
    custom_vae_path=PROJECT_ROOT / "models" / "medvae_4",
    downsampling_factor=4,
    vae_scaling_factor=0.12,
    report_to="wandb",
)
