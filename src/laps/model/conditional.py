import os
import random
from typing import Optional, Union

import numpy as np
import torch.utils.checkpoint
from diffusers.utils import is_wandb_available, make_image_grid
from transformers import CLIPTextModel, CLIPTokenizer

from laps.dataloaders.labels import (
    COLLATE_IMAGE_KEY,
    COLLATE_TEXT_KEY,
    IMAGE_KEY,
    TEXT_KEY,
)

__all__ = [
    "save_conditional_model_card",
    "tokenize_captions",
    "conditional_collate_fn",
    "get_conditional_collate_fn",
]


def save_conditional_model_card(
    args: dict,
    repo_id: str,
    images=None,
    repo_folder=None,
    dataset_str: str = None,
    for_training: bool = False,
):
    """
    Saving huggingface model card for a conditional model.
    """

    pretrained_model = args.get("pretrained_model_name_or_path", "N/A")
    default_prompt = args.get("default_text_prompt", "N/A")
    dataset = args.get("dataset", "N/A")
    if isinstance(dataset, list):
        dataset = "-".join(dataset)

    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, min(len(images), 8))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    if dataset_str is not None:
        dataset_str_full = f"on the **{dataset}** dataset "
    else:
        dataset_str_full = ""

    yaml = f"""
    ---
    license: creativeml-openrail-m
    base_model: {pretrained_model}
    tags:
    - stable-diffusion
    - stable-diffusion-diffusers
    - text-to-image
    - diffusers
    inference: true
    ---
    """.replace(
        "    ", ""
    )
    model_card = f"""
    # Text-to-image finetuning - {repo_id}

    This pipeline was finetuned from **{pretrained_model}**
    {dataset_str_full}for brain image generation.
    Below are some example images generated with the finetuned pipeline: \n
    {img_str}

    ## Pipeline usage

    You can use the pipeline like so:

    ```python
    from diffusers import StableDiffusionPipeline
    import torch

    pipeline = StableDiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float32)
    prompt = "{default_prompt}"
    image = pipeline(prompt).images[0]
    image.save("my_image.png")
    ```

    ## Training info
    """.replace(
        "    ", ""
    )

    training_info = f"""These are the key hyperparameters used during training:

    * Epochs: {args.get("num_train_epochs", "N/A")}
    * Max Train Steps: {args.get("max_train_steps", "N/A")}
    * Learning rate: {args.get("learning_rate", "N/A")}
    * embeds learning rate: {args.get("embeds_lr", "N/A")}
    * Batch size: {args.get("train_batch_size", "N/A")}
    * Classifier free guidance: {args.get("cfg_scaling", "N/A")}
    * VAE scaling: {args.get("vae_scaling_factor", "N/A")}
    * Input perturbation: {args.get("input_perturbation", "N/A")}
    * Noise offset: {args.get("noise_offset", "N/A")}
    * Gradient accumulation steps: {args.get("gradient_accumulation_steps", "N/A")}
    * Image resolution: {args.get("resolution", "N/A")}
    * Mixed-precision: {args.get("mixed_precision", "N/A")}
    * Max rotation degree: {args.get("rot_degree", "N/A")}
    * Prediction Type: {args.get("prediction_type", "N/A")}
    * SNR Gamma: {args.get("snr_gamma", "N/A")}
    """.replace(
        "    ", ""
    )

    model_card += training_info

    if for_training:
        wandb_info = ""
        if is_wandb_available():
            import wandb

            wandb_run_url = None
            if wandb.run is not None:
                wandb_run_url = wandb.run.url
        if wandb_run_url is not None:
            wandb_info = f"""
            More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
            """.replace(
                "    ", ""
            )
        model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def tokenize_captions(tokenizer: CLIPTokenizer, examples, is_train=True):
    """
    Returns the tokenized captions for the given examples.
    """

    captions = []
    for example in examples:
        caption = example[TEXT_KEY]
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                "Caption column should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids


def conditional_collate_fn(
    examples, tokenizer: CLIPTokenizer, constant_prompt: Optional[str] = None
):
    """
    Stack the pixel values and tokenize the captions and return as a dict
    for torch.dataloader collation.
    """

    pixel_values = torch.stack([example[IMAGE_KEY] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    # For constant-prompt conditioning, if desired.
    if constant_prompt is not None:
        for example in examples:
            example[TEXT_KEY] = constant_prompt

    input_ids = tokenize_captions(tokenizer, examples)
    input_ids = torch.stack([input_id for input_id in input_ids])
    return {COLLATE_IMAGE_KEY: pixel_values, COLLATE_TEXT_KEY: input_ids}


def get_conditional_collate_fn(tokenizer: CLIPTokenizer, prompt: Optional[str] = None):
    """
    Returns the collate function for the given tokenizer, for a constant prompt.
    """
    return lambda examples: conditional_collate_fn(examples, tokenizer, prompt)
