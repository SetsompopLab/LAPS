import os
from typing import Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers import UNet2DConditionModel, UNet2DModel
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_2d import (
    Transformer2DModel,
    Transformer2DSelfAttentionModel,
)
from diffusers.utils import is_wandb_available, make_image_grid

from laps.dataloaders.labels import COLLATE_IMAGE_KEY, IMAGE_KEY, QUALITY_KEY

__all__ = [
    "copy_weights",
    "replace_transformer_attention",
    "Unet2DConditionModel_to_Unet2DModel",
    "save_unconditional_model_card",
    "unconditional_collate_fn",
]


def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    """
    Utility function to copy state dict from one module to another.

    Parameters
    ----------
    src: nn.Module
        Source module to copy from.
    dst: nn.Module
        Destination module to copy to.

    Returns
    -------
    dst: nn.Module
    """

    assert (
        src.__class__.__name__ == dst.__class__.__name__
    ), "src and dst must have the same class name"

    if src is not None:

        if isinstance(src, nn.Module):
            dst.load_state_dict(src.state_dict(), strict=False)

        elif isinstance(src, nn.Parameter):
            dst.data.copy_(src.data)

    return dst


def replace_transformer_attention(
    old_attention: Transformer2DModel,
    new_attention: Union[Attention, Transformer2DSelfAttentionModel],
    crossattn_replacement: str,
    crossattn_bias_replacement: str = True,
) -> Union[Attention, Transformer2DSelfAttentionModel]:
    """
    Replace the weights in the old Transformer2DModel with the new Attention or Transformer2DSelfAttentionModel
    for ejecting the cross-attention from unet blocks.

    Parameters
    ----------
    old_attention: Transformer2DModel
        Old transformer attention model to copy from.
    new_attention: Union[Attention, Transformer2DSelfAttentionModel]
        New attention model to copy to.
    crossattn_replacement: str
        Type of replacement:
            "attention" : Replace Transformer2DModel with Attention, which is a single self-attention layer with no
                          cross-attention or residuals.
            "transformer" : Replace Transformer2DModel with Transformer2DSelfAttentionModel, which a custom architecture
                          that replaces the full cross-attention block with just the bias from the cross-attention block.
    crossattn_bias_replacement: str
        Whether to replace the cross-attention bias in the new attention model with the bias from the old attention model.
        Default is True.

    Returns
    -------
    new_attention: Union[Attention, Transformer2DSelfAttentionModel]
    """

    # supported replacement conditions
    assert (
        old_attention.__class__.__name__ == "Transformer2DModel"
    ), "old_attention expected to be Transformer2DModel"
    assert crossattn_replacement in [
        "attention",
        "transformer",
    ], "crossattn_replacement must be 'attention' or 'transformer'"
    assert (
        old_attention.transformer_blocks[0].norm_type == "layer_norm"
    ), "Transformer2DModel must use layer_norm"
    assert (
        old_attention.caption_projection is None
    ), "Transformer2DModel must not use caption_projection"
    assert (
        not old_attention.is_input_patches
    ), "Transformer2DModel must be continuous input"
    assert (
        not old_attention.is_input_vectorized
    ), "Transformer2DModel must be continuous input"

    if crossattn_replacement == "attention":
        assert (
            new_attention.__class__.__name__ == "Attention"
        ), "new_attention expected to be Attention"
        assert (
            len(old_attention.transformer_blocks) == 1
        ), "old_attention.transformer_blocks must have length 1"

        old_transformer_block = old_attention.transformer_blocks[0]
        new_attention = copy_weights(old_transformer_block.attn1, new_attention)
        # get group norm in there as well
        new_attention.group_norm = copy_weights(
            old_attention.norm, new_attention.group_norm
        )
        # ensure we have resid connection
        new_attention.residual_connection = True

        if crossattn_bias_replacement:
            new_attention.to_out[0].bias.data += (
                old_transformer_block.attn2.to_out[0].bias.data.detach().cpu()
            )

    else:
        assert (
            new_attention.__class__.__name__ == "Transformer2DSelfAttentionModel"
        ), "new_attention expected to be Transformer2DSelfAttentionModel"

        # replicate weights
        new_attention.norm = copy_weights(old_attention.norm, new_attention.norm)
        new_attention.proj_in = copy_weights(
            old_attention.proj_in, new_attention.proj_in
        )
        new_attention.proj_out = copy_weights(
            old_attention.proj_out, new_attention.proj_out
        )

        new_transformer_blocks = []

        for k in range(len(old_attention.transformer_blocks)):
            new_transformer_block = new_attention.transformer_blocks[k]
            old_transformer_block = old_attention.transformer_blocks[k]

            new_transformer_block.pos_embed = copy_weights(
                old_transformer_block.pos_embed, new_transformer_block.pos_embed
            )
            new_transformer_block.norm1 = copy_weights(
                old_transformer_block.norm1, new_transformer_block.norm1
            )
            new_transformer_block.attn1 = copy_weights(
                old_transformer_block.attn1, new_transformer_block.attn1
            )
            new_transformer_block.norm3 = copy_weights(
                old_transformer_block.norm3, new_transformer_block.norm3
            )
            new_transformer_block.ff = copy_weights(
                old_transformer_block.ff, new_transformer_block.ff
            )

            # NOTE: attn2 is cross-attention. This is removed in the Transformer2DSelfAttentionModel.
            # we can still transfer the effect of cross-attention bias.
            if crossattn_bias_replacement:
                new_transformer_block.attn1.to_out[0].bias.data += (
                    old_transformer_block.attn2.to_out[0].bias.data.detach().cpu()
                )

            new_transformer_blocks.append(new_transformer_block)

        new_attention.transformer_blocks = torch.nn.ModuleList(new_transformer_blocks)

    return new_attention


def Unet2DConditionModel_to_Unet2DModel(
    unet: UNet2DConditionModel,
    crossattn_replacement="attention",
    crossattn_bias_replacement: str = True,
) -> UNet2DModel:
    """
    Attempt to recover resnet and self-attention blocks, while discarding cross-attention with hidden embeddings from conditioning text/img/etc

    Parameters
    ----------
    unet: UNet2DConditionModel
        UNet2DConditionModel to copy from.
    crossattn_replacement: str
        see `replace_transformer_attention`
    crossattn_bias_replacement: str
        see `replace_transformer_attention`

    Returns
    -------
    unet_uncond: UNet2DModel
        UNet2DModel with cross-attention replaced with desired self-attention blocktype.
    """
    assert crossattn_replacement in [
        "attention",
        "transformer",
    ], "crossattn_replacement must be 'attention' or 'transformer'"

    cfg = unet.config

    # model parameters
    downsample_padding = cfg.downsample_padding
    downsample_type = cfg.downsample_type
    dropout = cfg.dropout
    flip_sin_to_cos = cfg.flip_sin_to_cos
    freq_shift = cfg.freq_shift
    in_channels = cfg.in_channels
    layers_per_block = cfg.layers_per_block
    mid_block_scale_factor = cfg.mid_block_scale_factor
    norm_eps = cfg.norm_eps
    norm_num_groups = cfg.norm_num_groups
    num_class_embeds = cfg.num_class_embeds
    num_train_timesteps = cfg.num_train_timesteps
    out_channels = cfg.out_channels
    projection_class_embeddings_input_dim = cfg.projection_class_embeddings_input_dim
    resnet_time_scale_shift = cfg.resnet_time_scale_shift
    reverse_transformer_layers_per_block = cfg.reverse_transformer_layers_per_block
    sample_size = cfg.sample_size
    time_cond_proj_dim = cfg.time_cond_proj_dim
    time_embedding_act_fn = cfg.time_embedding_act_fn
    time_embedding_dim = cfg.time_embedding_dim
    time_embedding_type = cfg.time_embedding_type
    timestep_post_act = cfg.timestep_post_act
    transformer_layers_per_block = cfg.transformer_layers_per_block
    upsample_type = cfg.upsample_type
    block_out_channels = cfg["block_out_channels"]
    center_input_samples = cfg["center_input_sample"]
    act_fn = cfg["act_fn"]
    class_embed_type = cfg["class_embed_type"]
    conv_in_kernel = cfg["conv_in_kernel"]
    conv_out_kernel = cfg["conv_out_kernel"]
    class_embeddings_concat = cfg["class_embeddings_concat"]
    attention_head_dim = cfg["attention_head_dim"]
    attention_type = cfg["attention_type"]
    upcast_attention = cfg.upcast_attention
    use_linear_projection = cfg.use_linear_projection  # TRUE
    resnet_skip_time_act = cfg.resnet_skip_time_act
    resnet_out_scale_factor = cfg.resnet_out_scale_factor
    # num_attention_heads = (
    #     cfg.num_attention_heads
    # )  # this should be none but remains unused due to fn call limitations

    # things we shouldn't need anymore
    only_cross_attention = cfg.only_cross_attention
    mid_block_only_cross_attention = cfg.mid_block_only_cross_attention
    dual_cross_attention = cfg.dual_cross_attention
    # encoder_hid_dim = cfg.encoder_hid_dim
    # encoder_hid_dim_type = cfg.encoder_hid_dim_type
    # cross_attention_dim = cfg.cross_attention_dim
    # addition_embed_type = cfg["addition_embed_type"]
    # addition_time_embed_dim = cfg["addition_time_embed_dim"]
    # cross_attention_norm = cfg["cross_attention_norm"]  # layer_norm
    # addition_embed_type_num_heads = cfg["addition_embed_type_num_heads"]

    # ensure conditional model is supported
    assert only_cross_attention in [None, False], "only_cross_attention must be False"
    assert dual_cross_attention in [None, False], "dual_cross_attention must be False"
    assert mid_block_only_cross_attention in [
        None,
        False,
    ], "mid_block_only_cross_attention must be False"

    # block types we will replace
    down_block_types = cfg["down_block_types"]
    mid_block_type = cfg.mid_block_type
    up_block_types = cfg.up_block_types

    new_down_block_types = []
    for down_block_type in down_block_types:
        if down_block_type == "CrossAttnDownBlock2D":
            new_down_block_types.append("SelfAttnDownBlock2D")
        elif down_block_type in [
            "DownBlock2D",
            "AttnDownBlock2D",
            "SelfAttnDownBlock2D",
        ]:
            new_down_block_types.append(down_block_type)
        else:
            raise ValueError(f"Unsupported down_block_type: {down_block_type}")

    new_mid_block_type = None
    if mid_block_type == "UNetMidBlock2DCrossAttn":
        new_mid_block_type = "UNetMidBlock2DSelfAttn"
    else:
        raise ValueError(f"Unsupported mid_block_type: {mid_block_type}")

    new_up_block_types = []
    for up_block_type in up_block_types:
        if up_block_type == "CrossAttnUpBlock2D":
            new_up_block_types.append("SelfAttnUpBlock2D")
        elif up_block_type in ["UpBlock2D", "AttnUpBlock2D", "SelfAttnUpBlock2D"]:
            new_up_block_types.append(up_block_type)
        else:
            raise ValueError(f"Unsupported up_block_type: {up_block_type}")

    # create dummy model
    unet_uncond = UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        center_input_sample=center_input_samples,
        time_embedding_type=time_embedding_type,
        freq_shift=freq_shift,
        flip_sin_to_cos=flip_sin_to_cos,
        down_block_types=new_down_block_types,
        mid_block_type=new_mid_block_type,
        up_block_types=new_up_block_types,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        mid_block_scale_factor=mid_block_scale_factor,
        downsample_padding=downsample_padding,
        dropout=dropout,
        act_fn=act_fn,
        norm_eps=norm_eps,
        transformer_layers_per_block=transformer_layers_per_block,
        reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
        class_embed_type=class_embed_type,
        num_class_embeds=num_class_embeds,
        time_embedding_dim=time_embedding_dim,
        time_embedding_act_fn=time_embedding_act_fn,
        timestep_post_act=timestep_post_act,
        time_cond_proj_dim=time_cond_proj_dim,
        conv_in_kernel=conv_in_kernel,
        conv_out_kernel=conv_out_kernel,
        projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
        class_embeddings_concat=class_embeddings_concat,
        resnet_time_scale_shift=resnet_time_scale_shift,
        attention_head_dim=attention_head_dim,
        num_attention_heads=None,  # =num_attention_heads, #this must be none due to naming issue described in fn call
        downsample_type=downsample_type,
        upsample_type=upsample_type,
        norm_num_groups=norm_num_groups,
        num_train_timesteps=num_train_timesteps,
        use_linear_projection=use_linear_projection,
        attention_type=attention_type,
        upcast_attention=upcast_attention,
        resnet_skip_time_act=resnet_skip_time_act,
        resnet_out_scale_factor=resnet_out_scale_factor,
        add_mid_block_attention=True,
        # attention instantiation type
        use_transformer_attentions=crossattn_replacement == "transformer",
    )

    # replace weights
    unet_uncond.conv_in = copy_weights(unet.conv_in, unet_uncond.conv_in)
    unet_uncond.time_proj = copy_weights(unet.time_proj, unet_uncond.time_proj)
    unet_uncond.time_embedding = copy_weights(
        unet.time_embedding, unet_uncond.time_embedding
    )
    unet_uncond.class_embedding = copy_weights(
        unet.class_embedding, unet_uncond.class_embedding
    )
    unet_uncond.time_embed_act = copy_weights(
        unet.time_embed_act, unet_uncond.time_embed_act
    )

    # down blocks
    for i, down_block_type in enumerate(down_block_types):

        old_down_block = unet.down_blocks[i]
        new_down_block = unet_uncond.down_blocks[i]

        # DOWN BLOCK REPLACEMENTS
        if down_block_type == "CrossAttnDownBlock2D":
            # replace with AttnDownBlock2D
            assert (
                new_down_block.__class__.__name__ == "SelfAttnDownBlock2D"
            ), "new_down_block expected to be AttnDownBlock2D"

            # copy resnets and dnownsamplers
            new_down_block.resnets = copy_weights(
                old_down_block.resnets, new_down_block.resnets
            )
            new_down_block.downsamplers = copy_weights(
                old_down_block.downsamplers, new_down_block.downsamplers
            )

            # replace each Transformer2DModel block with Attention or Transformer2DSelfAttentionModel
            new_attentions = []
            for j in range(len(old_down_block.attentions)):
                new_down_block_attention = replace_transformer_attention(
                    old_down_block.attentions[j],
                    new_down_block.attentions[j],
                    crossattn_replacement,
                    crossattn_bias_replacement=crossattn_bias_replacement,
                )

                new_attentions.append(new_down_block_attention)

            new_down_block.attentions = torch.nn.ModuleList(new_attentions)

            unet_uncond.down_blocks[i] = new_down_block

        elif down_block_type in [
            "DownBlock2D",
            "AttnDownBlock2D",
            "SelfAttnDownBlock2D",
        ]:
            # direct replacement
            unet_uncond.down_blocks[i] = copy_weights(
                old_down_block, unet_uncond.down_blocks[i]
            )
        else:
            raise ValueError(f"Unsupported down_block_type: {down_block_type}")

    # mid block
    if mid_block_type == "UNetMidBlock2DCrossAttn":

        mid_block = unet.mid_block
        new_mid_block = unet_uncond.mid_block
        assert (
            new_mid_block.__class__.__name__ == "UNetMidBlock2DSelfAttn"
        ), "new_mid_block expected to be UNetMidBlock2DSimpleCrossAttn"

        # copy non-attention weights
        new_mid_block.resnets = copy_weights(mid_block.resnets, new_mid_block.resnets)

        # change attention but preserve weights
        new_attentions = []
        for j in range(len(mid_block.attentions)):
            new_attentions.append(
                replace_transformer_attention(
                    mid_block.attentions[j],
                    new_mid_block.attentions[j],
                    crossattn_replacement,
                    crossattn_bias_replacement=crossattn_bias_replacement,
                )
            )
        new_mid_block.attentions = torch.nn.ModuleList(new_attentions)
    elif mid_block_type in ["UNetMidBlock2D", "UNetMidBlock2DSimpleCrossAttn"]:
        # preserve all weights in these cases
        unet_uncond.mid_block = copy_weights(unet.mid_block, unet_uncond.mid_block)
    else:
        raise ValueError(f"Unsupported mid_block_type: {mid_block_type}")

    # up blocks
    for i, up_block_type in enumerate(up_block_types):

        old_up_block = unet.up_blocks[i]
        new_up_block = unet_uncond.up_blocks[i]

        if up_block_type == "CrossAttnUpBlock2D":

            assert (
                new_up_block.__class__.__name__ == "SelfAttnUpBlock2D"
            ), "new_up_block expected to be AttnUpBlock2D"

            # copy resnets and upsamplers
            new_up_block.resnets = copy_weights(
                old_up_block.resnets, new_up_block.resnets
            )
            new_up_block.upsamplers = copy_weights(
                old_up_block.upsamplers, new_up_block.upsamplers
            )

            # replace each Transformer2DModel block with Attention or Transformer2DSelfAttentionModel
            new_attentions = []
            for j in range(len(old_up_block.attentions)):
                new_up_block_attention = replace_transformer_attention(
                    old_up_block.attentions[j],
                    new_up_block.attentions[j],
                    crossattn_replacement,
                    crossattn_bias_replacement=crossattn_bias_replacement,
                )

                new_attentions.append(new_up_block_attention)

            new_up_block.attentions = torch.nn.ModuleList(new_attentions)

            unet_uncond.up_blocks[i] = new_up_block

        elif up_block_type in ["UpBlock2D", "AttnUpBlock2D", "SelfAttnUpBlock2D"]:
            # direct replacement
            unet_uncond.up_blocks[i] = copy_weights(
                old_up_block, unet_uncond.up_blocks[i]
            )
        else:
            raise ValueError(f"Unsupported up_block_type: {up_block_type}")

    unet_uncond.conv_norm_out = copy_weights(
        unet.conv_norm_out, unet_uncond.conv_norm_out
    )
    unet_uncond.conv_act = copy_weights(unet.conv_act, unet_uncond.conv_act)
    unet_uncond.conv_out = copy_weights(unet.conv_out, unet_uncond.conv_out)

    return unet_uncond


def save_unconditional_model_card(
    args: dict,
    repo_id: str,
    images=None,
    repo_folder=None,
    dataset_str: str = None,
    for_training: bool = False,
):
    """
    Saving huggingface model card when creating an unconditional model from a conditional model.
    """

    pretrained_model = args.get("pretrained_model_name_or_path", "N/A")
    dataset = args.get("dataset", "N/A")
    if isinstance(dataset, list):
        dataset = "-".join([d.value for d in dataset])

    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, min(len(images), 8))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    if dataset_str is not None:
        dataset_str_full = f"on the **{dataset}** dataset(s) "
    else:
        dataset_str_full = ""

    yaml = f"""
    ---
    license: creativeml-openrail-m
    base_model: {pretrained_model}
    tags:
    - stable-diffusion
    - stable-diffusion-diffusers
    - diffusers
    inference: true
    ---
    """.replace(
        "    ", ""
    )

    model_card = f"""
    # Unconditioned stable diffusion finetuning - {repo_id}

    This pipeline was finetuned from **{pretrained_model}**
    {dataset_str_full}for brain image generation.
    Below are some example images generated with the finetuned pipeline: \n
    {img_str}

    ## Pipeline usage

    You can use the pipeline like so:

    ```python
    from diffusers import StableDiffusionUnconditionalPipeline
    import torch

    pipeline = StableDiffusionUnconditionalPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float32)
    image = pipeline(1).images[0]
    image.save("brain_image.png")
    ```

    ## Training info
    """.replace(
        "    ", ""
    )

    # Training info.
    training_info = f"For training info, refer the model card for the parent conditional model: {pretrained_model}."
    vae_type = "MEDVAE" if args["custom_vae_path"] else "SD VAE"
    if for_training:
        training_info = f"""These are the key hyperparameters used during training:

        * Epochs: {args.get("num_train_epochs", "N/A")}
        * Max Train Steps: {args.get("max_train_steps", "N/A")}
        * Learning rate: {args.get("learning_rate", "N/A")}
        * Batch size: {args.get("train_batch_size", "N/A")}
        * VAE scaling: {args.get("vae_scaling_factor", "N/A")}
        * VAE type: {vae_type}
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


def unconditional_collate_fn(examples):
    """
    Stack the pixel values and tokenize the captions and return as a dict
    for torch.dataloader collation.
    """

    pixel_values = torch.stack([example[IMAGE_KEY] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    quality = torch.tensor([example[QUALITY_KEY] for example in examples])

    return {COLLATE_IMAGE_KEY: pixel_values, QUALITY_KEY: quality}
