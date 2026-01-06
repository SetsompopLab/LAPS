import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import torch
import tyro
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration
from diffusers import (
    UNet2DModel,
)
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from transformers import CLIPTextModel
from transformers.utils import ContextManagers

from laps.configs.sd_training import TrainingArgs
from laps.model import (
    load_kl_vae,
    load_medvae,
    load_scheduler,
    save_conditional_model_card,
    save_unconditional_model_card,
)
from laps.train_sd import validate_args, validation_sample

logger = get_logger(__name__, log_level="INFO")


def main(args: TrainingArgs):
    output_dir = Path("./tmp_sd_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    logging_dir = args.output_dir / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Validate args
    args = validate_args(args)

    repo_id = create_repo(
        repo_id=args.hub_model_id,
        exist_ok=True,
        private=True,
    ).repo_id

    # initialize model parts
    noise_scheduler = load_scheduler(
        args.pretrained_model_name_or_path,
        scheduler_type="ddpm",
        rescale_betas_zero_snr=not args.dont_rescale_snr_to_zero,
        prediction_type=args.prediction_type,
        for_training=True,
        logger=logger,
    )

    # enforce prediction_type to be consistent
    assert (
        args.prediction_type == noise_scheduler.config.prediction_type
    ), f"Prediction type mismatch: {args.prediction_type} != {noise_scheduler.config.prediction_type}"

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin
            if accelerate.state.is_initialized()
            else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        if args.use_conditional:
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="text_encoder",
                revision=args.model_revision,
                variant=args.model_variant,
            )

        if args.custom_vae_path is not None:
            vae = load_medvae(
                args.custom_vae_path,
                vae_scaling_factor=args.vae_scaling_factor,
                revision=args.model_revision,
                variant=args.model_variant,
                downsampling_factor=args.downsampling_factor,
                requires_grad=False,
                verbose=True,
                logger=logger,
            )
        else:
            vae = load_kl_vae(
                args.pretrained_model_name_or_path,
                vae_scaling_factor=args.vae_scaling_factor,
                revision=args.model_revision,
                variant=args.model_variant,
                requires_grad=False,
                verbose=True,
                logger=logger,
            )

    unet_class = UNet2DModel
    tokenizer = None
    text_encoder = None

    unet = unet_class.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision,
    )
    unet.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = unet_class.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.model_revision,
            variant=args.model_variant,
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=unet_class,
            model_config=ema_unet.config,
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    if isinstance(model, CLIPTextModel):
                        model.save_pretrained(os.path.join(output_dir, "text_encoder"))
                    else:
                        model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), unet_class
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, CLIPTextModel):
                    load_model = CLIPTextModel.from_pretrained(
                        input_dir, subfolder="text_encoder"
                    )
                else:
                    load_model = unet_class.from_pretrained(input_dir, subfolder="unet")

                    model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    optim_params = unet.parameters()

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    unet, optimizer = accelerator.prepare(unet, optimizer)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    path = args.resume_from_checkpoint
    accelerator.print(f"Resuming from checkpoint {path}")
    accelerator.load_state(path)

    unet = unwrap_model(unet)
    if args.use_ema:
        ema_unet.copy_to(unet.parameters())

    weight_dtype = torch.float32
    images = validation_sample(
        vae=vae,
        unet=unet,
        args=args,
        accelerator=accelerator,
        weight_dtype=weight_dtype,
        res=(args.resolution, args.resolution),
        use_generator=args.seed is not None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        save_pipeline=True,
        unwrap_vae_unet=False,
    )
    images = [Image.fromarray(image) for image in images]

    if args.push_to_hub:

        if args.use_conditional:
            save_model_func = save_conditional_model_card
        else:
            save_model_func = save_unconditional_model_card

        save_model_func(
            vars(args),
            repo_id,
            images,
            repo_folder=args.output_dir,
            for_training=True,
        )

        wrapper_src = (
            Path(__file__).parents[2] / "src" / "laps" / "model" / "med_vae_wrapper.py"
        )
        wrapper_dst = Path(args.output_dir) / "vae" / "laps.model.med_vae_wrapper.py"
        wrapper_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(wrapper_src, wrapper_dst)

        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )


if __name__ == "__main__":
    main(tyro.cli(TrainingArgs))
