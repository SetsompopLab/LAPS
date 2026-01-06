import loguru
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from medvae.models import AutoencoderKL_2D
from medvae.utils.factory import FILE_DICT_ASSOCIATIONS, download_model_weights
from omegaconf import OmegaConf

from laps.model.med_vae_wrapper import MedVAEWrapper

__all__ = ["load_scheduler", "load_kl_vae", "load_medvae"]


def load_scheduler(
    pretrained_model_name_or_path: str,
    scheduler_type: str = "ddpm",
    rescale_betas_zero_snr: bool = False,
    prediction_type: str = None,
    for_training: bool = False,
    logger=loguru.logger,
) -> DDPMScheduler:

    assert scheduler_type in ["ddpm", "ddim"]
    if scheduler_type == "ddpm":
        sfunc = DDPMScheduler
    elif scheduler_type == "ddim":
        sfunc = DDIMScheduler
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    noise_scheduler = sfunc.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="scheduler",
        rescale_betas_zero_snr=rescale_betas_zero_snr,
    )

    logger.info(
        f"Loaded scheduler with prediction_type={noise_scheduler.config.prediction_type}."
    )

    if (prediction_type is not None) and (
        not (prediction_type == noise_scheduler.config.prediction_type)
    ):
        assert (
            for_training
        ), "prediction_type can only be set during training. otherwise must load default of scheduler."

    # rescaling only can be used for for v_prediction
    if rescale_betas_zero_snr:
        logger.info("Rescaling betas to reach terminal SNR=0.0.")
        if prediction_type is None:
            assert noise_scheduler.config.prediction_type == "v_prediction"
        else:
            assert prediction_type == "v_prediction"
    else:
        logger.info(
            f"Loaded Scheduler with non-zero Terminal beta = {noise_scheduler.config.beta_start}"
        )

    # Get the target for loss depending on the prediction type
    if prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=prediction_type)

    return noise_scheduler


def load_kl_vae(
    pretrained_model_name_or_path: str,
    revision: str = None,
    variant: str = None,
    vae_scaling_factor: float = None,
    requires_grad: bool = False,
    verbose: bool = False,
    logger=loguru.logger,
) -> AutoencoderKL:

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        revision=revision,
        variant=variant,
        subfolder="vae",
    )
    orig_scale_factor = vae.config.scaling_factor
    if vae_scaling_factor is not None:
        vae.config["scaling_factor"] = vae_scaling_factor
        vae.config.scaling_factor = vae_scaling_factor

    vae.requires_grad_(requires_grad)

    if verbose:

        scale_factor = vae.config.scaling_factor

        if orig_scale_factor != scale_factor:
            logger.info(
                f"Loaded VAE with scaling factor: {scale_factor} (original: {orig_scale_factor})"
            )
        else:
            logger.info(f"Loaded VAE with scaling factor: {scale_factor}")

    return vae


def load_medvae(
    ckpt_fpath: str,
    downsampling_factor: int,
    vae_scaling_factor: float | None = None,
    logger=loguru.logger,
    **kwargs,
):
    logger.info(f"Loading MedVAE from {ckpt_fpath}.")
    logger.info(f"Downsampling factor: {downsampling_factor}.")
    if downsampling_factor == 4:
        config_fpath = download_model_weights(
            FILE_DICT_ASSOCIATIONS["medvae_4_4_2d_c"]["config"]
        )
    else:
        raise ValueError(
            f"Unsupported downsampling factor: {downsampling_factor}. Supported values are 4."
        )
    conf = OmegaConf.load(config_fpath)
    conf["ddconfig"]["in_channels"] = 2
    conf["ddconfig"]["out_ch"] = 2
    conf.embed_dim = 4
    conf.ddconfig.z_channels = 4

    vae = AutoencoderKL_2D(
        ddconfig=conf.ddconfig,
        embed_dim=conf.embed_dim,
    )

    accelerator = Accelerator(mixed_precision="no")
    vae = accelerator.prepare(vae)

    accelerator.load_state(ckpt_fpath)

    vae.eval()

    scaling_factor = vae_scaling_factor or 1.0

    vae = accelerator.unwrap_model(vae)

    return MedVAEWrapper(
        vae, scaling_factor=scaling_factor, downsampling_factor=downsampling_factor
    )
