import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version

from ...configuration_utils import FrozenDict
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, LoraLoaderMixin
from ...models import AutoencoderKL, UNet2DModel
from ...models.attention_processor import FusedAttnProcessor2_0
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .data_consistency_utils import (
    dc_image_optimization,
    dc_latent_manifold_optimization,
    dc_latent_optimization,
    dc_plds,
    debug_encoder_plot,
    debug_latent_list_plot,
    debug_latent_plot,
    debug_plot,
    ddim_inversion,
    estimate_zt_from_z0_hat_for_ldps,
    predict_z0_hat,
    SimplePlateauLrScheduler,
    stochastic_resampling,
    ksp_loss,
    cplx_encode,
    cplx_decode,
)
from ...schedulers.scheduling_ddim import DDIMScheduler
from .pipeline_output import StableDiffusionPipelineOutput
from .pipeline_stable_diffusion import (
    EXAMPLE_DOC_STRING,
    retrieve_timesteps,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class StableDiffusionUnconditionalPipeline(
    DiffusionPipeline, LoraLoaderMixin, FromSingleFileMixin
):
    r"""
    Pipeline for unconditional image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    model_cpu_offload_seq = "unet->vae"
    _optional_components = []
    _exclude_from_cpu_offload = []
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DModel,
        scheduler: KarrasDiffusionSchedulers,
        requires_safety_checker: bool = False,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (n) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to n in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        height,
        width,
        callback_steps,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, device, generator, dtype=None, latents=None):
        
        if dtype is None:
            dtype = torch.float32 #self.dtype #TODO: check self.dtype

        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.fuse_qkv_projections
    def fuse_qkv_projections(self, unet: bool = True, vae: bool = True):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query,
        key, value) are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is experimental.

        </Tip>

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.
        """
        self.fusing_unet = False
        self.fusing_vae = False

        if unet:
            self.fusing_unet = True
            self.unet.fuse_qkv_projections()
            self.unet.set_attn_processor(FusedAttnProcessor2_0())

        if vae:
            if not isinstance(self.vae, AutoencoderKL):
                raise ValueError("`fuse_qkv_projections()` is only supported for the VAE of type `AutoencoderKL`.")

            self.fusing_vae = True
            self.vae.fuse_qkv_projections()
            self.vae.set_attn_processor(FusedAttnProcessor2_0())

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.unfuse_qkv_projections
    def unfuse_qkv_projections(self, unet: bool = True, vae: bool = True):
        """Disable QKV projection fusion if enabled.

        <Tip warning={true}>

        This API is experimental.

        </Tip>

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.

        """
        if unet:
            if not self.fusing_unet:
                logger.warning("The UNet was not initially fused for QKV projections. Doing nothing.")
            else:
                self.unet.unfuse_qkv_projections()
                self.fusing_unet = False

        if vae:
            if not self.fusing_vae:
                logger.warning("The VAE was not initially fused for QKV projections. Doing nothing.")
            else:
                self.vae.unfuse_qkv_projections()
                self.fusing_vae = False

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return 1.0 # CFG becomes irrelevant in unconditional version

    @property
    def guidance_rescale(self):
        return 1.0 # rescaling becomes irrelevant in unconditional version

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    @property
    def do_classifier_free_guidance(self):
        return False

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        batch_size: int,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        num_images_per_query: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        do_denormalize: Optional[bool] = True, # Will re-scale images from [-1, 1] to [0,1]
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`): The batch size for the generation.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.            
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            height,
            width,
            callback_steps,
            callback_on_step_end_tensor_inputs,
        )

        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        device = self._execution_device

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        do_dc = False
        measurements = kwargs.get("measurements", None)
        if measurements is not None:
            do_dc = True
            start_with_prior = kwargs.get("start_with_prior", False)
            prior_start_timestep = kwargs.get("prior_start_timestep", timesteps[0])
            priors = kwargs.get("priors", None)
            if start_with_prior and priors is None:
                raise ValueError(
                    "If `start_with_prior` is `True`, you need to provide `priors` as a tensor of shape"
                    " `(batch_size, num_channels_latents, height, width)`."
                )
            if start_with_prior:
                # update timesteps to start from the closest timestep to init_with_prior_start_timestep
                timepoint_ind = torch.argmin(torch.abs(timesteps - prior_start_timestep))
                if timesteps[timepoint_ind] < prior_start_timestep and timepoint_ind > 0:
                    timepoint_ind -= 1
                timesteps = timesteps[timepoint_ind:]
                num_inference_steps = len(timesteps)
                prior_start_timestep = timesteps[0]

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_query,
            num_channels_latents,
            height,
            width,
            device,
            generator,
            dtype=torch.float32, # FIXME: figure out better way to set
            latents=latents,
        )

        if do_dc and start_with_prior:
            # encode priors in latent space:
            prior_latents = self.vae.encode(priors).latent_dist.mode() * self.vae.config.scaling_factor

            if isinstance(self.scheduler, DDIMScheduler) and kwargs.get("ddim_inversion", False):
                # logger.info(f"Running DDIM Inversion for initialization")
                latents = ddim_inversion(
                    self.scheduler,
                    self.unet,
                    prior_latents,
                    timesteps.clone(),
                    self.cross_attention_kwargs,
                )
            else:
                latents = self.scheduler.add_noise(prior_latents, latents, prior_start_timestep)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.2 FIXME: Implement optional Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            raise Warning("Time conditional projection is not yet verified in unconditional version.")
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_query)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        #### Added code for DC ####
        if measurements is not None:
            # FIXME: maybe set one of these to all timesteps by default
            image_dc_timesteps = kwargs.pop("image_dc_timesteps")
            latent_dc_timesteps = kwargs.pop("latent_dc_timesteps")
            forward_model = kwargs["forward_model"]
            opt_params = kwargs["opt_params"]
            dc_type = kwargs["dc_type"]
            gamma = kwargs["gamma"]
            dc_caching = kwargs.get("dc_caching", False)
            # flag to indicate we have a cached z_o_opt, and we can start using it. Turns True after first DC step.
            use_dc_caching = False
            debug_dc = kwargs.get("debug_dc", False)
            gt_latents = kwargs.get("gt_latents", None)
            output_dc = kwargs.get("output_dc", False)
            latent_dc_cg_init = kwargs.get("latent_dc_cg_init", False)
            forward_scales = kwargs.get("forward_scales", None)
        #### End of added code for DC ####

        # create learning scaling scheduler
        if do_dc:
            LR_Scaler = SimplePlateauLrScheduler(initial_lr_scale=1)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # shift timesteps by 1
        timesteps_prev = timesteps[1:]
        timesteps_prev = torch.concatenate([timesteps_prev, timesteps_prev[-1:] * 0], dim=0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                ).sample

                #### Added code for DC ####
                if do_dc and dc_type == "hard":
                    if (t in image_dc_timesteps) or (t in latent_dc_timesteps):
                        # estimate z_0
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        beta_prod_t = 1 - alpha_prod_t
                        z_0_hat = predict_z0_hat(
                            self.scheduler.config.prediction_type,
                            noise_pred,
                            latents,
                            beta_prod_t,
                            alpha_prod_t,
                        )

                        # run DC
                        if not debug_dc:
                            if t in latent_dc_timesteps:

                                # add the option to initialize the latent optimization with image DC solution
                                if latent_dc_cg_init:
                                    z_0_hat = dc_image_optimization(
                                        forward_model,
                                        self.vae,
                                        opt_params["image"],
                                        measurements,
                                        z_0_hat,
                                        encode_output=True,
                                        generator=generator,
                                    )

                                z_0_opt, forward_scales = dc_latent_optimization(forward_model,
                                                                                self.vae,
                                                                                opt_params["latent"],
                                                                                measurements,
                                                                                z_0_hat,
                                                                                forward_scales=forward_scales,
                                                                                generator=generator)
                            else:
                                z_0_opt = dc_image_optimization(
                                    forward_model,
                                    self.vae,
                                    opt_params["image"],
                                    measurements,
                                    z_0_hat,
                                    encode_output=True,
                                    generator=generator,
                                )
                        else:
                            if gt_latents is not None:
                                z_0_opt = gt_latents
                            else:
                                raise ValueError("gt_latents must be provided for debug_dc")

                        if dc_caching:
                            use_dc_caching = True
                            cached_z_0_opt = z_0_opt

                if do_dc and dc_type == "manifold_ldps":
                    if (t in latent_dc_timesteps):
                        # refine latents to be consistent with the measurements
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        beta_prod_t = 1 - alpha_prod_t
                        latents, noise_pred, loss, forward_scales = dc_latent_manifold_optimization(
                            forward_model,
                            self.vae,
                            self.unet,
                            opt_params["latent"],
                            measurements,
                            latents,
                            t,
                            alpha_prod_t,
                            beta_prod_t,
                            self.scheduler.config.prediction_type,
                            lr_scale=LR_Scaler.get_lr_scale(),
                            forward_scales=forward_scales,
                            generator=generator,
                            unet_args = dict(
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                            )
                        )

                        # update learning rate only on last 200 steps
                        if t <= 200:
                            LR_Scaler.step(loss)

                #### End of added code for DC ####

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                #### Added code for DC ####
                if do_dc and dc_type == "hard":
                    if (t in image_dc_timesteps) or (t in latent_dc_timesteps) or use_dc_caching:
                        # take the optimized z_0_hat or the cached one.
                        z_0_opt = z_0_opt if ((t in image_dc_timesteps) or (t in latent_dc_timesteps)) else cached_z_0_opt

                        # map back to z_{t-1}
                        prev_timestep = 0 if (i + 1) > (len(timesteps) - 1) else timesteps[i + 1]
                        prev_prev_timestep = prev_timestep - 1

                        latents = stochastic_resampling(
                            z_0_opt, latents, self.scheduler, prev_timestep, prev_prev_timestep, gamma, simple=False
                        )

                if do_dc and dc_type == "ldps":
                    if t in latent_dc_timesteps:
                        # estimate z_0 based on new latents
                        prev_timestep = 0 if (i + 1) > (len(timesteps) - 1) else timesteps[i + 1]
                        alpha_prod_t = self.scheduler.alphas_cumprod[prev_timestep]
                        beta_prod_t = 1 - alpha_prod_t
                        z_0_hat = predict_z0_hat(
                            self.scheduler.config.prediction_type,
                            noise_pred,
                            latents,
                            beta_prod_t,
                            alpha_prod_t,
                        )

                        z_0_opt, forward_scales, = dc_latent_optimization(forward_model,
                                                                        self.vae,
                                                                        opt_params["latent"],
                                                                        measurements,
                                                                        z_0_hat,
                                                                        forward_scales=forward_scales,
                                                                        generator=generator)

                        # general z_t estimate for any scheduler predction type
                        latents = estimate_zt_from_z0_hat_for_ldps(
                            self.scheduler.config.prediction_type,
                            noise_pred,
                            z_0_opt,
                            beta_prod_t,
                            alpha_prod_t,
                            z_t_orig = latents, # only used for sample prediction
                        )
                     
                if do_dc and dc_type == "plds":
                    # Manifold LDPS but after the scheduler step
                    if (t in latent_dc_timesteps):
                        # refine latents to be consistent with the measurements
                        t_prev = timesteps_prev[i]
                        alpha_prod_t = self.scheduler.alphas_cumprod[t_prev]
                        beta_prod_t = 1 - alpha_prod_t
                        latents, loss = dc_plds(
                            forward_model,
                            self.vae,
                            self.unet,
                            opt_params["latent"],
                            measurements,
                            latents,
                            t_prev,
                            alpha_prod_t,
                            beta_prod_t,
                            self.scheduler.config.prediction_type,
                            lr_scale=LR_Scaler.get_lr_scale(),
                            generator=generator,
                            unet_args = dict(
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                            )
                        )

                        # Don't scale learning rate for truly PLDS method
                        if opt_params["latent"].get("n_iters", 1) > 1:
                            if t <= 200:
                                LR_Scaler.step(loss)

                #### End of added code for DC ####

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                    
                    if do_dc and dc_type in ["plds", "manifold_ldps"]:
                        # update desc of pbar with loss
                        progress_bar.set_description(f"DC Loss: {loss.item():.4f}")

        # Output DC
        output_dc = False # TODO: removeme? 
        final_loss = None
        if not output_type == "latent":
            #### Added code for DC ####
            # Do a final DC step
            if do_dc and output_dc:
                image = dc_image_optimization(
                                forward_model,
                                self.vae,
                                opt_params["image"],
                                measurements,
                                latents,
                                encode_output=False,
                                generator=generator,
                )
            #### End of added code for DC ####
            else:
                image = self.vae.decode(
                    latents / self.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator
                )[0]
                if do_dc:
                    with torch.no_grad():
                        im_cplx = image[:, 0] + 1j * image[:, 1]
                        y_hat = forward_model(im_cplx)

                        s = torch.linalg.vecdot(
                            y_hat.flatten(1), measurements.flatten(1)
                        ) / torch.linalg.vecdot(y_hat.flatten(1), y_hat.flatten(1))
                        s = s[:, None, None]
                        im_cplx = im_cplx * s
                        image = torch.stack(
                            [im_cplx.real, im_cplx.imag], dim=1
                        )

                        final_loss = ksp_loss(y_hat * s[:, None], measurements).squeeze()
                        final_loss = final_loss.cpu()
        else:
            #### Added code for DC ####
            # Do a final DC step
            if do_dc and output_dc:
                latents = dc_image_optimization(
                                forward_model,
                                self.vae,
                                opt_params["image"],
                                measurements,
                                latents,
                                encode_output=True,
                                generator=generator,
                )
            #### End of added code for DC ####
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=[do_denormalize] * image.shape[0])

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None, final_loss=final_loss)
