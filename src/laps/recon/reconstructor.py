import gc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.data_consistency_utils import (
    batched_conjugate_gradient,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_unconditional import (
    StableDiffusionUnconditionalPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import rearrange
from loguru import logger

from laps.model.load import load_medvae
from laps.model.med_vae_wrapper import MedVAEWrapper
from laps.modl.models import Modl
from laps.recon.algs import conjugate_gradient
from laps.recon.classic_recons import LACS_fista_recon
from laps.recon.linops import CartesianSenseLinop, linop
from laps.utils import certainty_weight_map, clear_cache, ensure_torch, get_torch_device


@dataclass
class ReconstructorOutput:
    recon: torch.Tensor
    error: Optional[torch.Tensor] = None
    extra_outputs: Optional[Dict[str, Any]] = None


@dataclass
class ReconParams:
    device: Union[str, int, torch.device] = "cpu"
    im_size: Tuple[int, int] = (256, 256)
    debug: Optional[bool] = None


class Reconstructor:
    def __init__(self, forward_model: linop, params: ReconParams = ReconParams()):
        """
        Base class for all reconstructors.

        Args:
            forward_model (linop): The forward model used for reconstruction.
                Has a defined forward, adjoint, and normal operator.
            params (ReconParams): Parameters for the reconstructor.
        """
        self.forward_model = forward_model
        self.params = params
        self.device = params.device
        self.im_size = params.im_size
        self.debug = params.debug

    def reconstruct(self, measurements, **kwargs) -> ReconstructorOutput:
        """
        Reconstruct the image from the measurements using the forward model.

        Args:
            measurements (torch.Tensor): The measurements to reconstruct from.
                Will be given in shape (B, Nc, *im_size)
            **kwargs: Additional arguments for the reconstruction process.

        Returns:
            ReconstructorOutput: The reconstructed image and optional error estimate.
        """
        raise NotImplementedError("Reconstruction method not implemented.")


# ------------------------------------------- Stable diffusion reconstructor ----------------------------------------- #
@dataclass
class AutoTpConfig:
    """
    Automatic Tp selection
    """

    prior_recon_config: ReconParams = field(default_factory=lambda: ReconParams())
    reg_scale: float = 0.5
    reg_shift: float = 0.0
    clip_min: int = 100
    clip_max: int = 500


@dataclass
class LDMParams(ReconParams):
    """
    Base params for unconditional LDM model
    """

    model_name_or_path: str = "yurman/uncond-sd2-base-complex"  # 2 channel
    is_medvae: bool = False
    num_inference_steps: int = 100
    dc_type: str = "manifold_ldps"  # hard/ldps
    opt_params: dict = field(
        default_factory=lambda: {
            "latent": {
                "n_iters": 10,
                "lr": 1e-2,
                "threshold": 1,
                "latent_consistency": 0,
                "z_t_lam": 0.0,  # add z_t_lam * ||z - z_t||^2 to loss
            },
            "image": {
                "n_iters": 10,
                "threshold": 1e-5,
                "lambda_l2": 1e-3,
            },
        }
    )
    gamma: float = 200.0
    scheduler_ty: Optional[str] = "DDIM"
    n_avgs: int = 1
    dc_latent_steps: Optional[List[Tuple[float, float, int]]] = field(
        default_factory=lambda: [
            (0, 1, 1),
        ],
    )
    dc_image_steps: Optional[List[Tuple[float, float, int]]] = field(
        default_factory=lambda: [
            (0.33, 0.66, 3),
        ],
    )

    # do resampling in between DC steps with the latest z_0_hat
    dc_caching: bool = False
    start_with_prior: bool = True
    start_with_cg: bool = (
        False  # Optionally use CG recon for prior initialization instead
    )
    prior_start_timestep: Union[int, str] = "auto"  # can be an int or "auto"
    ddim_inversion: bool = False

    # auto TP parameters
    auto_tp_config: Optional[AutoTpConfig] = field(
        default_factory=lambda: AutoTpConfig()
    )

    # additional DC params
    output_dc: bool = False
    latent_dc_cg_init: bool = False

    output_dc_config: dict = field(
        default_factory=lambda: {
            "n_iters": 5,
            "threshold": 1e-5,
            "lambda_l2": 1e-4,
            "lambda_ldm": 0.0,
            "lambda_l2_from_data": True,
        }
    )


@dataclass
class MedVaeLDMParams(LDMParams):
    prediction_type: str = "v_prediction"
    is_medvae: bool = True


@dataclass
class TextConditionalLDMParams(LDMParams):
    prompt: str = "An empty, flat black image with a MRI brain axial scan in the center"
    classifier_free_guidance: float = 1  # <= 1 for no guidance


class DiffusersReconstructor(Reconstructor):
    """
    Generic class for all diffusers reconstructors which use pipeline models
    """

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        forward_model: CartesianSenseLinop,
        params: LDMParams = LDMParams(),
    ):
        self.device = get_torch_device(params.device)
        self.im_size = params.im_size
        self.num_inference_steps = params.num_inference_steps
        self.n_avgs = params.n_avgs
        self.pipeline = pipeline
        self.n_channel = self.pipeline.vae.config.in_channels

        if params.scheduler_ty is not None:
            if params.scheduler_ty == "DDIM":
                self.pipeline.scheduler = DDIMScheduler.from_config(
                    self.pipeline.scheduler.config
                )
            elif params.scheduler_ty == "DDPM":
                self.pipeline.scheduler = DDPMScheduler.from_config(
                    self.pipeline.scheduler.config
                )
            else:
                raise ValueError(f"Scheduler type {params.scheduler_ty} not supported")
        else:
            logger.warning("No scheduler type provided, using default scheduler")

        timesteps, n_timesteps = retrieve_timesteps(
            self.pipeline.scheduler,
            params.num_inference_steps,
            device=torch.device("cpu"),
        )
        if n_timesteps is None:
            raise ValueError(
                "failed to retrieve timesteps from pipeline and diffusers parameters."
            )
        timesteps = ensure_torch(timesteps)

        dc_args = {}
        dc_args["image_dc_timesteps"] = []
        if params.dc_image_steps is not None:
            for range_ in params.dc_image_steps:
                image_dc_end, image_dc_start = n_timesteps - int(
                    range_[0] * n_timesteps
                ), n_timesteps - int(range_[1] * n_timesteps)
                dc_args["image_dc_timesteps"].extend(
                    list(timesteps[image_dc_start : image_dc_end : range_[2]].numpy())
                )

        dc_args["latent_dc_timesteps"] = []
        if params.dc_latent_steps is not None:
            for range_ in params.dc_latent_steps:
                latent_dc_end, latent_dc_start = n_timesteps - int(
                    range_[0] * n_timesteps
                ), n_timesteps - int(range_[1] * n_timesteps)
                dc_args["latent_dc_timesteps"].extend(
                    list(timesteps[latent_dc_start : latent_dc_end : range_[2]].numpy())
                )
        dc_args["opt_params"] = params.opt_params
        dc_args["gamma"] = params.gamma
        dc_args["forward_model"] = forward_model
        dc_args["dc_type"] = params.dc_type
        dc_args["dc_caching"] = params.dc_caching
        dc_args["output_dc"] = params.output_dc
        dc_args["latent_dc_cg_init"] = params.latent_dc_cg_init
        dc_args["start_with_prior"] = params.start_with_prior
        dc_args["start_with_cg"] = params.start_with_cg
        dc_args["prior_start_timestep"] = params.prior_start_timestep
        dc_args["output_dc_config"] = params.output_dc_config
        dc_args["ddim_inversion"] = params.ddim_inversion
        self.dc_args = dc_args

        # set up auto-tp reconstructor
        self.auto_tp = dc_args["start_with_prior"] and (
            self.params.prior_start_timestep == "auto"
        )
        # self.alpha_bars = None
        self.auto_tp_config = params.auto_tp_config
        self.auto_tp_abar = self.pipeline.scheduler.alphas_cumprod.cpu()
        self.auto_tp_tt = torch.arange(1000).cpu()
        if self.auto_tp:
            # compute prior start timestep from reconstruction
            auto_tp_recon_config = params.auto_tp_config.prior_recon_config
            auto_tp_recon_config.device = self.device
            auto_tp_recon_config.im_size = self.im_size
            # had to duplicate here unfortunately
            if isinstance(auto_tp_recon_config, CGParams):
                auto_tp_reconstructor = CGReconstructor
                auto_tp_reconstructor_name = "CG"
            elif isinstance(auto_tp_recon_config, LACSParams):
                auto_tp_reconstructor = LACSReconstructor
                auto_tp_reconstructor_name = "CS"
            elif isinstance(auto_tp_recon_config, ModlParams):
                auto_tp_reconstructor = MoDLReconstructor
                auto_tp_reconstructor_name = "MoDL"
                kwargs = dict(load_model_to_cpu=True)
            elif isinstance(auto_tp_recon_config, LDMParams):
                auto_tp_reconstructor = StableDiffusionReconstructor
                if auto_tp_recon_config.start_with_cg:
                    auto_tp_reconstructor_name = "CAPS"
                elif auto_tp_recon_config.start_with_prior:
                    auto_tp_reconstructor_name = "LAPS"
                else:
                    auto_tp_reconstructor_name = "LDM"
                assert (
                    auto_tp_recon_config.prior_start_timestep != "auto"
                ), "Recursive auto-tp reconstructor is not supported, "
                kwargs = dict(load_model_to_cpu=True)
            else:
                raise ValueError(
                    f"Unsupported reconstructor type for auto-tp: {type(auto_tp_recon_config)}"
                )
            logger.info(
                f"Using {auto_tp_reconstructor_name} reconstructor for auto-tp prior initialization"
            )
            self.auto_tp_reconstructor = auto_tp_reconstructor(
                forward_model=forward_model,
                params=auto_tp_recon_config,
                **kwargs,
            )
        elif dc_args["start_with_prior"]:
            assert isinstance(
                self.params.prior_start_timestep, (int, float)
            ), "Prior start timestep must be 'auto' or integer"

        self.params = params
        self.forward_model = forward_model
        super().__init__(
            forward_model=forward_model,
            params=params,
        )

    def _cplx2vae(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input (B, *im_size) complex -> (B, C, *im_size) float
        """
        # convert complex -> channeled [-1, 1]
        scale = torch.amax(x.abs(), dim=(-1, -2), keepdim=True)
        x /= scale

        out = torch.stack([x.real, x.imag, x.abs()], dim=1)

        if self.n_channel == 2:
            out = out[:, :2]

        return out

    def _vae2cplx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input (B, C, *im_size) float -> (B, *im_size) complex
        """
        # convert channeled [-1, 1] -> complex
        out = x[:, 0] + x[:, 1] * 1j
        if self.n_channel == 3:
            out = ((out.abs() + x[:, 2]) / 2) * torch.exp(1j * out.angle())
        return out

    def _encode(self, x: torch.Tensor, add_averages=False) -> torch.Tensor:
        """
        Input (B, *im_size) complex -> (B, C_latent, H_latent, W_latent) float
        """
        x = self._cplx2vae(x)
        if add_averages:
            x = x.repeat_interleave(self.n_avgs, dim=0)
        x = self.pipeline.vae.encode(x).latent_dist.mode()
        x = x * self.pipeline.vae.config.scaling_factor
        return x

    def batch_normalize(self, shifted, target):
        scale = torch.linalg.vecdot(
            shifted.flatten(1), target.flatten(1)
        ) / torch.linalg.vecdot(shifted.flatten(1), shifted.flatten(1))
        num_dims = shifted.ndim - 1
        scale = scale.view(-1, *([1] * num_dims))
        return shifted * scale

    def get_auto_tp(self, x_prior, x_init):

        abar = self.auto_tp_abar.clone()
        abar = abar.to(self.device)
        tt = self.auto_tp_tt.clone()
        tt = tt.to(self.device)

        zp = self._encode(x_prior)
        z_init = self._encode(x_init)

        # normalize z_init to zp
        z_init = self.batch_normalize(z_init, zp)

        # Compute function at each timestep t
        gamma_t = 0.5 * torch.log((2 - abar) / (2 * (1 - abar)))
        t1 = torch.linalg.norm(zp) / (2 - abar)
        t2 = torch.linalg.norm(zp - z_init) / (2 * (1 - abar))
        tps = gamma_t + (abar / 2) * (t1 - t2)
        tp_opt = tt[torch.argmax(tps)]

        # adjust scaling/shift
        tp_opt = tp_opt * self.auto_tp_config.reg_scale + self.auto_tp_config.reg_shift

        # clip
        tp_opt_clipped = int(
            torch.clip(
                tp_opt,
                min=self.auto_tp_config.clip_min,
                max=self.auto_tp_config.clip_max,
            ).item()
        )
        tp_opt = int(tp_opt.item())

        logger.info(f"Auto TP selected: {tp_opt_clipped} (unclipped = {tp_opt})")

        return tp_opt_clipped

    def _prepare_args(self, measurements, **kwargs):

        # prep dc args
        dc_args = self.dc_args.copy()
        extra_outputs = None

        # auto tp
        if self.auto_tp:
            self.auto_tp_reconstructor.forward_model = self.forward_model
            if isinstance(self.auto_tp_reconstructor, DiffusersReconstructor):
                self.auto_tp_reconstructor.dc_args["forward_model"] = self.forward_model
            init_recon = self.auto_tp_reconstructor.reconstruct(
                measurements, **kwargs
            ).recon
            clear_cache()

            # get and format prior
            priors = kwargs["priors"].abs() * torch.exp(1j * init_recon.angle())

            # compute auto tp
            self.pipeline.to(self.device)
            tp_auto = self.get_auto_tp(
                x_prior=priors.clone(),
                x_init=init_recon.clone(),
            )
            logger.info(f"Auto TP selected: {tp_auto}")
            extra_outputs = {"auto_tp": tp_auto}
            dc_args["prior_start_timestep"] = tp_auto
            dc_args["priors"] = self._cplx2vae(
                priors.repeat_interleave(self.n_avgs, dim=0)
            )
        elif self.dc_args["start_with_prior"] or self.dc_args["start_with_cg"]:
            cg_recon = conjugate_gradient(
                AHA=self.forward_model.normal,
                AHb=self.forward_model.adjoint(measurements),
                num_iters=25,
                tolerance=1e-10,
                lamda_l2=1e-3,
            )
            if self.dc_args["start_with_prior"]:
                priors = kwargs["priors"].abs() * torch.exp(1j * cg_recon.angle())
                priors = priors.repeat_interleave(self.n_avgs, dim=0)
                priors = self._cplx2vae(priors)
                dc_args["priors"] = priors

            elif self.dc_args["start_with_cg"]:
                # Prior is CG Recon for denoising case
                dc_args["start_with_prior"] = True  # for diffusers pipeline
                dc_args["priors"] = self._cplx2vae(cg_recon).repeat_interleave(
                    self.n_avgs, dim=0
                )

        # repeat measurements for averaging
        dc_args["n_measurements"] = measurements.shape[0]
        measurements_rep = measurements.repeat_interleave(self.n_avgs, dim=0)
        dc_args["measurements"] = measurements_rep

        if "debug" in kwargs and kwargs["debug"]:
            logger.info("Using GT latents during DC steps for debugging")
            dc_args["debug_dc"] = True
            self.pipeline.to(self.device)
            dc_args["gt_latents"] = self._encode(kwargs["x0"], add_averages=True)

        return dc_args, extra_outputs

    def _postprocess_outputs(self, x, measurements, dc_args, final_loss=None):
        # convert channeled [-1, 1] -> complex
        out = x[:, 0] + x[:, 1] * 1j

        # mask to area of sensitivy map support
        brain_mask = self.forward_model.get_brain_mask()
        out = out * brain_mask

        # output shape is (n_measurements * n_avgs, C, H, W)
        out = rearrange(
            out,
            "(n_meas n_avg) h w -> n_meas n_avg h w",
            n_meas=dc_args["n_measurements"],
            n_avg=self.n_avgs,
        )

        # error over recons without output dc
        recon_var = torch.std(out.abs(), dim=(1))
        recon_mean = torch.mean(out.abs(), dim=(1))

        rel_uncertainty_map = certainty_weight_map(
            recon_var,
            recon_mean,
            mask=brain_mask,
            method="inverse",
            epsilon=1e-5,
            smooth_sigma=0.1,
        )[1].abs()

        if dc_args["output_dc"]:
            out = rearrange(
                out,
                "n_meas n_avg h w -> (n_meas n_avg) h w",
            )
            if (
                final_loss is not None
                and dc_args["output_dc_config"]["lambda_l2_from_data"]
            ):
                l2_out = torch.sum((out.abs() ** 2).flatten(1), dim=1).cpu()
                l2_lambda = torch.mean(final_loss / (1e2 * l2_out + 1e-10))
            else:
                l2_lambda = dc_args["output_dc_config"]["lambda_l2"]

            if dc_args["output_dc_config"]["lambda_ldm"] is not None:
                lam = dc_args["output_dc_config"]["lambda_ldm"]
                AHb = self.forward_model.adjoint(measurements) + lam * out
                A = lambda x: self.forward_model.normal(x) + lam * x
            else:
                AHb = self.forward_model.adjoint(measurements)
                A = self.forward_model.normal

            logger.info("Adding final data consistency step to averaged images")
            out = batched_conjugate_gradient(
                A,
                AHb,
                x0=out,
                max_iterations=dc_args["output_dc_config"]["n_iters"],
                tolerance=dc_args["output_dc_config"]["threshold"],
                lambda_l2=l2_lambda,
            )

            out = rearrange(
                out,
                "(n_meas n_avg) h w -> n_meas n_avg h w",
                n_meas=dc_args["n_measurements"],
                n_avg=self.n_avgs,
            )

        # calculate mean recon.
        avg_recon = torch.mean(out.abs(), dim=(1)) * torch.exp(
            1j * out.angle().mean(dim=(1))
        )

        return avg_recon, rel_uncertainty_map


class StableDiffusionReconstructor(DiffusersReconstructor):
    def __init__(
        self,
        forward_model: CartesianSenseLinop,
        params: LDMParams = LDMParams(),
        load_model_to_cpu: bool = False,
    ):
        self.params = params
        self.forward_model = forward_model
        self.idle_device = "cpu" if load_model_to_cpu else params.device

        assert (
            self.params.im_size[0] == self.params.im_size[1] == 256
        ), "256x256 only in LDM!"

        # determine type of model
        self.is_conditional = False
        if isinstance(params, TextConditionalLDMParams):
            self.is_conditional = True

        if self.is_conditional:
            self.pipeline_class = StableDiffusionPipeline
            assert isinstance(
                self.params, TextConditionalLDMParams
            ), "TextConditionalLDMParams should be used for text conditional LDMs"
        else:
            self.pipeline_class = StableDiffusionUnconditionalPipeline

        # check if using medvae model or not
        if isinstance(params, MedVaeLDMParams):
            unet = UNet2DModel.from_pretrained(
                params.model_name_or_path,
                subfolder="unet",
                revision=None,
            )
            unet.eval()
            vae = MedVAEWrapper.from_pretrained(
                params.model_name_or_path, subfolder="vae"
            )
            vae.eval()
            pipeline = StableDiffusionUnconditionalPipeline.from_pretrained(
                params.model_name_or_path,
                vae=vae,
                unet=unet,
                revision=None,
                variant=None,
                trust_remote_code=True,
            )
            pipeline.torch_dtype = torch.float32
            pipeline.scheduler.register_to_config(
                prediction_type=params.prediction_type
            )
            pipeline.set_progress_bar_config(disable=False)
            self.pipeline = pipeline
        else:
            self.pipeline = self.pipeline_class.from_pretrained(
                params.model_name_or_path, torch_dtype=torch.float32
            ).to(self.idle_device)

        super().__init__(
            pipeline=self.pipeline,
            forward_model=forward_model,
            params=params,
        )

    def reconstruct(self, measurements, **kwargs):

        # prep dc args with averaging
        dc_args, extra_outputs = self._prepare_args(measurements, **kwargs)

        # put pipeline to device
        self.pipeline.to(self.device)

        if self.is_conditional:
            assert isinstance(
                self.params, TextConditionalLDMParams
            ), "TextConditionalLDMParams should be used for text conditional LDMs"
            req_pipeline_args = dict(
                prompt=[dc_args["priors"]] * dc_args["n_measurements"],
                guidance_scale=self.params.classifier_free_guidance,
                num_images_per_prompt=self.n_avgs,
            )
        else:
            req_pipeline_args = dict(
                batch_size=dc_args["n_measurements"],
                num_images_per_query=self.n_avgs,
            )

        x = self.pipeline(
            **req_pipeline_args,
            width=self.im_size[1],
            height=self.im_size[0],
            num_inference_steps=self.num_inference_steps,
            output_type="pt",
            do_denormalize=False,
            **dc_args,
        )
        final_loss = x.final_loss
        x = x.images

        clear_cache()

        # recover mean and variance of recons
        avg_recon, recon_err = self._postprocess_outputs(
            x, measurements, dc_args, final_loss=final_loss
        )

        # put pipeline back to cpu
        self.pipeline.to(self.idle_device)
        clear_cache()

        return ReconstructorOutput(
            recon=avg_recon, error=recon_err, extra_outputs=extra_outputs
        )


# ------------------------------------------- CG/CG-prior reconstructor ---------------------------------------------- #


@dataclass
class CGParams(ReconParams):
    max_iter: int = 20
    tolerance: float = 1e-10
    lamda_l2: float = 1e-3
    use_prior: bool = False
    lamda_prior: float = 1e-3


class AHbReconstructor(Reconstructor):
    def __init__(self, forward_model: linop, params: ReconParams = ReconParams()):
        super().__init__(forward_model, params)

    def reconstruct(self, measurements, **kwargs):
        recon = self.forward_model.adjoint(measurements)
        return ReconstructorOutput(recon=recon)


class CGReconstructor(Reconstructor):
    def __init__(self, forward_model: linop, params: CGParams = CGParams()):
        super().__init__(forward_model, params)
        self.params = params
        self.forward_model = forward_model

    def reconstruct(self, measurements, **kwargs):
        if self.params.use_prior:
            assert (
                "priors" in kwargs
            ), "CG reconstructor defined with prior, but no prior was given"
            assert (
                kwargs["priors"].abs().max() <= 1
            ), "We assume prior is normalized [0, 1]"
            prior = kwargs["priors"]
        
        recons = []
        B = measurements.shape[0]
        for i in range(B):
            if self.params.use_prior:
                AHA = (
                    lambda x: self.forward_model.normal(x) + self.params.lamda_prior * x
                )
                AHb = (
                    self.forward_model.adjoint(measurements[i])
                    + self.params.lamda_prior * prior[i]
                )
            else:
                AHA = self.forward_model.normal
                AHb = self.forward_model.adjoint(measurements[i])

            x = conjugate_gradient(
                AHA,
                AHb,
                num_iters=self.params.max_iter,
                lamda_l2=self.params.lamda_l2,
                tolerance=self.params.tolerance,
            )
            recons.append(x)

        recons = torch.stack(recons)

        return ReconstructorOutput(recon=recons)


# ------------------------------------------- LA CS reconstructor ---------------------------------------------- #


@dataclass
class LACSParams(ReconParams):
    lamda_1: float = 1e-5  # wavelet control
    lamda_2: float = 1e-5  # prior control
    eps: float = 0.1
    max_iter: int = 5
    max_fista_iter: int = 40
    tol: float = 1e-6
    wave_name: str = "db4"


class LACSReconstructor(Reconstructor):
    def __init__(self, forward_model: linop, params: LACSParams = LACSParams()):
        super().__init__(forward_model, params)
        self.forward_model = forward_model
        self.params = params

    def reconstruct(self, measurements, **kwargs):
        recons = []
        priors = kwargs["priors"]  # complex data
        for meas, prior in zip(measurements, priors):
            recon = LACS_fista_recon(
                self.forward_model,
                meas,
                prior,
                lamda1=self.params.lamda_1,
                lamda2=self.params.lamda_2,
                eps=self.params.eps,
                num_iters=self.params.max_iter,
                num_fista_iters=self.params.max_fista_iter,
                tol=self.params.tol,
                wave_name=self.params.wave_name,
                verbose=False,
            )
            recons.append(recon)

        x = torch.stack(recons)

        return ReconstructorOutput(recon=x)


@dataclass
class ModlParams(ReconParams):
    n_layers: int = 5
    unroll_iters: int = 10
    n_filters: int = 64
    norm_type: str = "instance-affine"
    scale_denoiser: bool = True
    path: str = "path/to/modl/model.pth"


class MoDLReconstructor(Reconstructor):
    def __init__(
        self,
        forward_model: linop,
        params: ModlParams = ModlParams(),
        load_model_to_cpu=True,
    ):
        modl = Modl(
            n_layers=params.n_layers,
            unroll_iters=params.unroll_iters,
            n_filters=params.n_filters,
            norm_type=params.norm_type,
            scale_denoiser=params.scale_denoiser,
        )
        checkpoint = torch.load(params.path, map_location="cpu")
        modl.load_state_dict(checkpoint, strict=False)
        modl.eval()
        self.idle_device = get_torch_device(
            "cpu" if load_model_to_cpu else params.device
        )

        self.model = modl
        super().__init__(forward_model, params)

    def reconstruct(self, measurements, **kwargs):
        self.model = self.model.to(measurements.device)
        with torch.no_grad():
            recons = self.model(measurements, self.forward_model)
        self.model = self.model.to(self.idle_device)
        return ReconstructorOutput(recon=recons)
