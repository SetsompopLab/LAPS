from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from laps import PROJECT_ROOT
from laps.metrics import PSNR
from laps.recon.adadiff.utils.networks.ncsnpp_generator_adagn import NCSNpp
from laps.recon.linops import linop
from laps.recon.reconstructor import ReconParams, Reconstructor, ReconstructorOutput
from laps.utils import normalize


@dataclass
class AdaDiffModelParams:
    num_channels: int = 2
    not_use_tanh: bool = False
    z_emb_dim: int = 256
    num_channels_dae: int = 64
    ch_mult: list[int] = field(default_factory=lambda: [1, 1, 1, 2, 2])
    num_res_blocks: int = 2
    attn_resolutions: list[int] = field(default_factory=lambda: [18])
    dropout: float = 0.0
    resamp_with_conv: bool = True
    conditional: bool = True
    fir: bool = True
    fir_kernel: list[int] = field(default_factory=lambda: [1, 3, 3, 1])
    skip_rescale: bool = True
    resblock_type: str = "biggan"
    progressive: str = "none"
    progressive_input: str = "residual"
    progressive_combine: str = "sum"
    embedding_type: str = "positional"
    fourier_scale: float = 16.0
    image_size: int = 256
    nz: int = 100
    n_mlp: int = 3
    centered: bool = True


@dataclass
class AdaDiffParams(ReconParams):
    model: AdaDiffModelParams = field(default_factory=AdaDiffModelParams)
    checkpoint_path: str = field(
        default_factory=lambda: str(PROJECT_ROOT / "models" / "adadiff.pth")
    )
    lr_g: float = 1e-3
    beta1: float = 0.5
    beta2: float = 0.9
    itr_inf: int = 1000
    num_timesteps: int = 8
    beta_min: float = 0.1
    beta_max: float = 20.0
    use_geometric: bool = False


def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1.0 - torch.exp(2.0 * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Posterior_Coefficients:
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (
                torch.tensor([1.0], dtype=torch.float32, device=device),
                self.alphas_cumprod[:-1],
            ),
            0,
        )
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(
            coefficients.posterior_log_variance_clipped, t, x_t.shape
        )
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = 1 - (t == 0).type(torch.float32)
        return (
            mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise
        )

    sample_x_pos = p_sample(x_0, x_t, t)
    return sample_x_pos


class AdaDiffReconstructor(Reconstructor):
    LOG_INTERVAL = 100

    def __init__(self, forward_model: linop, params: AdaDiffParams = AdaDiffParams()):
        super().__init__(forward_model, params)
        self.params = params
        self.model = NCSNpp(params.model)
        self.model.load_state_dict(torch.load(params.checkpoint_path))

    def _init_model(self):
        model = NCSNpp(self.params.model)
        model.load_state_dict(self.model.state_dict())

        return model.to(self.device).train()

    def reconstruct(
        self,
        measurements,
        priors,
        verbose: bool = True,
        log_images: bool = False,
        **kwargs,
    ):
        if log_images:
            self.log_dir = Path("adadiff_debug")
            self.log_dir.mkdir(parents=True, exist_ok=True)

        model = self._init_model()

        optimizerG = optim.Adam(
            model.parameters(),
            lr=self.params.lr_g,
            betas=(self.params.beta1, self.params.beta2),
        )
        pos_coeff = Posterior_Coefficients(self.params, device=self.device)

        x_gt = kwargs.get("x0", None)  # for debugging

        mask = self.forward_model.mask.squeeze(0).abs() > 0.5
        AHb = self.forward_model.adjoint(measurements).squeeze(0)
        measurements = measurements.squeeze(0)

        sample_diff = self.sample_from_model(
            pos_coeff,
            model,
            self.params.num_timesteps,
            measurements,
            AHb,
            mask,
            self.params.model.nz,
        )

        t_time = torch.zeros([1], device=self.device)
        latent_z = torch.randn(1, self.params.model.nz, device=self.device)

        def complex_loss_fn(pred, gt):
            return torch.mean(torch.abs(pred - gt))

        pbar = tqdm(
            range(self.params.itr_inf),
            desc="AdaDiff Reconstruction fine tunning generator",
            leave=False,
            disable=not verbose,
        )

        if log_images:
            self.save_image(
                torch.complex(sample_diff[0, 0], sample_diff[0, 1]),
                target=x_gt[0] if x_gt is not None else None,
                name="adadiff_recon_init",
            )

        for i in pbar:
            model.zero_grad()
            sample = model(sample_diff, t_time, latent_z)
            sample_cmplx = torch.complex(sample[0, 0], sample[0, 1])
            y_hat = self.forward_model(sample_cmplx)
            gain = self.get_gain(y_hat, measurements, mask)
            loss = complex_loss_fn(y_hat * gain, measurements)
            loss.backward()
            optimizerG.step()

            if x_gt is not None:
                norm_recon = normalize(sample_cmplx, x_gt[0], ofs=False, mag=True)
                psnr = PSNR(norm_recon[None, ...].detach().cpu().abs(), x_gt.abs())
                pbar.set_postfix(
                    loss=f"Loss={loss.item():.2e}, pSNR={psnr.item():.3f}, gain={gain.item():.3f}"
                )
            else:
                pbar.set_postfix(loss=f"Loss={loss.item():.2e}, gain={gain.item():.3f}")

            if log_images:
                if i % self.LOG_INTERVAL == 0:
                    self.save_image(
                        sample_cmplx,
                        target=x_gt[0] if x_gt is not None else None,
                        name=f"adadiff_recon_{i}",
                    )

        recon = (
            self.data_consistency(sample_cmplx, measurements, mask, match_gain=True)
            .detach()
            .cpu()
        )

        model = model.cpu()
        del model

        return ReconstructorOutput(recon=recon, error=None)

    @staticmethod
    def get_time_schedule(num_timesteps, device):
        n_timestep = num_timesteps
        eps_small = 1e-3
        t = np.arange(0, n_timestep + 1, dtype=np.float64)
        t = t / n_timestep
        t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small

        return t.to(device)

    @staticmethod
    @torch.no_grad()
    def get_gain(src, tgt, mask=None):
        if mask is not None:
            src_in_mask = src[:, mask].flatten().abs()
            tgt_in_mask = tgt[:, mask].flatten().abs()
            gain = torch.linalg.vecdot(src_in_mask, tgt_in_mask) / torch.linalg.vecdot(
                src_in_mask, src_in_mask
            )
        else:
            gain = torch.linalg.vecdot(src.abs(), tgt.abs()) / torch.linalg.vecdot(
                src.abs(), src.abs()
            )

        return gain

    def data_consistency(self, x, measurements, mask, match_gain=True):
        ksp_hat = self.forward_model(x, use_mask=False)
        if match_gain:
            gain = self.get_gain(measurements, ksp_hat, mask)
        else:
            gain = 1.0
        ksp_full = (ksp_hat * (~mask)) + ((gain * measurements) * mask)
        # match measurements to model output
        x = self.forward_model.adjoint(ksp_full, use_mask=False)

        return x

    @torch.no_grad()
    def sample_from_model(
        self,
        coefficients,
        generator,
        n_time,
        measurements,
        AHb,
        mask,
        nz,
    ):
        x = AHb
        x = x / x.abs().max()
        x = torch.stack((x.real, x.imag),)[
            None, ...
        ]  # model needs a batch dim
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), nz, device=x.device)  # .to(x.device)
            x_0 = generator(x, t_time, latent_z)
            x_0 = torch.complex(x_0[:, 0], x_0[:, 1]).squeeze(0)
            x_0 = self.data_consistency(x_0, measurements, mask, match_gain=True)
            x_0 = torch.stack(
                (x_0.real, x_0.imag),
            )[None, ...]
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
        x_0 = generator(x, t_time, latent_z)
        x = x_0.detach()

        return x

    @torch.no_grad()
    def save_image(self, recon, target=None, name: str = ""):
        if not recon.is_complex():
            recon_cmplx = torch.complex(recon[0], recon[1]).detach()
        else:
            recon_cmplx = recon.detach()
        if target is not None:
            recon_cmplx = normalize(recon_cmplx, target, ofs=False, mag=True)

        recon_phase = recon_cmplx.angle().cpu().numpy()
        recon_cmplx = recon_cmplx.abs()
        recon_cmplx = recon_cmplx / recon_cmplx.max()
        recon_cmplx = recon_cmplx.cpu().numpy()

        if target is not None:
            target_phase = target.angle().cpu().numpy()
            target = target.abs()
            target = target / target.max()
            target = target.cpu().numpy()

        n_cols = 2 if target is not None else 1

        fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 5, 10))
        if n_cols == 1:
            axs = axs[None, ...]
        plt.subplots_adjust(top=0.85)  # Add more space at the top

        axs[0, 0].imshow(recon_cmplx, cmap="gray", vmin=0, vmax=1)
        axs[0, 0].set_title("Reconstructed Magnitude")
        axs[1, 0].imshow(recon_phase, cmap="gray", vmin=-np.pi, vmax=np.pi)
        axs[1, 0].set_title("Reconstructed Phase")

        if target is not None:
            axs[0, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axs[0, 1].set_title("Target Magnitude")
            axs[1, 1].imshow(target_phase, cmap="gray", vmin=-np.pi, vmax=np.pi)
            axs[1, 1].set_title("Target Phase")

        for ax in axs.flatten():
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(self.log_dir / f"{name}.png")

        plt.close()
