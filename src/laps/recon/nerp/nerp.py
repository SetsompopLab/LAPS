from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from tqdm import tqdm

from laps.recon.algs import conjugate_gradient
from laps.recon.linops import linop
from laps.recon.nerp.networks import FFN, SIREN, Positional_Encoder
from laps.recon.reconstructor import ReconParams, Reconstructor, ReconstructorOutput
from laps.utils import clear_cache, normalize


@dataclass
class NERPEncoderParams:
    embedding: str = "gauss"
    embedding_size: int = 256
    coordinates_size: int = 2
    scale: float = 3.0


@dataclass
class NERPModelParams:
    model: str = "SIREN"
    network_input_size: int = 512
    network_output_size: int = 2
    network_depth: int = 8
    network_width: int = 512


@dataclass
class NERPPreTrainParams:
    max_iter: int = 1000
    lr: float = 1e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999

    # Early stopping parameters
    improvement_threshold: float = 0.98  # relative loss improvement threshold
    patience: int = 50
    min_iterations: int = 300  # minimum iterations before allowing early stopping


@dataclass
class NERPTrainParams:
    max_iter: int = 1000
    lr: float = 1e-5
    weight_decay: float = 0
    beta1: float = 0.9
    beta2: float = 0.999

    # Early stopping parameters
    improvement_threshold: float = 0.98  # relative loss improvement threshold
    patience: int = 100
    min_iterations: int = 300  # minimum iterations before allowing early stopping


@dataclass
class NERPParams(ReconParams):
    pre_train_params: NERPPreTrainParams = field(default_factory=NERPPreTrainParams)
    train_params: NERPTrainParams = field(default_factory=NERPTrainParams)
    model: NERPModelParams = field(default_factory=NERPModelParams)
    encoder: NERPEncoderParams = field(default_factory=NERPEncoderParams)
    # init prior image with CG recon phase
    init_prior_w_cg_phase: bool = True
    # after fitting to prior, add a scale to align better with aquired ksp
    add_scale_fix: bool = True
    # if True, will compute the initial scale fix from the CG recon
    init_scale_with_cg: bool = True


class NERPReconstructor(Reconstructor):
    LOG_INTERVAL = 100

    def __init__(self, forward_model: linop, params: NERPParams = NERPParams()):
        super().__init__(forward_model, params)

        assert params.model.model in ["SIREN", "FFN"], "Model must be SIREN or FFN"

        if params.model.model == "FFN":
            logger.warning(
                "FFN's output goes through a sigmoid, so it is in range [0, 1]. If you are"
                "using a complex prior, this can be problematic."
            )

        self.params = params

        self.encoder = Positional_Encoder(self.params.encoder)
        self.shape = None

    def _init_model(self):
        """
        Initialize the model and move it to the device.
        Returns the model in train mode.
        This has to be run before each reconstruction, as we need to re-initialize the model.
        """
        if self.params.model.model == "SIREN":
            model = SIREN(self.params.model)
        elif self.params.model.model == "FFN":
            model = FFN(self.params.model)
        else:
            raise ValueError(f"Model {self.params.model.model} not supported")

        return model.to(self.device).train()

    def reconstruct(
        self,
        measurements,
        priors,
        verbose: bool = True,
        log_images: bool = False,
        **kwargs,
    ):
        assert priors.is_complex()
        assert priors.ndim == 3, "prior shape should be (1, h, w)"

        # normalize prior to have max abs value of 1, as network output is in range [-1, 1]
        priors = priors / priors.abs().max()

        if log_images:
            self.log_dir = Path("nerp_debug")
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # init prior with CG recon phase
        if self.params.init_prior_w_cg_phase:
            cg_recon = conjugate_gradient(
                AHA=self.forward_model.normal,
                AHb=self.forward_model.adjoint(measurements),
                num_iters=25,
                tolerance=1e-10,
                lamda_l2=1e-3,
            )

            priors = priors.abs() * torch.exp(1j * cg_recon.angle())

        model = self._init_model()

        # if the current recon shape is the same as the previous one, we can reuse the same embedding
        shape = priors.shape[1:]
        if shape != self.shape:
            grid = self.create_grid(*shape)
            with torch.no_grad():
                self.emb = self.encoder(grid)
                emb = self.emb.to(self.device)
            self.shape = shape
        else:
            emb = self.emb.to(self.device)

        # prior is complex, with a batch size of 1: (1, h, w)
        prior = torch.concatenate((priors.real, priors.imag), dim=0)  # (2, h, w)
        self.prior = prior.clone()

        # First we train nerp on the prior image
        prior_recon = self.pre_train(
            model, emb, prior, verbose=verbose, log_images=log_images, **kwargs
        )

        # model's output is in the range [-1, 1], however, the scale of the recon might be different,
        # due to arbitrary ksp scaling.
        scale_fix = torch.eye(2).float()
        if self.params.add_scale_fix and self.params.init_scale_with_cg:
            scale_fix = self.get_scale_fix(prior_recon, cg_recon)

        # Then we train nerp on the measurements
        self.train(
            model,
            emb,
            measurements,
            scale_fix=scale_fix,
            verbose=verbose,
            log_images=log_images,
            **kwargs,
        )

        with torch.no_grad():
            output = model(emb).permute(2, 0, 1)
            recon = torch.complex(output[0], output[1])[
                None
            ].detach()  # add batch dimension

        # Clean ups
        model.cpu()
        self.encoder.cpu()
        self.emb.cpu()
        clear_cache()

        if log_images:
            self.save_image(
                recon[0], self.prior, target=kwargs.get("x0", None), name="final"
            )

        return ReconstructorOutput(recon=recon, error=None)

    def pre_train(
        self,
        model,
        emb,
        prior_image,
        verbose: bool = False,
        log_images: bool = True,
        **kwargs,
    ):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.params.pre_train_params.lr,
            betas=(
                self.params.pre_train_params.beta1,
                self.params.pre_train_params.beta2,
            ),
            weight_decay=self.params.pre_train_params.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        prior_image = prior_image.to(self.device)

        pbar = tqdm(
            range(self.params.pre_train_params.max_iter),
            desc="Pre-training on prior image",
            leave=False,
            disable=not verbose,
        )

        patience_counter = 0
        best_loss = float("inf")

        for i in pbar:
            optimizer.zero_grad()
            output = model(emb).permute(2, 0, 1)
            loss = 0.5 * loss_fn(output, prior_image)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()

            # count iterations without improvement
            if (
                current_loss
                < best_loss * self.params.pre_train_params.improvement_threshold
            ):
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Check for convergence after minimum iterations
            converged = False
            if (
                i >= self.params.pre_train_params.min_iterations
                and patience_counter >= self.params.pre_train_params.patience
            ):
                converged = True

            pbar.set_postfix(
                loss=f"Loss={loss.item():.2e}, pSNR={-10 * torch.log10(2 * loss).item():.3f}, patience={patience_counter}"
            )

            if log_images:
                if i % self.LOG_INTERVAL == 0:
                    self.save_image(
                        output,
                        self.prior,
                        target=kwargs.get("x0", None),
                        name=f"pre_train_{i}",
                    )

            if converged:
                if verbose:
                    logger.info(f"Pre-training converged at iteration {i + 1}")
                break

        with torch.no_grad():
            output = model(emb).permute(2, 0, 1)

        return output

    def train(
        self,
        model,
        emb,
        measurements,
        scale_fix,
        verbose: bool = False,
        log_images: bool = True,
        **kwargs,
    ):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.params.train_params.lr,
            betas=(
                self.params.train_params.beta1,
                self.params.train_params.beta2,
            ),
        )

        x_gt = kwargs.get("x0", None)  # for debugging
        if x_gt is not None:
            x_gt = x_gt[0].to(self.device)
            x_gt = x_gt / x_gt.abs().max()

        def complex_loss_fn(pred, gt):
            return torch.mean(torch.abs(pred - gt) ** 2)

        measurements = measurements.to(self.device)

        pbar = tqdm(
            range(self.params.train_params.max_iter),
            desc="Training on measurements",
            leave=False,
            disable=not verbose,
        )

        # if we wand to train this scale_fix, we need to add it to the optimizer
        if self.params.add_scale_fix:
            scale_fix = scale_fix.to(self.device)
            scale_fix.requires_grad = True
            optimizer.add_param_group({"params": scale_fix})

        patience_counter = 0
        best_loss = float("inf")

        for i in pbar:
            optimizer.zero_grad()
            output = model(emb)
            output = output @ scale_fix
            output = output.permute(2, 0, 1)
            cmplx_output = torch.complex(output[0], output[1])
            ksp_hat = self.forward_model(cmplx_output)[None]  # add batch dimension
            loss = complex_loss_fn(ksp_hat, measurements)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()

            # Early stopping logic
            if (
                current_loss
                < best_loss * self.params.train_params.improvement_threshold
            ):
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Check for convergence after minimum iterations
            converged = False
            if (
                i >= self.params.train_params.min_iterations
                and patience_counter >= self.params.train_params.patience
            ):
                converged = True

            if x_gt is not None:
                norm_recon = normalize(cmplx_output, x_gt, ofs=False, mag=True)
                psnr = -10 * torch.log10(complex_loss_fn(norm_recon, x_gt))
                pbar.set_postfix(
                    loss=f"Loss={loss.item():.2e}, pSNR={psnr.item():.3f}, patience={patience_counter}"
                )
            else:
                pbar.set_postfix(
                    loss=f"Loss={loss.item():.2e}, patience={patience_counter}"
                )

            if log_images:
                if i % self.LOG_INTERVAL == 0:
                    self.save_image(
                        output,
                        self.prior,
                        target=kwargs.get("x0", None),
                        name=f"train_{i}",
                    )

            if converged:
                if verbose:
                    logger.info(f"Training converged at iteration {i + 1}")
                break

    def get_scale_fix(self, prior_recon, cg_recon):
        if prior_recon.is_complex():
            prior_recon = torch.concatenate((prior_recon.real, prior_recon.imag), dim=0)
        if cg_recon.is_complex():
            cg_recon = torch.concatenate((cg_recon.real, cg_recon.imag), dim=0)

        # we find a scaling factor such that || prior_recon * scale - cg_recon ||_2 is minimized.
        # Since prior and cg_recon our complex, this scaling factor would be a (2, 2) matrix.
        A = rearrange(prior_recon, "c h w -> (h w) c")
        b = rearrange(cg_recon, "c h w -> (h w) c")
        scale = torch.linalg.lstsq(A, b).solution

        return scale

    @staticmethod
    def create_grid(h, w):
        grid_y, grid_x = torch.meshgrid(
            [torch.linspace(0, 1, steps=h), torch.linspace(0, 1, steps=w)]
        )
        grid = torch.stack([grid_y, grid_x], dim=-1)

        return grid

    @torch.no_grad()
    def save_image(self, recon, prior, target=None, name: str = ""):
        if not recon.is_complex():
            recon_cmplx = torch.complex(recon[0], recon[1]).detach()
        else:
            recon_cmplx = recon.detach()
        recon_phase = recon_cmplx.angle().cpu().numpy()
        recon_cmplx = recon_cmplx.abs()
        recon_cmplx = recon_cmplx / recon_cmplx.max()
        recon_cmplx = recon_cmplx.cpu().numpy()

        if not prior.is_complex():
            prior_cmplx = torch.complex(prior[0], prior[1]).detach()
        else:
            prior_cmplx = prior.detach()
        prior_phase = prior_cmplx.angle().cpu().numpy()
        prior_cmplx = prior_cmplx.abs()
        prior_cmplx = prior_cmplx / prior_cmplx.max()
        prior_cmplx = prior_cmplx.cpu().numpy()

        if target is not None:
            target_phase = target[0].angle().cpu().numpy()
            target = target[0].abs()
            target = target / target.max()
            target = target.cpu().numpy()

        n_cols = 3 if target is not None else 2

        fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 5, 10))
        plt.subplots_adjust(top=0.85)  # Add more space at the top

        axs[0, 0].imshow(recon_cmplx, cmap="gray", vmin=0, vmax=1)
        axs[0, 0].set_title("Reconstructed Magnitude")
        axs[1, 0].imshow(recon_phase, cmap="gray", vmin=-np.pi, vmax=np.pi)
        axs[1, 0].set_title("Reconstructed Phase")

        axs[0, 1].imshow(prior_cmplx, cmap="gray", vmin=0, vmax=1)
        axs[0, 1].set_title("Prior Magnitude")
        axs[1, 1].imshow(prior_phase, cmap="gray", vmin=-np.pi, vmax=np.pi)
        axs[1, 1].set_title("Prior Phase")

        if target is not None:
            axs[0, 2].imshow(target, cmap="gray", vmin=0, vmax=1)
            axs[0, 2].set_title("Target Magnitude")
            axs[1, 2].imshow(target_phase, cmap="gray", vmin=-np.pi, vmax=np.pi)
            axs[1, 2].set_title("Target Phase")

        for ax in axs.flatten():
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(self.log_dir / f"{name}.png")

        plt.close()
