import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

import wandb
from laps.recon.reconstructor import ReconstructorOutput
from laps.utils import normalize

from .utils import dark_mode


@dataclass
class PlotConfig:
    """Configuration for image plotting parameters."""

    error_scale: float = 5.0
    figure_size_per_col: float = 4.0
    dpi: int = 300
    fontsize: int = 16
    mask_color: str = "purple"
    error_text_color: str = "white"
    quantile_norm: float = 0.99
    epsilon: float = 1e-9

    # Colormap configurations
    mag_cmap: str = "gray"
    error_cmap: str = "inferno"
    var_cmap: str = "jet"
    phase_cmap: str = "jet"
    mask_cmap: str = "gray"


@dataclass
class PlotParams:
    """Parameters for different plot types."""

    mag_params: dict
    error_params: dict
    var_params: dict
    phase_params: dict
    mask_params: dict

    @classmethod
    def from_config(cls, config: PlotConfig) -> "PlotParams":
        """Create PlotParams from PlotConfig."""
        return cls(
            mag_params={"cmap": config.mag_cmap, "vmin": 0, "vmax": 1},
            error_params={
                "cmap": config.error_cmap,
                "vmin": 0,
                "vmax": 1 / config.error_scale,
            },
            var_params={"cmap": config.var_cmap, "vmin": 0, "vmax": 10},
            phase_params={"cmap": config.phase_cmap, "vmin": -np.pi, "vmax": np.pi},
            mask_params={"cmap": config.mask_cmap, "vmin": 0, "vmax": 1},
        )


class ImageLogger:
    """Logger for reconstruction images with improved modularity and error handling."""

    def __init__(
        self,
        log_dir: Union[str, Path],
        wandb_enable: bool = True,
        config: Optional[PlotConfig] = None,
    ):
        """
        Initialize ImageLogger.

        Args:
            log_dir: Directory for logging outputs
            config: Configuration for plotting parameters
        """
        self.log_dir = Path(log_dir)
        self.fig_dir = self.log_dir / "figs"
        self.recon_dir = self.log_dir / "recon"
        self.wandb_enable = wandb_enable

        # Create directories
        for directory in [self.log_dir, self.fig_dir, self.recon_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.config = config or PlotConfig()
        self.plot_params = PlotParams.from_config(self.config)
        self.index = 0

    def calculate_l1_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate L1 error between two complex images."""
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("Input tensors must be torch.Tensor")
        return (x.abs() - y.abs()).abs()

    def calculate_nrmse(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate normalized root mean square error."""
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("Input tensors must be torch.Tensor")

        x_abs = x.abs()
        y_abs = y.abs()
        norm_x = torch.norm(x_abs)

        if norm_x == 0:
            return torch.tensor(float("inf"))

        return torch.norm(x_abs - y_abs) / norm_x

    def _normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        """Normalize target image using quantile normalization."""
        quantile_val = torch.quantile(target.abs(), self.config.quantile_norm)
        return target / (quantile_val + self.config.epsilon)

    def _setup_figure_layout(
        self, n_reconstructors: int, has_variance: bool, has_prior: bool
    ) -> Tuple[int, int]:
        """Determine figure layout dimensions."""
        n_rows = 4 if has_variance else 3  # mag, error, [variance], phase
        n_cols = n_reconstructors + 1  # target, prior, + reconstructors
        if has_prior:
            n_cols += 1
        return n_rows, n_cols

    def _setup_axes_labels(
        self,
        ax: np.ndarray,
        n_cols: int,
        recon_labels: Sequence[str],
        has_variance: bool,
        has_prior: bool,
    ) -> int:
        """Setup axis labels and titles."""
        error_scale = self.config.error_scale
        phase_row = 3 if has_variance else 2

        # Row labels
        ax[0, 0].set_ylabel("Mag")
        ax[1, 0].set_ylabel(f"Error: {error_scale}x")
        if has_variance:
            ax[2, 0].set_ylabel("Variance")
        ax[phase_row, 0].set_ylabel("Phase")

        # Column titles
        ax[0, 0].set_title("GT")
        ax[0, 1].set_title("Prior")
        ofs = 2 if has_prior else 1
        for idx, label in enumerate(recon_labels):
            ax[0, idx + ofs].set_title(label)

        return phase_row

    def _plot_target_and_prior(
        self,
        ax: np.ndarray,
        target: torch.Tensor,
        prior: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        phase_row: int,
        blank_img: np.ndarray,
    ) -> None:
        """Plot target and prior images."""
        # Target magnitude and phase
        ax[0, 0].imshow(target.abs().cpu().numpy(), **self.plot_params.mag_params)
        ax[phase_row, 0].imshow(
            target.angle().cpu().numpy(), **self.plot_params.phase_params
        )

        # Prior magnitude, error, and phase
        if prior is not None:
            ax[0, 1].imshow(prior.abs().cpu().numpy(), **self.plot_params.mag_params)
            ax[1, 1].imshow(
                self.calculate_l1_error(target, prior).cpu().numpy(),
                **self.plot_params.error_params,
            )
            ax[phase_row, 1].imshow(
                prior.angle().cpu().numpy(), **self.plot_params.phase_params
            )

            # Add NRMSE text to prior error
            nrmse_val = self.calculate_nrmse(target, prior)
            ax[1, 1].text(
                0.02,
                0.98,
                f"NRMSE: {nrmse_val:.3f}",
                ha="left",
                va="top",
                fontsize=self.config.fontsize,
                color=self.config.error_text_color,
                transform=ax[1, 1].transAxes,
            )

        # Handle mask or show blank in error position for target
        if mask is not None:
            mask_data = mask[0] if mask.ndim == 3 else mask
            ax[1, 0].imshow(mask_data.cpu().numpy(), **self.plot_params.mask_params)
            ax[1, 0].text(
                0.02,
                0.98,
                "Mask",
                ha="left",
                va="top",
                fontsize=self.config.fontsize,
                color=self.config.mask_color,
                transform=ax[1, 0].transAxes,
            )
        else:
            ax[1, 0].imshow(blank_img, **self.plot_params.error_params)

    def _plot_reconstructions(
        self,
        ax: np.ndarray,
        recon_dict: Dict[str, ReconstructorOutput],
        target: torch.Tensor,
        phase_row: int,
        has_variance: bool,
        has_prior: bool,
        blank_img: np.ndarray,
        image_idx: int,
    ) -> None:
        """Plot reconstruction results."""

        start_col = 2 if has_prior else 1  # Start from 2 if prior is included
        for col_idx, (label, recon_output) in enumerate(
            recon_dict.items(), start=start_col
        ):
            recon = recon_output.recon[image_idx]
            recon = normalize(recon, target, mag=True, ofs=False)

            # Magnitude and phase
            ax[0, col_idx].imshow(
                recon.abs().cpu().numpy(), **self.plot_params.mag_params
            )
            ax[phase_row, col_idx].imshow(
                recon.angle().cpu().numpy(), **self.plot_params.phase_params
            )

            # Error
            error_img = self.calculate_l1_error(target, recon).cpu().numpy()
            ax[1, col_idx].imshow(error_img, **self.plot_params.error_params)

            # Add NRMSE text
            nrmse_val = self.calculate_nrmse(target, recon)
            ax[1, col_idx].text(
                0.02,
                0.98,
                f"NRMSE: {nrmse_val:.3f}",
                ha="left",
                va="top",
                fontsize=self.config.fontsize,
                color=self.config.error_text_color,
                transform=ax[1, col_idx].transAxes,
            )

            # Variance (if applicable)
            if has_variance:
                if recon_output.error is not None:
                    var_data = recon_output.error[image_idx].cpu().numpy()
                else:
                    var_data = blank_img.copy()
                ax[2, col_idx].imshow(var_data, **self.plot_params.var_params)

            # Save reconstruction to disk
            self._save_reconstruction_data(recon, recon_output.error, image_idx, label)

    def _save_reconstruction_data(
        self,
        recon: torch.Tensor,
        error: Optional[torch.Tensor],
        image_idx: int,
        label: str,
    ) -> None:
        """Save reconstruction and uncertainty data to disk."""
        # Save reconstruction
        recon_path = self.recon_dir / f"ex{self.index}_recon_{label}.npy"
        np.save(recon_path, recon.cpu().numpy())

        # Save uncertainty if available
        if error is not None:
            uncertainty_path = (
                self.recon_dir / f"ex{self.index}_uncertainty_{label}.npy"
            )
            np.save(uncertainty_path, error[image_idx].cpu().numpy())

    def _save_target_and_prior_data(
        self,
        target: torch.Tensor,
        prior: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Save target and prior data to disk."""
        np.save(self.recon_dir / f"ex{self.index}_targ.npy", target.cpu().numpy())
        if prior is not None:
            np.save(self.recon_dir / f"ex{self.index}_prior.npy", prior.cpu().numpy())
        if mask is not None:
            mask_data = mask[0] if mask.ndim == 3 else mask
            np.save(self.recon_dir / f"ex{self.index}_mask.npy", mask_data.cpu().numpy())

    def _finalize_plot(self, fig: Figure, ax: Union[Axes, np.ndarray]) -> np.ndarray:
        """Finalize plot formatting and save."""
        # Remove ticks
        if isinstance(ax, np.ndarray):
            for a in ax.ravel():
                a.set_xticks([])
                a.set_yticks([])
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        # Apply dark mode and save
        fig.tight_layout()
        fig, ax = dark_mode(fig, ax)

        fig_path = self.fig_dir / f"recon{self.index}.png"
        fig.savefig(fig_path, dpi=self.config.dpi)
        plt.close(fig)

        # Return image as array for wandb logging
        return np.array(Image.open(fig_path))

    def _log_to_wandb(
        self,
        image_array: np.ndarray,
        wandb_prefix: str,
        image_idx: int,
        step: Optional[int],
    ) -> None:
        """Log image to wandb."""
        try:
            if step is not None:
                wandb.log(
                    {f"{wandb_prefix}recon_{image_idx}": wandb.Image(image_array)},
                    step=step,
                )
            else:
                wandb.log({f"{wandb_prefix}recon": wandb.Image(image_array)})
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")

    def log_images(
        self,
        recon_dict: Dict[str, ReconstructorOutput],
        targets: torch.Tensor,
        priors: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        wandb_prefix: str = "val/",
        n_images_to_log: int = -1,
        step: Optional[int] = None,
    ) -> None:
        """
        Log reconstruction images with improved modularity.

        Args:
            recon_dict: Dictionary of reconstruction outputs
            targets: Target images
            priors: Prior images
            mask: Optional mask images
            wandb_prefix: Prefix for wandb logging
            n_images_to_log: Number of images to log (-1 for all)
            step: Training step for wandb logging
        """
        if not recon_dict:
            raise ValueError("recon_dict cannot be empty")

        batch_size = targets.shape[0]

        # Determine if any reconstruction has variance/error
        has_variance = any(output.error is not None for output in recon_dict.values())
        has_prior = priors is not None

        # Setup figure dimensions
        n_rows, n_cols = self._setup_figure_layout(
            len(recon_dict), has_variance, has_prior
        )

        # Determine number of images to log
        n_images_to_log = min(
            batch_size if n_images_to_log <= 0 else n_images_to_log, batch_size
        )

        for i in range(n_images_to_log):
            # Normalize images
            target = self._normalize_target(targets[i])

            if priors is not None:
                prior = normalize(priors[i], target, mag=True, ofs=False)
            else:
                prior = None

            # Create figure
            fig_width = self.config.figure_size_per_col * n_cols
            fig_height = self.config.figure_size_per_col * n_rows
            fig, ax = plt.subplots(
                n_rows, n_cols, figsize=(fig_width, fig_height), dpi=self.config.dpi
            )

            # Ensure ax is 2D for consistency
            if n_rows == 1:
                ax = ax.reshape(1, -1)
            if n_cols == 1:
                ax = ax.reshape(-1, 1)

            # Setup labels and get phase row index
            phase_row = self._setup_axes_labels(
                ax, n_cols, list(recon_dict.keys()), has_variance, has_prior
            )

            # Create blank image for padding
            blank_img = np.zeros_like(target.abs().cpu().numpy())

            # Plot target and prior
            current_mask = mask[i] if mask is not None and mask.ndim == 3 else mask
            self._plot_target_and_prior(
                ax, target, prior, current_mask, phase_row, blank_img
            )

            # Fill variance row for target and prior if needed
            if has_variance:
                ax[2, 0].imshow(blank_img, **self.plot_params.var_params)
                if has_prior:
                    ax[2, 1].imshow(blank_img, **self.plot_params.var_params)

            # Plot reconstructions
            self._plot_reconstructions(
                ax, recon_dict, target, phase_row, has_variance, has_prior, blank_img, i
            )

            # Save raw data
            self._save_target_and_prior_data(target, prior, mask=current_mask)

            # Finalize and save plot
            image_array = self._finalize_plot(fig, ax)

            # Log to wandb
            if self.wandb_enable:
                self._log_to_wandb(image_array, wandb_prefix, i, step)

            self.index += 1
