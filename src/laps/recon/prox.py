from typing import Optional

import numpy as np
import sigpy as sp
import torch
import torch.nn as nn

from laps.recon.linops import WaveletLinop

__all__ = ["L1Wav", "prox_vector", "soft_thresh"]

"""
A proximal gradient is defined as
prox_g(x) = argmin_x 1/2 ||x - w||^2 + g(x)
"""


def soft_thresh(lamda: float, input: torch.Tensor) -> torch.Tensor:
    """
    Soft threshold.

    Γ_λ(x) = (|x| - λ) * sign(x) if |x| > λ, else 0
    """

    abs_input = input.abs()

    sign = torch.where(abs_input > 0, input / abs_input, torch.zeros_like(input))

    mag = abs_input - lamda
    mag = (mag.abs() + mag) / 2

    out = mag * sign

    return out


def prox_vector(x: torch.Tensor, lamda: float) -> torch.Tensor:
    """
    Solves:
      g_{λ}(x) = argmin_u { λ ||u||_1 + 1/2 ||u - x||^2 }

    via soft thresholding
    """
    u = soft_thresh(lamda, x)
    return lamda * torch.norm(u, p=1) + (1 / 2) * torch.norm(u - x) ** 2


class L1Wav(nn.Module):
    """Wavelet proximal operator mimicking Sid's implimentation"""

    def __init__(
        self,
        shape: tuple,
        lamda: float,
        axes: Optional[tuple] = None,
        rnd_shift: Optional[int] = 3,
        wave_name: Optional[str] = "db4",
    ):
        """
        Parameters:
        -----------
        shape - tuple
            the image/volume dimensions
        lamda - float
            Regularization strength
        axes - tuple
            axes to compute wavelet transform over
        rnd_shift - int
            randomly shifts image by rnd_shift in each dim before applying prox
        wave_name - str
            the type of wavelet to use from:
            ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus',
            'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor', ...]
            see https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#wavelet-families
        """
        super().__init__()

        # Using a fixed random number generator so that recons are consistent
        self.rng = torch.Generator()
        self.rng.manual_seed(1000)
        self.rnd_shift = rnd_shift

        # Save wavelet params
        if axes is None:
            axes = tuple([i for i in range(len(shape))])
        self.lamda = lamda
        self.axes = axes
        self.W = WaveletLinop(shape, axes=axes, wave_name=wave_name)

    def forward(
        self, input: torch.Tensor, alpha: Optional[float] = 1.0
    ) -> torch.Tensor:
        """
        Proximal operator for l1 wavelet

        Parameters
        ----------
        input - torch.tensor <complex> | CPU
            image/volume input
        alpha - float
            proximal 'alpha' term
        """

        # for scaling lamda so usage is more consistent
        xnorm = torch.norm(input)

        # Random stuff
        shift = torch.randint(
            -self.rnd_shift, self.rnd_shift + 1, (1,), generator=self.rng
        ).item()
        phase = (
            torch.exp(1j * torch.rand((1,), generator=self.rng) * 2 * np.pi)
            .to(torch.complex64)
            .to(input.device)
        )

        # Roll each axis
        nd = len(self.axes)
        input = torch.roll(input, (shift,) * nd, dims=self.axes)

        # Appoly random phase ...
        input *= phase

        # Apply prox
        output = self.W.H(soft_thresh(self.lamda * alpha * xnorm, self.W(input)))

        # Undo random phase ...
        output *= phase.conj()

        # Unroll
        output = torch.roll(output, (-shift,) * nd, dims=self.axes)

        return output
