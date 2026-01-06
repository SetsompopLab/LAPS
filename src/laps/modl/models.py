from typing import Optional

import torch
import torch.nn as nn

from laps.modl.utils import DataConsistency, c2r, get_conv_block, r2c


class CNNDenoiser(nn.Module):
    def __init__(
        self,
        n_layers=5,
        n_filters=64,
        use_prior=False,
        norm_type="instance-affine",
    ):
        super().__init__()

        self.use_prior = use_prior
        self.norm_type = norm_type

        layers = []

        if use_prior:
            # prior is real valued, so only one additional channel
            layers += get_conv_block(3, n_filters, norm_type=norm_type)
        else:
            layers += get_conv_block(2, n_filters, norm_type=norm_type)

        for _ in range(n_layers - 2):
            layers += get_conv_block(n_filters, n_filters, norm_type=norm_type)

        layers += get_conv_block(n_filters, 2, norm_type=norm_type, last_layer=True)

        self.model = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (B, 2, H, W)
        prior: Opt (B, H, W)

        Out: (B, 2, H, W)
        """
        x = c2r(x, axis=1)

        idt = x
        if self.use_prior:
            assert prior is not None, "Prior must be provided when use_prior is True"
            x = torch.cat([x, prior], dim=1)

        dw = self.model(x) + idt
        dw = r2c(dw, axis=1)

        return dw


class Modl(nn.Module):
    def __init__(
        self,
        n_layers,
        unroll_iters,
        n_filters,
        use_prior=False,
        train_dc_lambda=True,
        scale_denoiser=True,
        norm_type: str = "instance-affine",
        weights_init_scale: Optional[float] = None,
    ):
        super().__init__()

        self.use_prior = use_prior

        self.unroll_iters = unroll_iters
        self.n_filters = n_filters

        self.denoiser = CNNDenoiser(
            n_layers,
            n_filters=n_filters,
            use_prior=use_prior,
            norm_type=norm_type,
        )
        self.dc = DataConsistency(lambda_requires_grad=train_dc_lambda)

        self.scale_denoiser = scale_denoiser

        # initialize weights to be by a fraction if init_scale provided
        if weights_init_scale is not None:
            for m in self.denoiser.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=weights_init_scale)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=weights_init_scale)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, measurements, A, prior=None):

        if self.use_prior:
            assert prior is not None

        with torch.no_grad():
            x0 = A.H(measurements)
            # x0 = x0 / torch.linalg.vector_norm(x0, dim=(1, 2), keepdim=True)

            # Initialize with DC rather than AHb
            z_0 = torch.zeros_like(x0, dtype=torch.complex64)
            x_k = self.dc(z_0, x0.clone(), A)
            # x_k = x0.clone()

        for k in range(self.unroll_iters):

            if self.scale_denoiser:
                # scale before and after denoising
                with torch.no_grad():
                    scale = x_k.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
                x_k = x_k / scale
            else:
                scale = 1

            # Denoising
            z_k = self.denoiser(x_k, prior)

            if self.scale_denoiser:
                z_k = z_k * scale

            # Conjugate gradient
            x_k = self.dc(z_k, x0, A)

        return x_k
