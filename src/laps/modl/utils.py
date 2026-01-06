from math import prod
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def c2r(complex_img: torch.Tensor, axis=0) -> torch.Tensor:
    """
    :input shape: row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    return torch.stack(
        (complex_img.squeeze(dim=axis).real, complex_img.squeeze(dim=axis).imag),
        dim=axis,
    )


def r2c(real_img: torch.Tensor, axis=0) -> torch.Tensor:
    """
    :input shape: 2 x row x col (float32)
    :output shape: row x col (complex64)
    """
    if axis == 0:
        complex_img = real_img[0] + 1j * real_img[1]
    elif axis == 1:
        complex_img = real_img[:, 0] + 1j * real_img[:, 1]
    else:
        raise NotImplementedError

    return complex_img


def conjugate_gradient(
    AHA: Callable[[torch.Tensor], torch.Tensor],
    AHb: torch.Tensor,
    max_iterations: int = 10,
    tolerance: float = 1e-7,
    masked_update: bool = False,
    verbose: bool = True,
):
    """
    Batched conjugate gradient. Shape (B, X, Y)
    """
    ndim = AHb.ndim - 1
    B = prod(AHb.shape[:-ndim])
    sdims = tuple(range(-ndim, 0))

    zero_update = torch.zeros_like(AHb)

    x = AHb.clone() * 0.0
    r = AHb.clone()
    p = AHb.clone()

    rTr = torch.real(torch.sum(r.conj() * r, dim=sdims, keepdim=True))

    tol = torch.ones_like(rTr) * tolerance
    umask = rTr <= tol

    i = 0
    while i < max_iterations and (umask.sum() < B):
        AHAp = AHA(p)

        pAp = torch.real(torch.sum(p.conj() * AHAp, dim=sdims, keepdim=True))

        alpha = rTr / pAp

        # update only converged elements
        if masked_update:
            x = x + torch.where(umask, zero_update, alpha * p)
            r = r - torch.where(umask, zero_update, alpha * AHAp)
        else:
            x = x + alpha * p
            r = r - alpha * AHAp

        rTr_next = torch.real(torch.sum(torch.conj(r) * r, dim=sdims, keepdim=True))

        beta = rTr_next / rTr

        if masked_update:
            p = torch.where(umask, zero_update, r + beta * p)
        else:
            p = r + beta * p

        # iterate
        i = i + 1
        rTr = rTr_next
        umask = rTr <= tol

    if verbose:
        if i == max_iterations:
            print(
                f"Conjugate gradient did not converge after {max_iterations} iterations. "
                f"Convergence threshold: {tolerance}, "
                f"Unconverged elements: {B - umask.sum()}"
            )
        else:
            print(
                f"Conjugate gradient converged after {i} iterations. "
                f"Convergence threshold: {tolerance}, "
                f"Unconverged elements: {B - umask.sum()}"
            )

    return x


class ATALambda(nn.Module):
    def __init__(self, forward_model, lamda):
        super().__init__()
        self.forward_model = forward_model
        self.lamda = lamda

    def forward(self, im):
        return self.forward_model.normal(im) + self.lamda * im


class DataConsistency(nn.Module):
    def __init__(
        self, lamda_init=0.05, lambda_requires_grad=True, cg_max_iterations=25
    ):
        super().__init__()
        self.lamda = nn.Parameter(
            torch.tensor(lamda_init), requires_grad=lambda_requires_grad
        )
        self.cg_max_iterations = cg_max_iterations

    def forward(self, z_k, x0, forward_model):
        b = x0 + self.lamda * z_k
        A = ATALambda(forward_model, self.lamda)

        rec_im = conjugate_gradient(
            A, b, max_iterations=self.cg_max_iterations, verbose=False
        )

        return rec_im


class ResNetBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, bias: bool = False, is_last: bool = False
    ):
        super().__init__()
        self.is_last = is_last

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.identity_correction = None
        if in_channels != out_channels:
            self.identity_correction = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
                bias=bias,
            )
            self.bn3 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.identity_correction is not None:
            identity = self.identity_correction(x)
            identity = self.bn3(identity)

        out += identity
        # Don't use ReLU in the output
        if not self.is_last:
            out = F.relu(out)

        return out


def get_conv_block(
    in_channels, out_channels, last_layer=False, norm_type="instance-affine"
):

    assert norm_type in [
        "instance",
        "instance-affine",
        "batch",
        "batch-affine",
        "group",
        "group-affine",
    ]

    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding="same",
    )

    affine = "affine" in norm_type
    if "batch" in norm_type:
        bn = nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=False)
    elif "instance" in norm_type:
        bn = nn.InstanceNorm2d(out_channels, affine=affine, track_running_stats=False)
    elif "group" in norm_type:
        num_groups = out_channels // 8 if out_channels >= 8 else 1
        bn = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels,
            affine=affine,
        )
    else:
        raise NotImplementedError(f"Normalization type {norm_type} is not implemented.")

    if last_layer:
        return nn.Sequential(conv, bn)
    else:
        return nn.Sequential(conv, bn, nn.ReLU())
