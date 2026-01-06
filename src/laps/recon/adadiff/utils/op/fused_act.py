# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

""" Originated from https://github.com/rosinality/stylegan2-pytorch
The license for the original version of this file can be found in this directory (LICENSE_MIT).
"""

import torch
from torch import nn
from torch.nn import functional as F


def _fused_leaky_relu_impl(input, bias, negative_slope=0.2, scale=2**0.5):
    """Pure PyTorch implementation of fused bias + LeakyReLU + scale.

    This can be compiled with torch.compile for better performance.
    """
    if bias is not None:
        # Reshape bias to broadcast correctly
        if input.ndim == 4:  # NCHW format
            bias = bias.view(1, -1, 1, 1)
        elif input.ndim == 2:  # NC format
            bias = bias.view(1, -1)
        else:
            # Handle other dimensions by expanding bias
            shape = [1] * input.ndim
            shape[1] = bias.shape[0]  # Assume bias applies to channel dimension
            bias = bias.view(shape)

        input = input + bias

    # Apply LeakyReLU and scale
    output = F.leaky_relu(input, negative_slope=negative_slope)
    return output * scale


class FusedLeakyReLU(nn.Module):
    """PyTorch implementation of fused bias + LeakyReLU + scale operation.

    This module can be optimized with torch.compile for better performance.
    """

    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
    """Fused bias + LeakyReLU + scale operation using pure PyTorch.

    This function can be compiled with torch.compile for optimization:
        compiled_fn = torch.compile(fused_leaky_relu)

    Args:
        input: Input tensor
        bias: Bias tensor to add (can be None)
        negative_slope: Negative slope for LeakyReLU
        scale: Scale factor to apply after activation

    Returns:
        Output tensor after bias + LeakyReLU + scale
    """
    return _fused_leaky_relu_impl(input, bias, negative_slope, scale)


# For backward compatibility, compile the implementation if torch.compile is available
# Note: Automatic compilation disabled due to potential issues during testing
# Users can manually compile if needed: torch.compile(fused_leaky_relu, mode="max-autotune")
# try:
#     # Try to compile the implementation for better performance
#     _fused_leaky_relu_impl = torch.compile(_fused_leaky_relu_impl, mode="max-autotune")
# except Exception:
#     # If compilation fails, use the original implementation
#     pass
