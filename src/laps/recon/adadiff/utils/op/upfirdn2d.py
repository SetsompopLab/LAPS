# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

""" Originated from https://github.com/rosinality/stylegan2-pytorch
The license for the original version of this file can be found in this directory (LICENSE_MIT).
"""

from collections import abc

import torch
from torch.nn import functional as F


def _upfirdn2d_native_impl(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    """Pure PyTorch implementation of upfirdn2d operation.

    This can be compiled with torch.compile for better performance.
    """
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Upsampling, FIR filtering, and downsampling operation using pure PyTorch.

    This function can be compiled with torch.compile for optimization:
        compiled_fn = torch.compile(upfirdn2d)

    Args:
        input: Input tensor [batch, channels, height, width]
        kernel: Filter kernel [kernel_height, kernel_width]
        up: Upsampling factor (int or tuple)
        down: Downsampling factor (int or tuple)
        pad: Padding (tuple of 2 values for symmetric padding)

    Returns:
        Output tensor after upsampling, filtering, and downsampling
    """
    return _upfirdn2d_native_impl(
        input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )


def upfirdn2d_ada(input, kernel, up=1, down=1, pad=(0, 0)):
    """More flexible version of upfirdn2d with asymmetric up/down and padding support.

    Args:
        input: Input tensor [batch, channels, height, width]
        kernel: Filter kernel [kernel_height, kernel_width]
        up: Upsampling factor (int or tuple of 2)
        down: Downsampling factor (int or tuple of 2)
        pad: Padding (tuple of 2 or 4 values)

    Returns:
        Output tensor after upsampling, filtering, and downsampling
    """
    if not isinstance(up, abc.Iterable):
        up = (up, up)

    if not isinstance(down, abc.Iterable):
        down = (down, down)

    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])

    return _upfirdn2d_native_impl(input, kernel, *up, *down, *pad)


# Legacy alias for backward compatibility
upfirdn2d_native = _upfirdn2d_native_impl


# For backward compatibility, compile the implementation if torch.compile is available
# Note: Automatic compilation disabled due to potential issues with dynamic shapes
# Users can manually compile if needed: torch.compile(upfirdn2d, mode="max-autotune")
# try:
#     # Try to compile the implementation for better performance
#     _upfirdn2d_native_impl = torch.compile(_upfirdn2d_native_impl, mode="max-autotune")
# except Exception:
#     # If compilation fails, use the original implementation
#     pass
