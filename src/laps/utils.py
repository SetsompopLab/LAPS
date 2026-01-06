import gc
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cupy as cp
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from einops import einsum, rearrange
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

"""
Interface-related Utils
"""

logger = get_logger(__name__)


def trylogprint(string):
    try:
        logger.info(string)
    except Exception:
        print(string)


def create_exp_dir(logs_dir: Path, exp_name: str):
    current_datetime = datetime.now()
    date_string = current_datetime.strftime("%Y-%m-%d_%H-%M")
    exp_dir = logs_dir / exp_name / date_string
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def get_torch_device(device: Union[str, int, torch.device]) -> torch.device:
    """
    Get a torch device from a string or integer.
    Args:
        device: device to use, can be a string (e.g. "cuda", "cpu") or an integer (e.g. 0 for "cuda:0")
    Returns:
        torch.device: the device to use
    """
    if isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, int):
        return torch.device(f"cuda:{device}")
    elif isinstance(device, torch.device):
        return device
    else:
        raise ValueError(f"Unsupported device type: {type(device)}")


def ensure_torch(x, device=torch.device("cpu")) -> torch.Tensor:
    device = get_torch_device(device)

    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    else:
        raise ValueError(f"Unsupported type {type(x)} for conversion to torch.Tensor")


def torch_resize(input: torch.Tensor, oshape: tuple) -> torch.Tensor:
    """Resize with zero-padding or cropping.

    Args:
        input (torch.Tensor): Input array.
        oshape (tuple of ints): Output shape.

    Returns:
        torch.Tensor: Zero-padded or cropped result.
    """

    assert len(input.shape) == len(
        oshape
    ), "Input and output must have same number of dimensions."

    ishape = input.shape

    if ishape == oshape:
        return input

    ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape, oshape)]
    oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape, oshape)]

    copy_shape = [
        min(i - si, o - so) for i, si, o, so in zip(ishape, ishift, oshape, oshift)
    ]

    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = torch.zeros(oshape, dtype=input.dtype, device=input.device)
    output[oslice] = input[islice]

    return output


"""
Image visualiziation-related Utils (moved from visualize_image.py)
"""


def postprocess_cplx_sample(image):
    """
    Convert [..., H, W, C] image to complex image for viewing

    Assumes:
        - First 2 channels are [real, imaginary] component
        - input in range [-1, 1]

    Returns
        - [..., H, W] magnitude and phase images, scaled [0,255]
        for viewing as PIL image
    """
    im_cplx = image[..., 0] + 1j * image[..., 1]

    # magnitude
    mag = np.clip(np.abs(im_cplx), 0, 1)
    mag = normalize_image_to_uint8(mag)

    # phase
    phase = (np.angle(im_cplx) + np.pi) / (2 * np.pi)
    phase = normalize_image_to_uint8(phase)

    return mag, phase


def postprocess_magn_sample(image):
    """
    Convert [..., H, W, C] image to magnitude by averaging over
    Channel dimension

    Assumes:
        - input clipped and scaled [0,1] already

    Returns
        - [..., H, W] magnitude images scalled [0,255] for viewing
        as PIL image
    """

    image = image.mean(axis=-1)
    image = normalize_image_to_uint8(image)

    return image


def normalize_image_to_uint8(image):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    draw_img = image
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 0.1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def visualize_2d_image(image):
    """
    Prepare a 2D image for visualization.
    Args:
        image: image numpy array, sized (H, W)
    """
    image = convert_to_numpy(image)
    # draw image
    draw_img = normalize_image_to_uint8(image)
    draw_img = np.stack([draw_img, draw_img, draw_img], axis=-1)
    return draw_img


def to_avg_gray(image, rescale=False):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    image = np.mean(image, axis=-1).astype(np.uint8)

    if rescale:
        # rescale image to [0,1]
        image = image - np.min(image)
        image = image / np.max(image)

    image = np.stack([image, image, image], axis=-1)
    return image


def to_abs_img(image):
    image = np.asarray(image)
    # first two channels are real and imaginary
    cmplex = np.abs(image[0] + 1j * image[1])
    # third channel is abs
    image = np.stack([cmplex, image[2]], axis=-1)
    image = np.mean(image, axis=-1).astype(np.uint8)
    image = np.stack([image, image, image], axis=-1)
    return image


def shift_m1_1_to_0_1(x):
    """
    Shifts the input tensor values from the range [-1, 1] to the range [0, 1].

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Tensor with values shifted to the range [0, 1].
    """

    maxs = torch.max(rearrange(x, "... h w -> ... (h w)"), dim=-1)[0]
    mins = torch.min(rearrange(x, "... h w -> ... (h w)"), dim=-1)[0]

    x = (x - mins[..., None, None]) / (maxs[..., None, None] - mins[..., None, None])

    return x


def convert_to_numpy(
    x: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
) -> np.ndarray:
    """
    Convert input data to numpy array.
    Args:
        x: input data to convert. Can be one of:
            - numpy array
            - torch tensor
            - list of numpy arrays or torch tensors
    """
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.cpu().numpy()
    if isinstance(x, list):
        if torch.is_tensor(x[0]):
            return torch.stack(x).cpu().numpy()
        else:
            return np.stack(x)

    raise ValueError(f"Unsupported type {type(x)}")


"""
Recon Util
"""


def compress_coils(ksp_mat: torch.Tensor, new_coil_size: Union[int, float], *args):
    # Move to torch
    np_flag = type(ksp_mat) is np.ndarray
    ksp_mat = np_to_torch(ksp_mat)

    # Estimate coil subspace
    n_coil = ksp_mat.shape[0]
    u, s, vt = torch.linalg.svd(ksp_mat.reshape((n_coil, -1)), full_matrices=False)
    if new_coil_size is None:
        new_coil_size = n_coil
    elif type(new_coil_size) is float:
        cmsm = torch.cumsum(s, dim=0)
        n_coil = torch.argwhere(cmsm > new_coil_size * cmsm[-1]).flatten()[0].item()
    elif type(new_coil_size) is int:
        n_coil = new_coil_size
    n_coil = max(n_coil, 1)
    coil_subspace = u[:, :n_coil]  # / s[:n_coil]
    coil_subspace.imag *= -1

    comps = []
    for arg in args:
        arg_torch = np_to_torch(arg)
        arg_compressed = einsum(arg_torch, coil_subspace, "nc ..., nc nc2 -> nc2 ...")
        if np_flag:
            arg_compressed = torch_to_np(arg_compressed)
        comps.append(arg_compressed)

    if len(comps) == 0:
        return coil_subspace

    else:
        return [coil_subspace] + comps


def normalize(shifted, target, ofs=True, mag=False):
    """
    Assumes the following scaling/shifting offset:

    shifted = a * target + b

    solves for a, b and returns the corrected data

    Parameters:
    -----------
    shifted : torch.Tensor
        data to be corrected
    target : torch.Tensor
        reference data
    ofs : bool
        include b offset in the correction
    mag : bool
        use magnitude of data for correction

    Returns:
    --------
    torch.Tensor
        corrected data
    """

    try:
        if mag:
            col1 = torch.abs(shifted).flatten()
            y = torch.abs(target).flatten()
        else:
            col1 = shifted.flatten()
            y = target.flatten()

        if ofs:
            col2 = col1 * 0 + 1
            A = torch.stack([col1, col2]).T
            a, b = torch.linalg.lstsq(A, y, rcond=None)[0]
        else:
            b = 0
            a = torch.linalg.lstsq(
                col1[
                    None,
                ].T,
                y,
                rcond=None,
            )[0]

        out = a * shifted + b
    except Exception as e:
        print(f"Normalize Failed: {e}")
        out = shifted

    return out


def np_to_torch(*args) -> Union[torch.Tensor, tuple[torch.Tensor]]:
    """
    Converts numpy arrays to torch tensors,
    preserving device and dtype (uses CUPY)

    Parameters:
    -----------
    args : tuple
        numpy arrays to convert

    Returns:
    --------
    ret_args : tuple
        torch tensors
    """
    ret_args = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            ret_args.append(torch.as_tensor(arg))
        elif isinstance(arg, cp.ndarray):
            ret_args.append(torch.as_tensor(arg, device=torch.device(int(arg.device))))
        elif isinstance(arg, torch.Tensor):
            ret_args.append(arg)
        else:
            ret_args.append(None)

    if len(ret_args) == 1:
        ret_args = ret_args[0]

    return ret_args


def torch_to_np(
    *args,
) -> Union[np.ndarray, tuple[np.ndarray], cp.ndarray, tuple[cp.ndarray]]:
    """
    Converts torch tensors to numpy arrays,
    preserving device and dtype (uses CUPY)

    Parameters:
    -----------
    args : tuple
        torch tensors to convert

    Returns:
    --------
    ret_args : tuple
        numpy arrays
    """
    ret_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            if arg.is_cuda:
                with cp.cuda.Device(arg.get_device()):
                    ret_args.append(cp.asarray(arg))
            else:
                ret_args.append(arg.numpy())
        elif isinstance(arg, np.ndarray) or isinstance(arg, cp.ndarray):
            ret_args.append(arg)
        else:
            ret_args.append(None)

    if len(ret_args) == 1:
        ret_args = ret_args[0]

    return ret_args


def fftc(input, dim=(-2, -1), norm="ortho"):
    """
    Compute the centered fast Fourier transform along the specified dimensions.

    Args:
        input (torch.Tensor): The input tensor.
        dim (tuple): The dimensions along which to compute the FFT. Default is (-2, -1).
        norm (str): The normalization mode. Default is "ortho".

    Returns:
        torch.Tensor: The output tensor after applying the centered FFT.
    """
    tmp = torch.fft.ifftshift(input, dim=dim)
    tmp = torch.fft.fftn(tmp, dim=dim, norm=norm)
    output = torch.fft.fftshift(tmp, dim=dim)

    return output


def ifftc(input, dim=(-2, -1), norm="ortho"):
    """
    Compute the inverse centered fast Fourier transform (IFFT) along the specified dimensions.

    Args:
        input (torch.Tensor): The input tensor to compute the IFFT on.
        dim (tuple): The dimensions along which to compute the IFFT. Default is (-2, -1).
        norm (str): The normalization mode. Default is "ortho".

    Returns:
        torch.Tensor: The output tensor after applying the IFFT.

    """
    tmp = torch.fft.ifftshift(input, dim=dim)
    tmp = torch.fft.ifftn(tmp, dim=dim, norm=norm)
    output = torch.fft.fftshift(tmp, dim=dim)

    return output


"""
Utils for conditioning
"""


def rescale01(image):
    return (image - image.min()) / (image.max() - image.min())


def sobel_edge_detection(image):
    # Sobel operator kernels
    assert image.ndim == 4, "Input must be (B,C,H,W)"
    assert image.shape[1] == 1, "Input must have 1 channel"
    device = image.device

    sobel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(device)
    )
    sobel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(device)
    )

    edge_x = F.conv2d(image, sobel_x, padding=1)
    edge_y = F.conv2d(image, sobel_y, padding=1)

    edges = torch.sqrt(edge_x**2 + edge_y**2)

    return rescale01(edges)


def laplacian_edge_detection(image, std):
    # form laplacian gaussian kernel
    assert image.ndim == 4, "Input must be (B,C,H,W)"
    assert image.shape[1] == 1, "Input must have 1 channel"
    device = image.device
    N = int(6 * std + 1) if std >= 1 else 7
    x = torch.arange(-N // 2 + 1, N // 2 + 1, dtype=torch.float32)
    y = torch.arange(-N // 2 + 1, N // 2 + 1, dtype=torch.float32)
    x, y = torch.meshgrid(x, y)
    kernel = torch.exp(-(x**2 + y**2) / (2 * std**2)) * (
        1 - (x**2 + y**2) / (2 * std**2)
    )
    kernel *= -1 / (torch.pi * std**4)
    kernel /= kernel.abs().sum()
    kernel = kernel.view(1, 1, N, N).to(device)

    # Apply Laplacian kernel
    edges = F.conv2d(image, kernel, padding=N // 2)

    # Normalize edge regions to 0
    edges = [
        cv2.convertScaleAbs(e.cpu().numpy(), alpha=255 / e.max().item()) for e in edges
    ]
    edges = [torch.from_numpy(e) for e in edges]
    edges = torch.stack(edges, axis=0)

    return rescale01(edges)


def threshold_filter(img, thresh=0.2, hpf=False):
    # low pass filter images in shape (B, C, H, W)
    imgfft = torch.fft.fftn(img, dim=(-2, -1))
    freqs_x = torch.fft.fftfreq(img.shape[-2])
    freqs_y = torch.fft.fftfreq(img.shape[-1])
    Fx, Fy = torch.meshgrid(freqs_x, freqs_y)

    mask = (Fx**2 + Fy**2) > thresh**2
    mask = mask[None, None, ...].repeat(img.shape[0], img.shape[1], 1, 1)

    if hpf:
        mask = ~mask

    imgfft[mask] = 0
    imglpf = torch.fft.ifftn(imgfft, dim=(-2, -1)).abs()

    return rescale01(imglpf)


def create_gaussian_kernel(kernel_size, sigma):
    """Create (1,1,N,N) gaussian kernel"""
    # Create a coordinate grid
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    axs = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    x, y = torch.meshgrid(axs, axs)

    # Calculate the 2D Gaussian kernel
    gaussian_kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize the kernel so that the sum of all elements is 1
    gaussian_kernel /= gaussian_kernel.sum()

    return gaussian_kernel.view(1, 1, kernel_size, kernel_size)


def kernel_filter(image, kernel):
    """Assume input image is (B, 1, H, W) and kernel is (1, 1, N, N)"""
    assert image.ndim == 4, "Input image must be (B, C, H, W)"
    assert image.shape[1] == 1, "Input image must have 1 channel"

    filtered_image = F.conv2d(image, kernel, padding=kernel.shape[-1] // 2)
    return filtered_image


def get_control_function(control_method):
    """
    Get function to take in prior and return (3,H,W) control image with batching support
    """

    if control_method is None or control_method == "full":
        trylogprint(
            "Initializing control preprocessing function with full image as prior."
        )
        condition_func = lambda x: x  # noqa: E731
    elif control_method == "sobel":
        trylogprint(
            "Initializing control preprocessing function with sobel edge prior."
        )
        condition_func = sobel_edge_detection
    elif control_method.startswith("laplacian"):
        std = float(control_method.split("_")[1])
        trylogprint(
            f"Initializing control preprocessing function with laplacian (std={std}) edge prior."
        )
        condition_func = lambda x: laplacian_edge_detection(x, std)  # noqa: E731
    elif control_method.startswith("gaussian"):
        _, K, S = control_method.split("_")
        K, S = int(K), float(S)
        trylogprint(
            f"Initializing control preprocessing function with gaussian blurred (kernel_size={K}, std={S}) edge prior."
        )
        G = create_gaussian_kernel(K, S)
        condition_func = lambda x: kernel_filter(x, G)  # noqa: E731
    else:
        raise ValueError(f"Control method {control_method} not recognized")

    def condition_func_wrapper(input):
        """
        Input expected as (H,W), (C,H,W), or (B,C,H,W).
        If provided, C must be 1, or will take input to be mean over first channel

        Output is (3,H,W) control image (or (B,3,H,W) if input is batched)
        """

        input_ndim = input.ndim

        if input_ndim == 3:
            input = input[None, ...]
        elif input_ndim == 2:
            input = input[None, None, ...]

        if input.shape[1] != 1:
            input = input.mean(dim=1, keepdim=True)

        assert input.ndim == 4, "Input must be (B,C,H,W) or (C,H,W) or (H,W)"

        out = condition_func(input)

        out = torch.cat([out, out, out], dim=1)

        # support for unbatched input
        if input_ndim < 4:
            out = out[0]

        return out

    return condition_func_wrapper


def certainty_weight_map(
    x_avg,
    x_std,
    mask=None,
    method="logistic",
    alpha=10,
    threshold=0.1,
    epsilon=1e-8,
    smooth_sigma=1.0,
):
    """
    Compute a spatially smooth certiainty weight map W from x_avg and x_std.

    Parameters:
        x_avg (np.ndarray): Mean reconstructed image from diffusion model samples.
        x_std (np.ndarray): Per-voxel standard deviation (uncertainty) over reconstructions.
        method (str): Weighting method to use - options: 'inverse', 'logistic', 'exponential'.
        alpha (float): Controls the steepness of the function (used in logistic and exponential methods).
        threshold (float): Uncertainty threshold for logistic function.
        epsilon (float): Small constant to avoid division by zero.
        smooth_sigma (float): Standard deviation for the Gaussian smoothing.

    Returns:
        np.ndarray: The computed and smoothed weight map W.
    """

    device = x_avg.device
    dtype = x_avg.dtype
    x_avg = x_avg.cpu().numpy()
    x_std = x_std.cpu().numpy()

    # Compute relative uncertainty (normalized by the magnitude of x_avg)
    rel_uncert = x_std / (np.abs(x_avg) + epsilon)

    if smooth_sigma > 0:
        rel_uncert = gaussian_filter(rel_uncert, sigma=smooth_sigma)

    # Compute preliminary weights according to the chosen method
    if method == "inverse":
        W = 1.0 / (epsilon + rel_uncert)
    elif method == "logistic":
        W = 1.0 / (1.0 + np.exp(alpha * (rel_uncert - threshold)))
    elif method == "exponential":
        W = np.exp(-alpha * rel_uncert)
    else:
        raise ValueError(
            "Invalid method provided. Choose 'inverse', 'logistic', or 'exponential'."
        )

    # Normalize W to the range [0, 1]
    W = (W - np.min(W)) / (np.max(W) - np.min(W) + epsilon)

    if mask is not None:
        W *= mask.cpu().numpy()
        rel_uncert *= mask.cpu().numpy()

    W = torch.from_numpy(W).to(device).type(dtype)
    rel_uncert = torch.from_numpy(rel_uncert).to(device).type(dtype)
    return W, rel_uncert


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
