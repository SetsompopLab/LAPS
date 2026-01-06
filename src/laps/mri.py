from math import ceil
from typing import Optional, Tuple

import numpy as np
import sigpy.mri as mr
import torch
from loguru import logger
from scipy.ndimage import label

from pips.recon.linops import CartesianSenseLinop

"""
Tools for MRI undersampling simulation and computation.
"""


def get_measurements(
    x: torch.Tensor, forward_model: CartesianSenseLinop, noise_std: float = 0.0
):
    """
    Apply forward model of SENSE Linop with possible noise.
    """
    with torch.no_grad():
        y = forward_model(x)
        y = y + noise_std * (torch.randn_like(y) + 1j * torch.randn_like(y))
        # mask noise in y where we do not enforce data consistency
        y = y * forward_model.mask

    return y


def get_sim_maps(Nc: int, im_size: Tuple = (256, 256)) -> torch.Tensor:
    """
    Generate coil sensitivity maps for MRI simulation.

    Args:
        Nc (int): Number of coils.
        im_size (Tuple): Size of the image (height, width).

    Returns:
        torch.Tensor: Coil sensitivity maps of shape (Nc, *im_size).
    """
    mps = mr.birdcage_maps((Nc, *im_size), r=1.25).astype(np.complex64)
    return torch.from_numpy(mps)


def get_sim_mask(
    im_size: Tuple, mask_type: str = "poisson", accel_factor: float = 1.0, **kwargs
) -> torch.Tensor:

    if accel_factor == 1.0:
        return torch.ones(im_size)

    if mask_type == "cartesian":
        # Heuristic to make mask have fewer lines with higher accel
        if "ncalib" in kwargs:
            ncalib = kwargs.pop("ncalib")
        else:
            ncalib = int(64 / max(accel_factor, 4))
        return cartesian_mask(im_size, accel_factor, ncalib=ncalib, **kwargs)
    elif mask_type == "poisson":
        # number of calibration lines
        if "ncalib" in kwargs:
            ncalib = kwargs.pop("ncalib")
        else:
            ncalib = max(25, int(0.1 * im_size[0]))
        return poisson_mask(im_size, accel_factor, ncalib=ncalib, **kwargs)
    else:
        raise ValueError("mask_type must be poisson or cartesian")


def poisson_mask(
    im_size: Tuple, accel_factor: float, ncalib: Optional[int] = None, **kwargs
) -> torch.Tensor:
    """
    Generate a Poisson mask for undersampling.

    Args:
        im_size (Tuple): Size of the image (height, width).
        accel_factor (float): Acceleration factor for undersampling.

    Returns:
        torch.Tensor: Poisson mask for undersampling.
    """
    if ncalib is None:
        ncalib = max(21, int(0.1 * im_size[0]))
    calib = tuple([ncalib] * len(im_size))
    tol = 0.1 if accel_factor < 25 else 1
    mask = mr.poisson(im_size, accel_factor, calib=calib, dtype=float, tol=tol)
    return torch.from_numpy(mask)


def cartesian_mask(
    im_size: Tuple,
    accel_factor: float,
    ncalib: int,
    sampling: str = "vds",  # "uniform" or "VDS"
    vd_factor: Optional[float] = 0.8,
    max_iter: Optional[int] = 200,
    criteria: Optional[str] = "norm",
    **kwargs,
) -> torch.Tensor:
    """
    Generate a Cartesian mask for undersampling.

    Args:
        im_size (Tuple): Size of the image (height, width).
        accel_factor (float): Acceleration factor for undersampling.
        calib (float): Calibration factor (percentage of calibration lines in the middle of k-space).

    Returns:
        np.ndarray: Cartesian mask for undersampling.

    Raises:
        AssertionError: If the number of calibration lines is greater than or equal to the number of lines to sample.
    """

    lines_to_sample = int(im_size[0] / accel_factor)
    calib_lines = ncalib
    assert (
        calib_lines < lines_to_sample
    ), "Calibration lines must be less than lines to sample"
    assert criteria in ["max", "norm"], "criteria must be max or norm"

    N = im_size[0]
    mask = np.zeros(im_size)
    center = N // 2

    # Calibration
    mask[int(center - (calib_lines // 2)) : int(center + (calib_lines // 2)), :] = 1
    remaining_lines = [
        y
        for y in range(N)
        if y not in range(center - (calib_lines // 2), center + (calib_lines // 2))
    ]
    remaining_lines_to_sample = lines_to_sample - calib_lines

    # Try to find sampling pattern that minimizes some gap metric and keeps norm on all distances as small as possible
    max_diff = accel_factor * 2  # heuristic-ish
    rl_inds = np.arange(center - (calib_lines // 2), center + (calib_lines // 2))
    ky = np.array(remaining_lines) - center

    if sampling == "uniform":
        density = None
    elif sampling == "vds":
        density = (1 - (1.8 / N) * np.abs(ky)) ** vd_factor
        density /= np.sum(density)
    else:
        raise ValueError("sampling must be uniform or VDS")

    sampled_best = None
    best_diff = np.inf
    for i in range(max_iter):
        sampled = np.random.choice(
            remaining_lines, remaining_lines_to_sample, replace=False, p=density
        )
        sampled_full = np.sort(
            np.concatenate(
                [
                    np.array([-1, N + 1]),
                    sampled.flatten(),
                    rl_inds.flatten(),
                ]
            )
        )

        # Criteria: max or norm
        if criteria == "max":
            diff = np.max(np.abs(np.diff(sampled_full)))
        elif criteria == "norm":
            diff = np.linalg.norm(np.abs(np.diff(sampled_full)))

        if diff <= max_diff:
            best_diff = diff
            sampled_best = sampled
            break
        elif diff < best_diff:
            best_diff = diff
            sampled_best = sampled

    mask[sampled_best, :] = 1

    return torch.from_numpy(mask)


def is_1d_pe_dim(mask: torch.Tensor, dim: int) -> bool:
    """
    Check if 2D mask has 1D sampling pattern along specified axis.

    Args:
        mask: 2D tensor mask
        dim: Dimension to check (0 or 1)

    Returns:
        bool: True if mask has 1D pattern along specified dimension
    """

    # trim outsides
    mask = mask.to(torch.float32)
    inds = torch.where(mask.sum(dim=dim) > 0)[0]
    ts, te = inds.min().item(), inds.max().item()
    if dim == 0:
        mask = mask[:, ts:te]
    elif dim == 1:
        mask = mask[ts:te, :]
    return (mask.sum(dim=dim) - mask.sum(dim=dim).mean()).max() < 1e-1


def extract_center_mask_radial(
    mask: torch.Tensor, r: int
) -> Tuple[torch.Tensor, float, int]:
    """
    Extract a circular center region of a binary mask with radius r.

    Args:
        mask: Input mask tensor of shape (H, W)
        r: Radius for center extraction (in pixels)

    Returns:
        Tuple of (center_mask, sampled_ratio, n_center_points)
    """
    device = mask.device
    dtype = mask.dtype
    H, W = mask.shape
    cy, cx = H // 2, W // 2

    # Create grid of coordinates
    y = torch.arange(H, device=device).unsqueeze(1)
    x = torch.arange(W, device=device).unsqueeze(0)
    dist_sq = (y - cy) ** 2 + (x - cx) ** 2
    center_mask = (dist_sq <= r**2).to(dtype)

    # Compute sampled ratio
    n_center = center_mask.sum().item()
    n_sampled = (mask * center_mask).sum().item()
    sampled_ratio = n_sampled / n_center if n_center > 0 else 0.0

    return center_mask, sampled_ratio, int(n_center)


def extract_fully_sampled_center_np(mask: np.ndarray) -> np.ndarray:
    """
    Given a binary mask with a fully-sampled center region (usually a contiguous
    blob) and some scattered outer samples, return a new mask that is just that
    center region.

    Parameters
    ----------
    mask : np.ndarray
        Any N-D array of booleans or 0/1 ints where True/1 indicates a sampled
        point.

    Returns
    -------
    center_mask : np.ndarray
        A boolean mask of the same shape as `mask` where only the largest
        connected component is True.
    """
    # ensure binary
    binary = mask.astype(bool)

    # label all connected components (use full connectivity across all dims)
    structure = np.ones((3,) * binary.ndim, dtype=int)
    labels, num_components = label(binary, structure=structure)

    if num_components == 0:
        # no positives at all
        return np.zeros_like(binary)

    # count up size of each component (0 is background)
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # ignore background

    # pick the largest component
    largest_label = counts.argmax()

    # return a mask where only that component is True
    center_mask = labels == largest_label
    return center_mask


def extract_center_mask(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Extract the center of the mask with radius r

    Returns a mask with the center region and the number of center points
    """
    mask_center = extract_fully_sampled_center_np(mask.cpu().numpy())
    mask_center = torch.from_numpy(mask_center).to(mask.device).to(mask.dtype)

    return mask_center


def extract_center_kernel(
    mask: torch.Tensor, kernel_size: Tuple[int, int] = (3, 3)
) -> torch.Tensor:
    """
    Given a kernel size, extract the fully sampled center region of the mask
    for which every point in this region is sampled by the kernel size.

    Returns the mask of the center region and a slice to get the center region surrounding it.
    """

    k = torch.ones(kernel_size, dtype=torch.float32, device=mask.device)
    sum = k.sum().item()
    mask_center = extract_center_mask(mask).to(torch.float32)
    mask_center = torch.nn.functional.conv2d(
        mask_center[
            None,
            None,
        ],
        k[
            None,
            None,
        ],
        padding=(kernel_size[0] // 2, kernel_size[1] // 2),
    )[0, 0]
    mask_center = mask_center == sum

    # How to access the center region for calib extraction
    xval = torch.where(mask_center.any(dim=1))[0]
    yval = torch.where(mask_center.any(dim=0))[0]
    cent_slc = (
        slice(
            xval[0].item() - kernel_size[0] // 2,
            xval[-1].item() + 1 + kernel_size[0] // 2,
        ),
        slice(
            yval[0].item() - kernel_size[1] // 2,
            yval[-1].item() + 1 + kernel_size[1] // 2,
        ),
    )
    return mask_center, cent_slc


def extract_mask_bounds(
    mask: torch.Tensor, kernel_size: Tuple[int, int] = (3, 3)
) -> torch.Tensor:
    """
    Given a kernel size, extract the fully sampled center region of the mask
    for which every point in this region is sampled by the kernel size.

    Returns the mask of the center region and a slice to get the center region surrounding it.
    """

    k = torch.ones(kernel_size, dtype=torch.float32, device=mask.device)
    mask_outer = torch.nn.functional.conv2d(
        mask[
            None,
            None,
        ].float(),
        k[
            None,
            None,
        ],
        padding=(kernel_size[0] // 2, kernel_size[1] // 2),
    )[0, 0]
    mask_outer = mask_outer > 0

    return mask_outer


def extract_longest_continuous(inds: torch.Tensor) -> torch.Tensor:
    """
    Given a 1D tensor of integer indices, returns the longest continuous
    (step=1) subsequence. If no continuous run of length >= 2 is found,
    returns an empty tensor.
    """
    # Make sure it's 1D, sorted, unique
    if inds.numel() == 0:
        return inds.clone()
    sorted_inds = torch.unique(inds).sort().values

    best_start = sorted_inds[0].item()
    best_len = 1
    curr_start = best_start
    curr_len = 1

    # walk through sorted indices
    for i in range(1, sorted_inds.size(0)):
        if sorted_inds[i].item() == sorted_inds[i - 1].item() + 1:
            curr_len += 1
        else:
            # break in sequence
            if curr_len > best_len:
                best_start, best_len = curr_start, curr_len
            curr_start = sorted_inds[i].item()
            curr_len = 1

    # final check at end
    if curr_len > best_len:
        best_start, best_len = curr_start, curr_len

    # if the longest run is only length 1, consider "no continuous run"
    if best_len < 2:
        return sorted_inds.new_empty((0,))

    # build and return the result
    return torch.arange(
        best_start,
        best_start + best_len,
        dtype=sorted_inds.dtype,
        device=sorted_inds.device,
    )


def estimate_accel(
    mask: torch.Tensor, cent_slc: Optional[slice] = None
) -> Tuple[int, int, bool]:
    """
    Estimate the acceleration factor for each dimension of mask.
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D")

    # Try to determine if the sampling pattern is VD
    is_vd = False
    try:
        y_samps, z_samps = [], []
        ofs_max = min(mask.shape[0] // 4, mask.shape[1] // 4)
        ofs_list = np.unique(np.linspace(-ofs_max, ofs_max, num=15, dtype=int)).tolist()
        for ofs in ofs_list:

            yd = torch.where(mask[:, mask.shape[1] // 2 + ofs] > 0)[0]
            if len(yd) > 3:
                y_samps.append(yd[1:-1].diff().unique())

            zd = torch.where(mask[mask.shape[0] // 2 + ofs, :] > 0)[0]
            if len(zd) > 3:
                z_samps.append(zd[1:-1].diff().unique())

        y_samps = torch.cat(y_samps, dim=0).unique()
        z_samps = torch.cat(z_samps, dim=0).unique()
        samps_ints = torch.cat([y_samps, z_samps], dim=0).unique()
        samps_ints = samps_ints[samps_ints < 8]

        if samps_ints.numel() >= 4:
            is_vd = True
    except Exception as e:
        logger.warning(f"Error estimating VD sampling pattern: {e}")
        is_vd = False

    mask2 = mask.clone()
    if cent_slc is None:
        cent_slc = extract_center_kernel(mask, (3, 3))[1]

    mask2[cent_slc] = 0

    RR = []
    for d in range(mask2.ndim):
        samps_d = torch.where(mask2.sum(dim=d) > 0)[0].diff()
        samps_d = samps_d[samps_d > 0]
        samps_d = samps_d[samps_d < 10]
        if samps_d.numel() == 0:
            RR.append(1)
            continue
        us_types = samps_d.unique()
        if us_types.numel() == 1:
            RR.append(us_types.item())
            continue
        elif us_types.numel() > 1:
            # remove R=1
            us_types = us_types[us_types > 1]
            R_possible = us_types.min().item()
            if (samps_d == R_possible).sum() > 10:
                RR.append(R_possible)
                continue
            else:
                RR.append(1)
        RR.append(samps_d.mode().values.item())
    Rz, Ry = RR

    return Ry, Rz, is_vd


def determine_undersampling_type(mask: torch.Tensor) -> str:
    """
    Determine the undersampling type based on the mask.
    """

    Rro, Rpe, is_vd = estimate_accel(mask)

    accel_num_dim = 0
    if Rro > 1:
        accel_num_dim += 1
    if Rpe > 1:
        accel_num_dim += 1

    if is_vd or accel_num_dim == 2:
        return "2D"
    elif accel_num_dim == 1:
        return "1D_LR" if Rro == 1 else "1D_UD"
    else:
        return "FS"


def set_retrospective_undersample_parameters(
    R_1D: Optional[int] = None,
    R_2D: Optional[int] = None,
    vd_factor_1d: float = 0.8,
    vd_factor_2d: float = 1.5,
):
    """
    Set retrospective undersampling parameters.

    If using 1D undersampling, R_1D must be specified.
    If NOT using 1D undersampling, R_1D must be None.

    Same logic for R_2D.
    """

    has_1d = R_1D is not None
    has_2d = R_2D is not None

    assert has_1d or has_2d, "Must specify at least one of R_1D or R_2D"

    # how to deal with FS
    if has_1d and has_2d:
        prob_2D_on_FS = 0.5
    elif has_1d:
        prob_2D_on_FS = 0.0
    else:
        prob_2D_on_FS = 1.0

    # 1D Center region
    if not has_1d:
        R_1D = 1

    if R_1D <= 1:
        N_fs_max_1d = 25
    if R_1D <= 3:
        N_fs_max_1d = 21
    else:
        N_fs_max_1d = 15

    if not has_2d:
        R_2D = 1

    if R_2D <= 1:
        N_fs_max_2d = 25
    elif R_2D <= 15:
        N_fs_max_2d = 14
    else:
        N_fs_max_2d = 10

    kwargs_1D = {
        "N_fs_max": N_fs_max_1d,
        "sampling_type": "vds",
        "vd_factor": vd_factor_1d,
    }
    kwargs_2D = {
        "N_fs_max": N_fs_max_2d,
        "sampling_type": "vds",
        "vd_factor": vd_factor_2d,
    }
    return {
        "R_1D": R_1D,
        "R_2D": R_2D,
        "kwargs_1D": kwargs_1D,
        "kwargs_2D": kwargs_2D,
        "prob_2D_on_FS": prob_2D_on_FS,
    }


def retrospective_undersample_mask(
    mask: torch.Tensor,
    R_1D: Optional[int] = None,
    R_2D: Optional[int] = None,
    kwargs_1D: Optional[dict] = None,
    kwargs_2D: Optional[dict] = None,
    prob_2D_on_FS: float = 0.5,
    undersampling_type: Optional[str] = None,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Wrapper for mask undersampling. Will detect if mask is 1D or 2D, and perform the appropriate undersampling.

    Can give separate R and kwargs for 2D and 1D solutions, otherwise will just assume R_1D and kwargs_2D are what should be used.
    """

    assert (
        R_1D is not None or R_2D is not None
    ), "Must specify at least one of R_1D or R_2D"

    assert mask.ndim == 2, "Expected 2D mask input"

    mask = mask.abs() > 0

    if undersampling_type is not None:
        assert undersampling_type in ["1D_UD", "1D_LR", "2D", "FS"], (
            "undersampling_type must be one of: " "'1D_UD', '1D_LR', '2D', 'FS'"
        )
    else:
        undersampling_type = determine_undersampling_type(mask)

    mask = mask.to(torch.float32)

    if seed is not None:
        np.random.seed(seed)

    # First check if fully sampled
    dim_0_1d_on_fs = False
    dim_1_1d_on_fs = False
    dim_2d_on_fs = False
    fs = False
    if undersampling_type == "FS":
        fs = True
        use_2D = np.random.rand() < prob_2D_on_FS
        if not use_2D:
            if np.random.rand() < 0.5:
                dim_0_1d_on_fs = True
                mask_str = "1D along dim 0"
            else:
                dim_1_1d_on_fs = True
                mask_str = "1D along dim 1"
        else:
            dim_2d_on_fs = True
            mask_str = "2D"

        if verbose:
            logger.info(
                f"Detected fully sampled mask - using {mask_str} undersampling."
            )

    # check if just 1D
    if (not dim_2d_on_fs) and (dim_0_1d_on_fs or (undersampling_type == "1D_UD")):
        if verbose and not fs:
            logger.info(f"Detected 1D mask with phase encodes along dim 0.")

        assert R_1D is not None, "R_1D must be specified for 1D undersampling"

        mask = mask.T  # put pe_dim on 1st axis
        mask = retrospective_undersample_cartesian_mask(
            mask, R_1D, verbose=verbose, **(kwargs_1D or {})
        )
        mask = mask.T

    elif (not dim_2d_on_fs) and (dim_1_1d_on_fs or (undersampling_type == "1D_LR")):
        assert R_1D is not None, "R_1D must be specified for 1D undersampling"

        if verbose and not fs:
            logger.info(f"Detected 1D mask with phase encodes along dim 1.")
        mask = retrospective_undersample_cartesian_mask(
            mask, R_1D, verbose=verbose, **(kwargs_1D or {})
        )
    else:
        if verbose and not fs:
            logger.info(
                f"Detected 2D poisson mask with phase encodes along dim 0 and 1."
            )

        if kwargs_2D is not None:
            kwargs = kwargs_2D
        else:
            kwargs = kwargs_1D or {}
        if R_2D is not None:
            R = R_2D
        else:
            assert (
                R_1D is not None
            ), "R_1D must be specified for 2D undersampling if R_2D not specified."
            R = R_1D
        mask = retrospective_undersample_poisson_mask(
            mask, R, verbose=verbose, **kwargs
        )

    return mask


def retrospective_undersample_poisson_mask(
    mask: torch.Tensor,
    R: int,
    N_fs_max: int = 12,
    sampling_type: str = "vds",
    vd_factor: Optional[float] = 0.8,
    verbose: bool = False,
    calc_rate_in_bounds_only: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Ensure efficient implementation
    """
    # Determine fully sampled radius via binary search
    H, W = mask.shape
    max_r = min(H, W) // 2
    lo, hi = 0, max_r

    def _is_full(r):
        _, sampled_ratio, _ = extract_center_mask_radial(mask, r)
        return sampled_ratio == 1.0

    # binary search for largest r where sampled_ratio == 1.0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _is_full(mid):
            lo = mid
        else:
            hi = mid - 1
    N_fs_in = lo
    N_fs = min(N_fs_in, N_fs_max)

    # get final fully sampled center mask and point count
    fs_mask, _, N_fs_points = extract_center_mask_radial(mask, N_fs)

    # compute total points and desired output points
    N_in = mask.sum().item()
    if calc_rate_in_bounds_only:
        N_full = extract_mask_bounds(mask).sum().item()
    else:
        N_full = H * W
    R_in = N_full / N_in
    N_out = int(ceil(N_full / R))
    N_samp = max(0, N_out - N_fs_points)

    # Edge cases
    if N_out >= N_in:
        logger.warning(
            f"Accel factor {R} requested is lower than input {R_in}, cannot reduce. Returning input mask"
        )
        return mask
    if N_samp == 0:
        logger.warning(
            f"Accel factor requested is lower than input, cannot reduce. Returning input mask"
        )
        return mask
    if verbose:
        logger.info(
            f"Input: R={R_in}. {N_in} sampled lines. Output: R={R}: {N_out} lines with r={N_fs} fully sampled radius."
        )

    # mask of sampleable locations (outside FS region)
    sampleable = mask.bool() & ~fs_mask.bool()
    coords = torch.argwhere(sampleable).to(dtype=torch.float32)

    center = torch.tensor([H / 2, W / 2], device=mask.device, dtype=torch.float32)[
        None,
    ]
    dists = (coords - center).norm(dim=1)
    order = torch.argsort(dists)
    coords = coords[order]
    dists = dists[order]
    dists = dists / dists.max()

    # compute selection probabilities
    Np = len(dists)
    if sampling_type == "uniform":
        probs = torch.ones(Np, device=mask.device) / Np
    else:  # vds
        # linearly decreasing weight by radius
        vdf = torch.tensor([vd_factor], device=mask.device)
        probs = torch.pow(1.0 - (0.8 * dists), vdf)
        probs = probs / probs.sum()

    # choose sample indices
    chosen = torch.multinomial(probs, num_samples=N_samp, replacement=False)
    chosen_coords = coords[chosen]

    # set to 1 at chosen coords and at fs mask
    mask_out = fs_mask.to(dtype=torch.float32)
    mask_out[chosen_coords[:, 0].long(), chosen_coords[:, 1].long()] = 1.0

    return mask_out


def retrospective_undersample_cartesian_mask(
    mask: torch.Tensor,
    R: int,
    N_fs_max: int = 12,
    n_samp_iters: int = 100,
    sampling_type: str = "vds",
    vd_factor: Optional[float] = 0.8,
    verbose: bool = False,
    **kwargs,
) -> torch.Tensor:

    sampled_lines = (mask.sum(dim=0) > 0).to(torch.float32)
    inds_samp_in = torch.where(sampled_lines > 0)[0]

    # Input undersampling
    N = len(sampled_lines)
    N_in = sampled_lines.sum()
    R_in = N / N_in

    # find fs region
    gaps = torch.diff(inds_samp_in, prepend=inds_samp_in[0:1] * 100)
    fs_locs = torch.sort(extract_longest_continuous(inds_samp_in[gaps == 1])).values

    if len(fs_locs) == 0:
        logger.warning("No fully sampled region found in input mask.")
        N_fs_out = 0
        inds_samp = inds_samp_in
    else:
        N_fs_in = (fs_locs[-1] - fs_locs[0] + 1).item()
        N_fs_out = min(N_fs_in, N_fs_max)
        if N_fs_out < N_fs_in:
            # trim fs_width_locs to new region
            trim = (N_fs_in - N_fs_out) // 2
            fs_locs = fs_locs[trim:-trim]

        # inds we can sample from, maintaining fully sampled center
        inds_samp = inds_samp_in[
            (inds_samp_in > fs_locs[-1]) | (inds_samp_in < fs_locs[0])
        ]

    # resample.
    N_out = int(ceil(N / R))
    N_samp_out = int(max(0, N_out - N_fs_out))

    if N_samp_out == 0:
        logger.warning(
            f"Accel factor {R} requested is lower than input {R_in}, cannot reduce. Returning input mask"
        )
        return mask

    if sampling_type == "uniform":
        probs = torch.ones_like(inds_samp).to(torch.float32)
        probs /= probs.sum()
    elif sampling_type == "vds":
        base = ((1 - (1.8 / N) * (inds_samp - N // 2).abs())).to(torch.float32)
        vde = torch.tensor([vd_factor], device=inds_samp.device, dtype=torch.float32)
        probs = torch.pow(base, vde)
        probs /= probs.sum()
    else:
        raise ValueError("sampling_type must be uniform or vds")

    # Sample a bunch of times and pick the one with smallest max gap
    bounds = torch.tensor([-1, N]).to(inds_samp.dtype).to(inds_samp.device)
    bounds = bounds[None, :].repeat(n_samp_iters, 1)
    fs_locs_batched = fs_locs[None, :].repeat(n_samp_iters, 1)
    probs = probs[None, :].repeat(n_samp_iters, 1)

    inds_samp_batched = inds_samp[
        None,
    ].repeat(n_samp_iters, 1)
    inds_samp_out_locs = torch.multinomial(
        probs, num_samples=N_samp_out, replacement=False
    )
    inds_samp_out = inds_samp_batched[
        torch.arange(n_samp_iters)[:, None], inds_samp_out_locs
    ]

    inds_samp_full = torch.cat([inds_samp_out, fs_locs_batched, bounds], dim=1)
    inds_samp_full = torch.sort(inds_samp_full, dim=1)[0]

    max_gaps = torch.diff(inds_samp_full, dim=1).amax(dim=1)
    best_idx = max_gaps.argmin()
    best_samp = inds_samp_out[best_idx, :]

    inds_out = torch.sort(torch.cat([fs_locs, best_samp]))[0]

    R_out = N / len(inds_out)
    inds_remove = [i for i in range(N) if i not in inds_out]
    mask_out = mask.clone()
    mask_out[:, inds_remove] = 0

    if verbose:
        logger.info(
            f"Doing retrospective cartesian undersampling with type: {sampling_type}. Read {N} lines, has {N_in} sampled for R={R_in}"
        )
        logger.info(
            f"Sampling {N_out} lines, with {N_fs_out} lines fully sampled (detected {N_fs_in} fs lines of input mask)."
        )
        logger.info(f"Output mask: R desired = {R}, R effective = {R_out}")

    return mask_out
