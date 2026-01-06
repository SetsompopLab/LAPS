import gc
import time
from typing import Optional, Tuple

import sigpy as sp
import torch
from einops import einsum, rearrange
from tqdm import tqdm

from ..utils import ifftc, np_to_torch, torch_resize, torch_to_np
from .algs import power_method_matrix

__all__ = [
    "acs_from_ksp",
    "csm_from_espirit",
]


def acs_from_ksp(ksp_cart: torch.Tensor, Nacs, ndim=2) -> torch.Tensor:
    """
    Given kspace data in cartesian coordinates, extract the ACS region.

    Parameters
    ---------
    ksp_cart: array
        Kspace data in cartesian coordinates, with shape (..., Nc, X, Y)
    Nacs: int
        Size of ACS region

    Returns
    --------
    acs: array
        ACS region of kspace data in shape (..., Nc, Nacs, Nacs)
    """

    if ndim == 2:
        X, Y = ksp_cart.shape[-2:]

        acs = ksp_cart[
            ...,
            :,
            X // 2 - Nacs // 2 : X // 2 + Nacs // 2,
            Y // 2 - Nacs // 2 : Y // 2 + Nacs // 2,
        ]

    elif ndim == 3:
        X, Y, Z = ksp_cart.shape[-3:]

        acs = ksp_cart[
            ...,
            :,
            X // 2 - Nacs // 2 : X // 2 + Nacs // 2,
            Y // 2 - Nacs // 2 : Y // 2 + Nacs // 2,
            Z // 2 - Nacs // 2 : Z // 2 + Nacs // 2,
        ]

    else:
        raise ValueError("Invalid ndim. Must be 2 or 3.")

    return acs


def csm_from_espirit(
    ksp_cal: torch.Tensor,
    im_size: tuple,
    thresh: float = 0.02,
    kernel_width: int = 6,
    crp: Optional[float] = None,
    sets_of_maps: int = 1,
    max_iter: int = 100,
    verbose: bool = True,
    use_cupy_for_blocks: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Copy of Daniel's sigpy but with changes for memory efficiency, which is a
    copy of sigpy implementation of ESPIRiT calibration, but in torch:
    Martin Uecker, ... ESPIRIT - An Eigenvalue Approach to Autocalibrating Parallel MRI

    Parameters:
    -----------
    ksp_cal : torch.Tensor
        Calibration k-space data with shape (ncoil, *cal_size)
    im_size : tuple
        output image size
    thresh : float
        threshold for SVD nullspace
    kernel_width : int
        width of calibration kernel
    crp : float
        output mask based on copping eignevalues
    sets_of_maps : int
        number of sets of maps to compute
    max_iter : int
        number of iterations to run power method
    verbose : bool
        toggles progress bar

    Returns:
    --------
    mps : torch.Tensor
        coil sensitivity maps with shape (ncoil, *im_size)
    eigen_vals : torch.Tensor
        eigenvalues with shape (*im_size)
    """

    # Consts
    img_ndim = len(im_size)
    num_coils = ksp_cal.shape[0]
    device = ksp_cal.device

    # TODO torch this part
    # Get calibration matrix.
    # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
    if use_cupy_for_blocks:
        ksp_cal_sp = torch_to_np(ksp_cal)
        dev = sp.get_device(ksp_cal_sp)
        with dev:
            mat = sp.array_to_blocks(ksp_cal_sp, [kernel_width] * img_ndim, [1] * img_ndim)
            mat = mat.reshape([num_coils, -1, kernel_width**img_ndim])
            mat = mat.transpose([1, 0, 2])
            mat = mat.reshape([-1, num_coils * kernel_width**img_ndim])
        mat = np_to_torch(mat)
    else:
        # bug on cnidl where I can't use cupy
        ksp_cal_sp = ksp_cal.cpu().numpy()
        mat = sp.array_to_blocks(ksp_cal_sp, [kernel_width] * img_ndim, [1] * img_ndim)
        mat = mat.reshape([num_coils, -1, kernel_width**img_ndim])
        mat = mat.transpose([1, 0, 2])
        mat = mat.reshape([-1, num_coils * kernel_width**img_ndim])
        mat = torch.from_numpy(mat).to(device)

    # Perform SVD on calibration matrix
    start = time.perf_counter()
    if verbose:
        print("Computing SVD on calibration matrix: ", end="")
    _, S, VH = torch.linalg.svd(mat, full_matrices=False)
    VH = VH[S > thresh * S.max(), :]
    if verbose:
        end = time.perf_counter()
        print(f"{end - start:.3f}s")

    # memory management
    del mat
    torch.cuda.empty_cache()
    gc.collect()

    # Get kernels
    num_kernels = len(VH)
    kernels = VH.reshape([num_kernels, num_coils] + [kernel_width] * img_ndim)

    # Get covariance matrix in image domain
    AHA = torch.zeros(
        im_size + (num_coils, num_coils), dtype=ksp_cal.dtype, device=device
    )
    for kernel in tqdm(
        kernels, "Computing ESPIRIT covariance matrix", disable=not verbose, leave=False
    ):
        kernel = torch_resize(kernel, (num_coils, *im_size))
        aH = ifftc(kernel, dim=tuple(range(-img_ndim, 0)))
        aH = rearrange(aH, "nc ... -> ... nc 1")
        a = aH.swapaxes(-1, -2).conj()
        AHA += aH @ a
    AHA *= torch.prod(torch.tensor(im_size)).item() / kernel_width**img_ndim

    # cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # Get eigenvalues and eigenvectors
    mps_all = []
    evals_all = []
    for i in range(sets_of_maps):

        # power iterations
        mps, eigen_vals = power_method_matrix(AHA, num_iter=max_iter, verbose=verbose)

        # Phase relative to first map and crop
        mps *= torch.conj(mps[0] / (torch.abs(mps[0]) + 1e-12))
        if crp:
            mps *= eigen_vals > crp

        # Update AHA
        if (
            sets_of_maps > 1
        ):  # becasue this is memory-intensive. TODO: fix this problem in general.
            AHA -= einsum(mps * eigen_vals, mps.conj(), "Cl ..., Cr ... -> ... Cl Cr")
        mps_all.append(mps)
        evals_all.append(eigen_vals)

    if sets_of_maps == 1:
        mps = mps_all[0]
        eigen_vals = evals_all[0]
    else:
        mps = torch.stack(mps_all, dim=0)
        eigen_vals = torch.stack(evals_all, dim=0)

    # final cleanup
    del AHA
    torch.cuda.empty_cache()
    gc.collect()

    return mps, eigen_vals
