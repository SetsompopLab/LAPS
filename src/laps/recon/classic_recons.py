import time
from typing import Optional

import torch
from tqdm import tqdm

from laps.recon.algs import conjugate_gradient, fista
from laps.recon.lacs import ReconSFISTA
from laps.recon.linops import WaveletLinop, linop
from laps.recon.power import power_method_operator
from laps.utils import clear_cache

__all__ = ["CG_SENSE_recon", "FISTA_recon", "LACS_fista_recon"]


def CG_SENSE_recon(
    A: linop,
    ksp: torch.Tensor,
    max_iter: Optional[int] = 15,
    lamda_l2: Optional[float] = 0.0,
    max_eigen: Optional[float] = None,
    tolerance: Optional[float] = 1e-8,
    verbose: Optional[bool] = True,
) -> torch.Tensor:
    """
    Run CG SENSE recon:
    recon = (AHA + lamda_l2I)^-1 AHb

    Parameters:
    -----------
    A : linop
        The linear operator (see linop)
    ksp : torch.Tensor
        k-space data with shape (nc, ...)
    max_iter : int
        max number of iterations for recon algorithm
    lamda_l2 : float
        l2 lamda regularization for SENSE: ||Ax - b||_2^2 + lamda_l2||x||_2^2
    max_eigen : float
        maximum eigenvalue of AHA
    tolerance : float
        tolerance for CG algorithm
    verbose : bool
        Toggles print statements

    Returns:
    --------
    recon : torch.Tensor
        the reconstructed image/volume
    """

    # Consts
    device = ksp.device

    # Estimate largest eigenvalue so that lambda max of AHA is 1
    if max_eigen is None:
        x0 = torch.randn(A.ishape, dtype=torch.complex64, device=device)
        _, max_eigen = power_method_operator(A.normal, x0, verbose=verbose)
        max_eigen *= 1.01

    # Starting with AHb
    y = ksp.type(torch.complex64)
    AHb = A.adjoint(y) / (max_eigen**0.5)
    if max_iter == 0:
        return AHb

    # Clear data (we dont need it anymore)
    y = y.cpu()
    clear_cache()

    # Wrap normal with max eigen
    AHA = lambda x: A.normal(x) / max_eigen

    # Run CG
    recon = conjugate_gradient(
        AHA=AHA,
        AHb=AHb,
        num_iters=max_iter,
        lamda_l2=lamda_l2,
        tolerance=tolerance,
        verbose=verbose,
    )

    return recon


def FISTA_recon(
    A: linop,
    ksp: torch.Tensor,
    proxg: callable,
    max_iter: int = 40,
    max_eigen: Optional[float] = None,
    verbose: Optional[bool] = True,
) -> torch.Tensor:
    """
    Run FISTA recon
    recon = min_x ||Ax - b||_2^2 + g(x)

    Parameters:
    -----------
    A : linop
        The linear operator (see linop)
    ksp : torch.Tensor
        k-space data with shape (nc, nro, npe, ntr)
    proxg : callable
        proximal operator for g(x)
    max_iter : int
        max number of iterations for recon algorithm
    max_eigen : float
        maximum eigenvalue of AHA
    verbose : bool
        Toggles print statements

    Returns:
    --------
    recon : torch.Tensor
        the reconstructed image/volume
    """

    # Consts
    device = ksp.device

    # Estimate largest eigenvalue so that lambda max of AHA is 1
    if max_eigen is None:
        x0 = torch.randn(A.ishape, dtype=torch.complex64, device=device)
        _, max_eigen = power_method_operator(A.normal, x0, verbose=verbose)
        max_eigen *= 1.01

    # Starting with AHb
    y = ksp.type(torch.complex64)
    AHb = A.adjoint(y) / (max_eigen**0.5)

    # Clear data (we dont need it anymore)
    y = y.cpu()
    clear_cache()

    # Wrap normal with max eigen
    AHA = lambda x: A.normal(x) / max_eigen

    # Run fista
    recon = fista(AHA, AHb, proxg, max_iter, verbose=verbose)

    return recon


def LACS_fista_recon(
    A: linop,
    ksp: torch.Tensor,
    x0: torch.Tensor,  # prior image
    lamda1: float = 1e-3,  # range [0.001, 0.9]
    lamda2: float = 1e-3,  # range [0.001, 0.9]
    eps: Optional[float] = 0.1,
    num_iters: int = 5,
    num_fista_iters: int = 40,
    tol: Optional[float] = 1e-6,
    max_eigen: Optional[float] = None,
    wave_name: Optional[str] = "db4",
    verbose: Optional[bool] = True,
    **kwargs,
) -> torch.Tensor:
    r"""
    L1 Wavelet regularized reconstruction.

    Considers the problem

    .. math::
        \min_x \frac{1}{2} \|A x - y \|_2^2 + \lambda \| W1 Ψ x \|_1 + \lambda \| W2 (x - x0) \|_1

    where A is the forward model, Ψ is the wavelet operator, x0 is a prior scan,
    and W1 and W2 are diagonal weight matrices.

    Based on paper: https://arxiv.org/pdf/1407.2602
    """

    device = ksp.device
    y = ksp.clone().to(torch.complex64)

    # normalize inputs to unit norm
    norm_scale = A.H(ksp).norm(2)
    y /= norm_scale
    x0 /= norm_scale

    # Wavelet operator
    W = WaveletLinop(x0.shape, wave_name=wave_name)

    # outer iteration initializations
    W1_base = 1 / (1 + torch.abs(W(x0)))
    W1 = torch.ones(W.oshape, dtype=torch.complex64, device=device)
    W2 = torch.ones(W.ishape, dtype=torch.complex64, device=device)
    x_prev = torch.zeros_like(x0)

    # sub-algorithm
    SFISTA = ReconSFISTA(
        A,
        y,
        x0,
        W1,
        W2,
        lamda1,
        lamda2,
        num_iters=num_fista_iters,
        max_eigen=max_eigen,
        W=W,
        tol=tol,
        verbose=verbose,
        **kwargs,
    )

    # iteratively solve
    pbar = tqdm(total=num_iters, desc="LACS", disable=False, leave=True)
    for i in range(num_iters):

        # Recon
        x = SFISTA.forward(disable=not verbose, leave=False)

        # update weights
        sparse_diff = torch.abs(W(x - x0))
        th = sparse_diff / (1 + sparse_diff)
        W1 = W1_base.clone()
        W1[th > eps] = 1
        W2 = 1 / (1 + torch.abs(x - x0))

        criterion = torch.norm(x - x_prev) / torch.norm(x_prev)
        pbar.set_postfix({"criterion": criterion.item()})
        pbar.update(1)
        if criterion < tol:
            if verbose:
                print(f"LACS Converged after {i} iterations")
            break

        SFISTA.update_weights(W1, W2)
        x_prev = x.clone()

        clear_cache()

    pbar.close()

    return x * norm_scale
