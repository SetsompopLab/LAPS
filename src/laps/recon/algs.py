from typing import Callable, Optional, Tuple, Union

import sigpy as sp
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from laps.utils import np_to_torch, torch_to_np


def lin_solve(
    AHA: torch.Tensor,
    AHb: torch.Tensor,
    lamda: Optional[float] = 0.0,
    solver: str = "lstsq",
) -> torch.Tensor:
    """
    Solves (AHA + lamda I) @ x = AHb for x

    Parameters:
    -----------
    AHA : torch.Tensor
        square matrix with shape (..., n, n)
    AHb : torch.Tensor
        matrix with shape (..., n, m)
    lamda : float
        optional L2 regularization
    solver : str
        'pinv' - pseudo inverse
        'solve' - torch.linalg.solve
        'lstsq' - least squares
        'inv' - regular inverse
        'cg' - conjugate gradient
        'gd' - gradient descent

    Returns:
    --------
    x : torch.Tensor
        solution with shape (..., n, m)
    """
    tup = (AHA.ndim - 2) * (None,) + (slice(None),) * 2
    solver = solver.lower()
    if lamda > 0:
        AHA += lamda * torch.eye(AHA.shape[-1], dtype=AHA.dtype, device=AHA.device)[tup]
    if solver == "lstsq_torch":
        x = torch.linalg.lstsq(AHA, AHb).solution
    elif solver == "solve":
        x = torch.linalg.solve(AHA, AHb)
    elif solver == "lstsq":
        n, m = AHb.shape[-2:]
        AHA_cp = torch_to_np(AHA).reshape(-1, n, n)
        AHb_cp = torch_to_np(AHb).reshape(-1, n, m)
        dev = sp.get_device(AHA_cp)
        with dev:
            x = dev.xp.zeros_like(AHb_cp)
            for i in range(AHA_cp.shape[0]):
                x[i] = dev.xp.linalg.lstsq(AHA_cp[i], AHb_cp[i], rcond=None)[0]
        x = np_to_torch(x).reshape(AHb.shape)
    elif solver == "pinv":
        x = torch.linalg.pinv(AHA, hermitian=True) @ AHb
    elif solver == "inv":
        x = torch.linalg.inv(AHA) @ AHb
    elif solver == "cg":
        x = conjugate_gradient(lambda x: AHA @ x, AHb, num_iters=100, lamda_l2=lamda)
    else:
        raise NotImplementedError
    return x


def fista(
    AHA: nn.Module,
    AHb: torch.Tensor,
    proxg: Callable,
    num_iters: Optional[int] = 20,
    ptol_exit: Optional[float] = 0.5,
    verbose: Optional[bool] = True,
) -> torch.Tensor:
    """
    Solves ||Ax - b||_2^2 + lamda ||Gx||_1, where G is a linear function.
    The proximal operator of Gx if given by 'proxg'

    Parameters
    ----------
    AHA : nn.Module
        The gram or normal operator of A
    AHb : torch.tensor
        The A hermitian transpose times b
    proxg : nn.Module
        The torch
    num_iters : int
        Number of iterations
    ptol_exit : float
        percent tolerance exit condition
    verbose : bool
        toggles print statements

    Returns
    ---------
    x : torch.tensor
        Reconstructed tensor
    """

    # Start at AHb
    x = AHb.clone()
    z = x.clone()

    if num_iters <= 0:
        return x

    for k in tqdm(range(0, num_iters), "FISTA Iterations", disable=not verbose):
        x_old = x.clone()
        x = z.clone()

        gr = AHA(x) - AHb
        x = proxg(x - gr)
        if k == 0:
            z = x
        else:
            step = k / (k + 3)
            z = x + step * (x - x_old)
        ptol = 100 * torch.norm(x_old - x) / torch.norm(x)
        if ptol < ptol_exit:
            if verbose:
                print(f"Tolerance reached after {k + 1} iterations, exiting FISTA")
            break

    return x


def conjugate_gradient(
    AHA: Union[nn.Module, Callable],
    AHb: torch.Tensor,
    P: Optional[nn.Module] = None,
    num_iters: Optional[int] = 10,
    lamda_l2: Optional[float] = 0.0,
    tolerance: Optional[float] = 1e-8,
    return_resids: Optional[bool] = False,
    verbose=True,
) -> torch.Tensor:
    """Conjugate gradient for complex numbers. The output is also complex.
    Solve for argmin ||Ax - b||^2. Inspired by sigpy!

    Parameters:
    -----------
    AHA : nn.Module
        Linear operator representing the gram/normal operator of A
    AHb : torch.tensor
        The A hermitian transpose times b
    P : nn.Module
        Preconditioner
    num_iters : int
        Max number of iterations.
    lamda_l2 : float
        Replaces AHA with AHA + lamda_l2 * I
    tolerance : float
        Used as stopping criteria
    return_resids : bool
        toggles return of residuals
    verbose : bool
        toggles print statements

    Returns:
    ---------
    x : torch.tensor <complex>
        least squares estimate of x, same shape as x0 if provided
    """
    # Default preconditioner is identity matrix
    if P is None:
        P = lambda x: x

    # Tikonov regularization
    AHA_wrapper = lambda x: AHA(x) + lamda_l2 * x

    # Start at AHb
    x0 = AHb.clone()
    if num_iters == 0:
        return x0

    # Define iterative vars
    r = AHb - AHA_wrapper(x0)
    z = P(r)
    p = z.clone()

    resids = []

    # Main loop
    for i in tqdm(range(num_iters), "CG Iterations", disable=not verbose):

        # Apply model
        Ap = AHA_wrapper(p)
        pAp = torch.real(torch.sum(p.conj() * Ap)).item()

        # Update x
        # assert pAp > 0, 'A is not Semi-Definite'
        rz = torch.real(torch.sum(r.conj() * z))
        alpha = rz / pAp
        x0 = x0 + alpha * p

        # Update r
        r = r - alpha * Ap
        rnrm = torch.norm(r)
        resids.append(rnrm.item())
        if rnrm < tolerance:
            break

        # Update z
        z = P(r)

        # Update p
        beta = torch.real(torch.sum(r.conj() * z)) / rz
        p = z + beta * p

    if return_resids:
        return resids, x0
    else:
        return x0


def power_method_matrix(
    M: torch.Tensor,
    vec_init: Optional[torch.Tensor] = None,
    num_iter: int = 100,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses power method to find largest eigenvalue and corresponding eigenvector

    Parameters:
    -----------
    M : torch.Tensor
        input matrix with shape (..., n, n)
    vec_init : torch.Tensor
        initial guess of eigenvector with shape (..., n)
    num_iter : int
        number of iterations to run power method
    verbose : bool
        toggles progress bar

    Returns:
    --------
    eigen_vec : torch.Tensor
        eigenvector with shape (..., n)
    eigen_val : torch.Tensor
        eigenvalue with shape (...)
    """
    # Consts
    n = M.shape[-1]
    assert M.shape[-2] == n
    assert M.ndim >= 2
    num_iter = max(num_iter, 1)

    if vec_init:
        eigen_vec = vec_init[..., None]
    else:
        eigen_vec = torch.ones((*M.shape[:-1], 1), device=M.device, dtype=M.dtype)
    eigen_val = torch.linalg.norm(eigen_vec, axis=-2, keepdims=True)

    for i in tqdm(
        range(num_iter), "Power Iterations", disable=not verbose, leave=False
    ):
        eigen_vec = M @ eigen_vec
        eigen_val = torch.linalg.norm(eigen_vec, axis=-2, keepdims=True)
        eigen_vec = eigen_vec / eigen_val
    eigen_vec = rearrange(eigen_vec, "... n 1 -> n ...")
    eigen_val = eigen_val.squeeze()

    return eigen_vec, eigen_val
