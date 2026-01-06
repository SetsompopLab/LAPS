from typing import Callable, Optional, Tuple

import torch
from tqdm import tqdm


def power_method_operator(
    A: Callable,
    x0: torch.Tensor,
    num_iter: int = 15,
    verbose: bool = True,
) -> Tuple[torch.Tensor, float]:
    """
    Uses power method to find largest eigenvalue and corresponding eigenvector

    Parameters:
    -----------
    A : Callable
        linear operator, reccomended to be the normal operator AHA
    vec_init : torch.Tensor
        initial guess of eigenvector with shape (*vec_shape)
    num_iter : int
        number of iterations to run power method
    verbose : bool
        toggles progress bar

    Returns:
    --------
    eigen_vec : torch.Tensor
        eigenvector with shape (*vec_shape)
    eigen_val : float
        eigenvalue
    """

    for _ in tqdm(range(num_iter), "Max Eigenvalue", disable=not verbose, leave=False):

        z = A(x0)
        ll = torch.norm(z)
        x0 = z / ll

    if verbose:
        print(f"Max Eigenvalue = {ll.item()}")

    return x0, ll.item()
