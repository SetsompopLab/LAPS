from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from laps.recon.linops import WaveletLinop, linop
from laps.recon.power import power_method_operator
from laps.recon.prox import prox_vector, soft_thresh


class ReconSFISTA(nn.Module):
    r"""
    L1 Wavelet regularized reconstruction.

    Considers the problem

    .. math::
        \min_x \frac{1}{2} \|A x - y \|_2^2 + \lambda_1 \| W1 Ψ x \|_1 + \lambda_2 \| W2 (x - x0) \|_1

    where A is the forward model, Ψ is the wavelet operator, x0 is a prior scan,
    and W1 and W2 are diagonal weight matrices.

    Based on paper: https://arxiv.org/pdf/1407.2602
    """  # noqa : W605

    def __init__(
        self,
        A: linop,
        y: torch.Tensor,
        x0: torch.Tensor,
        W1: torch.Tensor,
        W2: torch.Tensor,
        lamda1: float = 1e-3,
        lamda2: float = 1e-3,
        num_iters: int = 40,
        max_eigen: Optional[float] = None,
        W: Optional[WaveletLinop] = None,
        tol: float = 1e-6,
        verbose: bool = True,
    ):
        super().__init__()

        self.im_size = x0.shape
        self.device = y.device
        self.y = y
        self.x0 = x0
        self.W1 = W1
        self.W2 = W2
        self.lm1 = lamda1
        self.lm2 = lamda2
        self.mu = 0.001 / ((lamda1 + lamda2) / 2)
        self.tol = tol
        self.num_iters = num_iters
        self.verbose = verbose

        # Forward model with eigen scaling
        # Get max eigenvalue of forward operator
        if max_eigen is None:
            x0 = torch.randn(A.ishape, dtype=torch.complex64).to(self.device)
            _, max_eigen = power_method_operator(A.normal, x0, verbose=verbose)
            max_eigen = max_eigen * 1.01
        self.A_max_eigen = max_eigen
        self.AHA = lambda x: A.normal(x) / max_eigen
        self.A = lambda x: A(x) / (max_eigen**0.5)
        self.AH = lambda y: A.H(y) / (max_eigen**0.5)

        # Precompute
        self.AHb = self.AH(y)

        if W is not None:
            self.W = W
        else:
            self.W = WaveletLinop(self.im_size)

        assert self.W1.shape == tuple(self.W.oshape), "W1 should be wavelet shape"
        assert self.W2.shape == tuple(self.W.ishape), "W2 should be image shape"

        # Compute Lipshitz constant
        self.L = self._eval_L()

    def update_weights(self, W1: torch.Tensor, W2: torch.Tensor):
        """
        Update weights, for iterative calling
        """
        self.W1 = W1
        self.W2 = W2
        # update lipshitz constant
        self.L = self._eval_L()

    def _grad_g1(self, x: torch.Tensor):
        w1px = self.W1 * self.W(x)
        return (1 / self.mu) * self.W.H(
            self.W1.conj() * (w1px - soft_thresh(self.lm1 * self.mu, w1px))
        )

    def _grad_g2(self, x: torch.Tensor):
        w2x = self.W2 * (x - self.x0)
        return (
            (1 / self.mu)
            * self.W2.conj()
            * (w2x - soft_thresh(self.lm2 * self.mu, w2x))
        )

    def _target(self, x: torch.Tensor):
        out = torch.linalg.norm(self.A(x) - self.y) ** 2
        out += prox_vector(self.W1 * self.W(x), self.lm1 * self.mu) / self.mu
        out += prox_vector(self.W2 * (x - self.x0), self.lm2 * self.mu) / self.mu
        return out

    def _eval_L(self):
        """
        Compute an upper bound L >= ||A||_2^2 + 1/\mu * (||W1 W||_2^2 + ||W2||_2^2)
        """  # noqa : W605
        # 1. ||A||_2^2 -> already scaled to max_eigen of 1
        # 2. Compute ||W1 W||_2^2
        # 3. Compute ||W2||_2^2

        x_rand = torch.randn(self.W.ishape, dtype=torch.complex64).to(self.device)
        WHW = lambda x: self.W.H((self.W1**2) * self.W(x))
        W1_eigm = (
            power_method_operator(WHW, x_rand, num_iter=15, verbose=False)[1] * 1.01
        )

        W2_eigm = (torch.max(torch.abs(self.W2)) ** 2).item()

        L = 1 + (1 / self.mu) * (W1_eigm + W2_eigm)

        if self.verbose:
            print(
                f"Eval Lipshitz L={L} (norms: AHA={self.A_max_eigen}, W1={W1_eigm}, W2={W2_eigm})"
            )

        return L

    def _update_stepsize(self, t: float, tp: float):
        """
        Update stepsize
        """
        t_temp = t
        t = (1 + ((1 + 4 * (t**2)) ** (0.5))) / 2
        tp = t_temp
        return t, tp

    def _argmin_target(self, *args):
        mn = float("inf")
        xmin = None
        for x in args:
            t = self._target(x)
            if t < mn:
                mn = t
                xmin = x
        return xmin

    def forward(self, **tqdm_args) -> torch.Tensor:
        """
        Runs SFISTA reconstruction
        """

        # Initializiations
        z = self.AHb.clone()
        z_prev = self.AHb.clone()
        x = self.AHb.clone()
        x_prev = self.AHb.clone()
        t = 1
        tp = 1

        if self.num_iters <= 0:
            return x

        pbar = tqdm(total=self.num_iters, desc="SFISTA", **tqdm_args)

        for k in range(0, self.num_iters):
            x_prev = x.clone()

            df = self.AHA(z) - self.AHb
            dg1 = self._grad_g1(x_prev)
            dg2 = self._grad_g2(x_prev)

            u = z - (1 / self.L) * (df + dg1 + dg2)

            t, tp = self._update_stepsize(t, tp)

            x = self._argmin_target(u, x_prev)

            z = x + (tp / t) * (u - x) + ((tp - 1) / t) * (x - x_prev)

            criterion = torch.norm(z - z_prev) / torch.norm(z_prev)

            if self.verbose:
                pbar.set_postfix({"gnorm": criterion.item()})
                pbar.update(1)

            if criterion < self.tol:
                if self.verbose:
                    print(f"Tol reached after {k + 1} iterations, exiting SFISTA")
                break

            z_prev = z.clone()

        pbar.close()

        return z
