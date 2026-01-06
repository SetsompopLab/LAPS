from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
from sigpy.linop import Wavelet as Wavelet_Sigpy

from laps.utils import fftc, ifftc, np_to_torch, torch_resize, torch_to_np


class linop(nn.Module):
    def __init__(self, ishape, oshape):
        super().__init__()
        self.ishape = ishape
        self.oshape = oshape

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    # Alias
    def H(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.adjoint(y, **kwargs)

    def normal(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    # Alias
    def N(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.normal(x, **kwargs)


class CartesianSenseLinop(linop):
    def __init__(
        self,
        mps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.complex64,
        coil_batch_size: Optional[int] = None,
        batch_size: int = 0,
        ishape: Optional[Sequence[int]] = None,
    ):
        """
        Initializes an instance of the MRIForward class which simulates MRI forward operations.

        Parameters:
            mps (torch.Tensor): The sensitivity maps of shape (n_coils, *im_size) or (batch_size, n_coils, *im_size) if batch_size > 0.
            mask (Optional[torch.Tensor], optional): A binary mask tensor of shape (*im_size) or (batch_size, *im_size) if batch_size > 0.
                If None, a mask of all ones is used. Default is None.
            dtype (torch.dtype, optional): The data type for computations. Default is torch.complex64.
            coil_batch_size (Optional[int], optional): The number of coils to process in a batch. If None, uses all coils. Default is None.
            batch_size (int, optional): The batch size for processing multiple images simultaneously. If 0, no batch dimension is used. Default is 0.
            ishape (Optional[Sequence[int]], optional): The input image shape. If input size differs from mps shape due to cropping/padding, specify here.
                Must match the number of image dimensions. If None, uses mps shape. Default is None.

        Notes:
            - If input size differs from mps shape due to cropping or padding, specify the input shape using `ishape`.
            - The output shape (`oshape`) is always determined by the mps shape.
            - The mask shape must match the image shape.
        """

        add_batch = 1 if batch_size > 0 else 0
        self.batch_size = batch_size
        self.im_size = mps.shape[1 + add_batch :]
        self.n_coils = mps.shape[add_batch]
        self.nd = len(self.im_size)
        self.dtype = dtype
        self.fftdims = tuple(range(-self.nd, 0))

        if coil_batch_size is None:
            coil_batch_size = self.n_coils
        self.coil_batch_size = coil_batch_size
        self.batch_over_coils = self.coil_batch_size < self.n_coils

        if ishape is not None:
            assert len(ishape) == len(
                self.im_size
            ), "Output shape must match number of input shape dimensions"
            self.ishape = ishape if batch_size == 0 else (batch_size, *self.im_size)
            self.im_size_in = ishape
        else:
            self.ishape = (
                self.im_size if batch_size == 0 else (batch_size, *self.im_size)
            )
            self.im_size_in = self.im_size

        self.do_input_pad = (self.im_size_in != self.im_size) and (ishape is not None)

        self.oshape = (
            (self.n_coils, *self.im_size)
            if batch_size == 0
            else (batch_size, self.n_coils, *self.im_size)
        )

        super().__init__(self.ishape, self.oshape)

        if mask is not None:
            assert (
                mask.shape[add_batch:] == self.im_size
            ), "Mask shape must match image shape"
            if self.batch_size > 0:
                self.mask = nn.Parameter(
                    mask.unsqueeze(1).to(self.dtype), requires_grad=False
                )
            else:
                self.mask = nn.Parameter(
                    mask[
                        None,
                    ].to(self.dtype),
                    requires_grad=False,
                )
        else:
            mask = torch.ones(self.im_size).to(self.dtype)
            self.mask = nn.Parameter(
                mask[
                    None,
                ],
                requires_grad=False,
            )

        if self.batch_size > 0:
            self.mps = nn.Parameter(
                mps.to(self.dtype),
                requires_grad=False,
            )
        else:
            self.mps = nn.Parameter(
                mps[
                    None,
                ].to(self.dtype),
                requires_grad=False,
            )

    def forward(self, x, **kwargs):
        """
        Runs the forward model.
        Args:
            x (torch.Tensor): input image of shape self.im_size_in or (B, *self.im_size_in).

        Returns:
            torch.Tensor: The output tensor after applying the forward operation.
        """

        use_mask = kwargs.get("use_mask", True)
        batched = (x.ndim == (len(self.ishape) + 1)) or self.batch_size > 0
        if not batched:
            x = x[
                None,
            ]
            mask = self.mask[
                None,
            ]
        else:
            mask = self.mask
        B = x.shape[0]

        if self.do_input_pad:
            x = torch_resize(x, (B, *self.im_size))

        # Batch to prevent OOM.
        if self.batch_over_coils:
            y = torch.zeros(
                B, self.n_coils, *self.im_size, dtype=self.dtype, device=x.device
            )
            for ci, cf in self.coil_batch_iterator():

                # Apply coils
                xi = x[:, None] * self.mps[:, ci:cf]
                # FFT
                xi = fftc(xi, self.fftdims)

                # Mask
                if use_mask:
                    y[:, ci:cf] = xi * mask
                else:
                    y[:, ci:cf] = xi

        else:
            x = x[:, None] * self.mps
            x = fftc(x, self.fftdims)
            if use_mask:
                y = x * mask
            else:
                y = x

        if not batched:
            y = y[0]

        return y

    def adjoint(self, y, **kwargs):
        """
        Runs the conjugate forward model.
        Args:
            ksp (torch.Tensor): input k-space of shape (self.n_coils, *self.im_size) or
                (B, self.n_coils, *self.im_size).

        Returns:
            torch.Tensor: The output tensor after applying the conjugate forward operation.
        """

        use_mask = kwargs.get("use_mask", True)
        batched = (y.ndim == (len(self.oshape) + 1)) or self.batch_size > 0
        if not batched:
            y = y[
                None,
            ]
            mask = self.mask[
                None,
            ]
        else:
            mask = self.mask
        B = y.shape[0]

        if use_mask:
            # Re-mask
            y = y * mask

        if self.batch_over_coils:
            x = torch.zeros((B, *self.im_size), dtype=self.dtype, device=y.device)
            for ci, cf in self.coil_batch_iterator():

                # IFFT
                xi = ifftc(y[:, ci:cf], self.fftdims)

                # Coil Combine
                x += torch.sum(xi * torch.conj(self.mps[:, ci:cf]), dim=1)

        else:
            x = ifftc(y, self.fftdims)
            x = torch.sum(x * torch.conj(self.mps), dim=1)

        if self.do_input_pad:
            x = torch_resize(x, (B, *self.im_size_in))

        if not batched:
            x = x[0]

        return x

    def normal(self, x, **kwargs):
        """
        Computes the normal operator A^H A more memory-efficiently.
        Args:
            x (torch.Tensor): input image of shape self.im_size_in.

        Returns:
            torch.Tensor: The output tensor after applying the normal operator.
        """
        use_mask = kwargs.get("use_mask", True)
        batched = (x.ndim == (len(self.ishape) + 1)) or self.batch_size > 0
        if not batched:
            x = x[
                None,
            ]
            mask = self.mask[
                None,
            ]
        else:
            mask = self.mask
        B = x.shape[0]

        if self.do_input_pad:
            x = torch_resize(x, (B, *self.im_size))

        if self.batch_over_coils:
            x_out = torch.zeros((B, *self.im_size), dtype=self.dtype, device=x.device)
            for ci, cf in self.coil_batch_iterator():
                xi = x[:, None] * self.mps[:, ci:cf]
                xi = fftc(xi, self.fftdims)
                if use_mask:
                    xi = xi * mask
                xi = ifftc(xi, self.fftdims)
                x_out += torch.sum(xi * torch.conj(self.mps[:, ci:cf]), dim=1)
        else:
            x = x[:, None] * self.mps
            x = fftc(x, self.fftdims)
            if use_mask:
                x = x * mask
            x = ifftc(x, self.fftdims)
            x_out = torch.sum(x * torch.conj(self.mps), dim=1)

        if self.do_input_pad:
            x_out = torch_resize(x_out, (B, *self.im_size_in))

        if not batched:
            x_out = x_out[0]

        return x_out

    def get_brain_mask(self) -> torch.Tensor:
        """
        Compute the mask for the brain region from sensitivity maps, with appropriate padding.
        """

        B = self.mps.shape[0]
        mask = torch.sum(self.mps * self.mps.conj(), dim=1).abs()

        if self.do_input_pad:
            mask = torch_resize(mask, (B, *self.im_size_in))

        mask = (mask > 0.1).float()
        return mask

    def coil_batch_iterator(self) -> Iterable:
        """
        Returns an iterator over the coil indices.
        """

        for ci in range(0, self.n_coils, self.coil_batch_size):
            yield ci, min(ci + self.coil_batch_size, self.n_coils)


class WaveletLinop(linop):
    """
    Wrapper for sigpy linop in torch
    """

    def __init__(
        self,
        shape: tuple,
        axes: Optional[tuple] = None,
        wave_name: str = "db4",
    ):
        """
        Parameters:
        -----------
        shape - tuple
            the image/volume dimensions
        axes - tuple
            axes to compute wavelet transform over
        wave_name - str
            the type of wavelet to use from:
            ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus',
            'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor', ...]
            see https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#wavelet-families
        """

        # Save wavelet params
        if axes is None:
            axes = tuple([i for i in range(len(shape))])
        self.W = Wavelet_Sigpy(shape, axes=axes, wave_name=wave_name)

        super().__init__(self.W.ishape, self.W.oshape)

    def forward(self, x, **kwargs) -> torch.Tensor:
        """
        Proximal operator for l1 wavelet

        Parameters
        ----------
        input - torch.tensor <complex> | CPU
            image/volume input
        """

        input_sigpy = torch_to_np(x)
        output_sigpy = self.W(input_sigpy)
        output = np_to_torch(output_sigpy)
        if isinstance(output, tuple):
            output = output[0]
        return output

    def adjoint(self, y, **kwargs) -> torch.Tensor:
        """
        Proximal operator for l1 wavelet

        Parameters
        ----------
        input - torch.tensor <complex> | CPU
            image/volume input
        """

        input_sigpy = torch_to_np(y)
        output_sigpy = self.W.H(input_sigpy)
        output = np_to_torch(output_sigpy)
        if isinstance(output, tuple):
            output = output[0]
        return output

    def normal(self, x, **kwargs) -> torch.Tensor:
        """
        Proximal operator for l1 wavelet

        Parameters
        ----------
        input - torch.tensor <complex> | CPU
            image/volume input
        """
        input_sigpy = torch_to_np(x)
        output_sigpy = self.W.H(self.W(input_sigpy))
        output = np_to_torch(output_sigpy)
        if isinstance(output, tuple):
            output = output[0]
        return output
