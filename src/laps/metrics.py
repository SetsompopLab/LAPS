"""
Inference Metrics
"""

from typing import Dict, Sequence, Union

import ants
import numpy as np
import torch
from einops import rearrange
from loguru import logger
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    VisualInformationFidelity,
)

from laps.utils import convert_to_numpy


def compute_patch_based_metrics(
    methods: Union[str, Sequence[str]],
    samps: torch.Tensor,
    targs: torch.Tensor,
    priors: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    Wrapper function to compute a set of inference metrics for a set of methods between
    samples, targets, and priors (if given).

    Metrics include:
    - Identity Distance (ID): 'id'
    - Peak Signal to Noise Ratio (PSNR): 'psnr'
    - Structural Similarity Index (SSIM): 'ssim'

    Data is returned in a dictionary with keys corresponding to the method names
    and values corresponding to the metric values.

    If save_path is provided, will save the metrics to this path as well.

    For all array inputs, expect input in one of the following forms:
    - (B,*im_shape) array (numpy or tensor)
    - list of (*im_shape) arrays (numpy or tensor)


    Returns
    -------
    metrics : dict
        Dictionary of metrics. First level keys are "samp_targ", and "samp_prior", and
        "targ_prior" if priors are provided. Second level keys are the method names requested
        to be computed. Values returned are either arrays of metric values for each image in
        the batch, or average metric values across the batch if return_averages is True.
    """
    VALID_METHODS = ["id", "psnr", "nrmse", "ssim", "mssim", "vif", "anc"]

    # validate inputs
    if isinstance(methods, str):
        methods = [methods]
    for method in methods:
        assert (
            method.lower() in VALID_METHODS
        ), f"Unknown method {method}. Must be one of {VALID_METHODS}"

    # Get patches from the images, and masks based on prior-target similarity level
    p_prior, p_targ, p_samp, val_mask, sims = get_similarity_patches(
        priors, targs, samps
    )
    val_mask = convert_to_numpy(val_mask).astype(np.float32)
    p_prior = convert_to_numpy(p_prior).astype(np.float32)
    p_targ = convert_to_numpy(p_targ).astype(np.float32)
    p_samp = convert_to_numpy(p_samp).astype(np.float32)
    sims = convert_to_numpy(sims).astype(np.float32)

    # Only get valid patches
    val_mask = rearrange(val_mask, "B N ... -> (B N) ...")
    p_samp = rearrange(p_samp, "B N ... -> (B N) ...")
    p_targ = rearrange(p_targ, "B N ... -> (B N) ...")
    p_prior = rearrange(p_prior, "B N ... -> (B N) ...")
    sims = rearrange(sims, "B N ... -> (B N) ...")

    p_prior = p_prior[val_mask == 1]
    p_targ = p_targ[val_mask == 1]
    p_samp = p_samp[val_mask == 1]
    sims = sims[val_mask == 1]

    # compute metrics
    out = {"Similarity": sims}
    for method in methods:
        if method.lower() == "id":
            out[f"samp_targ:{method.lower()}"] = IdentityDistance(p_samp, p_targ)
            out[f"samp_prior:{method.lower()}"] = IdentityDistance(p_samp, p_prior)
            out[f"targ_prior:{method.lower()}"] = IdentityDistance(p_targ, p_prior)
        elif method.lower() == "psnr":
            out[f"samp_targ:{method.lower()}"] = PSNR(p_samp, p_targ)
            out[f"samp_prior:{method.lower()}"] = PSNR(p_samp, p_prior)
            out[f"targ_prior:{method.lower()}"] = PSNR(p_targ, p_prior)
        elif method.lower() == "ssim":
            out[f"samp_targ:{method.lower()}"] = SSIM(p_samp, p_targ)
            out[f"samp_prior:{method.lower()}"] = SSIM(p_samp, p_prior)
            out[f"targ_prior:{method.lower()}"] = SSIM(p_targ, p_prior)
        elif method.lower() == "mssim":
            out[f"samp_targ:{method.lower()}"] = MSSIM(p_samp, p_targ)
            out[f"samp_prior:{method.lower()}"] = MSSIM(p_samp, p_prior)
            out[f"targ_prior:{method.lower()}"] = MSSIM(p_targ, p_prior)
        elif method.lower() == "nrmse":
            out[f"samp_targ:{method.lower()}"] = NRMSE(p_samp, p_targ)
            out[f"samp_prior:{method.lower()}"] = NRMSE(p_samp, p_prior)
            out[f"targ_prior:{method.lower()}"] = NRMSE(p_targ, p_prior)
        elif method.lower() == "vif":
            out[f"samp_targ:{method.lower()}"] = VIF(p_samp, p_targ)
            out[f"samp_prior:{method.lower()}"] = VIF(p_samp, p_prior)
            out[f"targ_prior:{method.lower()}"] = VIF(p_targ, p_prior)
        elif method.lower() == "anc":
            out[f"samp_targ:{method.lower()}"] = ANC(p_samp, p_targ)
            out[f"samp_prior:{method.lower()}"] = ANC(p_samp, p_prior)
            out[f"targ_prior:{method.lower()}"] = ANC(p_targ, p_prior)

    return out


def compute_inference_metrics(
    methods: Union[str, list[str]],
    samps: torch.Tensor,
    targs: torch.Tensor,
    priors: torch.Tensor,
    return_averages: bool = True,
):
    """
    Wrapper function to compute a set of inference metrics for a set of methods between
    samples, targets, and priors (if given).

    Metrics include:
    - Identity Distance (ID): 'id'
    - Peak Signal to Noise Ratio (PSNR): 'psnr'
    - Structural Similarity Index (SSIM): 'ssim'

    Doesn't compute on a patch-basis, but rather on the whole image.

    """

    VALID_METHODS = ["id", "psnr", "nrmse", "ssim", "mssim", "vif", "anc"]

    # validate inputs
    if isinstance(methods, str):
        methods = [methods]
    for method in methods:
        assert (
            method.lower() in VALID_METHODS
        ), f"Unknown method {method}. Must be one of {VALID_METHODS}"

    # compute metrics
    out = {}
    for method in methods:
        if method.lower() == "id":
            out[f"samp_targ:{method.lower()}"] = IdentityDistance(samps, targs)
            out[f"samp_prior:{method.lower()}"] = IdentityDistance(samps, priors)
            out[f"targ_prior:{method.lower()}"] = IdentityDistance(targs, priors)
        elif method.lower() == "psnr":
            out[f"samp_targ:{method.lower()}"] = PSNR(samps, targs)
            out[f"samp_prior:{method.lower()}"] = PSNR(samps, priors)
            out[f"targ_prior:{method.lower()}"] = PSNR(targs, priors)
        elif method.lower() == "ssim":
            out[f"samp_targ:{method.lower()}"] = SSIM(samps, targs)
            out[f"samp_prior:{method.lower()}"] = SSIM(samps, priors)
            out[f"targ_prior:{method.lower()}"] = SSIM(targs, priors)
        elif method.lower() == "mssim":
            out[f"samp_targ:{method.lower()}"] = MSSIM(samps, targs)
            out[f"samp_prior:{method.lower()}"] = MSSIM(samps, priors)
            out[f"targ_prior:{method.lower()}"] = MSSIM(targs, priors)
        elif method.lower() == "nrmse":
            out[f"samp_targ:{method.lower()}"] = NRMSE(samps, targs)
            out[f"samp_prior:{method.lower()}"] = NRMSE(samps, priors)
            out[f"targ_prior:{method.lower()}"] = NRMSE(targs, priors)
        elif method.lower() == "vif":
            out[f"samp_targ:{method.lower()}"] = VIF(samps, targs)
            out[f"samp_prior:{method.lower()}"] = VIF(samps, priors)
            out[f"targ_prior:{method.lower()}"] = VIF(targs, priors)
        elif method.lower() == "anc":
            out[f"samp_targ:{method.lower()}"] = ANC(samps, targs)
            out[f"samp_prior:{method.lower()}"] = ANC(samps, priors)
            out[f"targ_prior:{method.lower()}"] = ANC(targs, priors)

    # compute averages if requested
    if return_averages:
        for key in out.keys():
            out[key] = np.mean(out[key]).item()

    return out


def get_similarity_patches(
    priors: torch.Tensor,
    targets: torch.Tensor,
    samps: torch.Tensor,
    patch_size: int = 32,
    stride: int = 16,
    valid_threshold=0.1,
):
    assert len(priors.shape) == 3, "assumes prior is N, H, W"
    assert len(targets.shape) == 3, "assumes prior is N, H, W"
    assert len(samps.shape) == 3, "assumes prior is N, H, W"

    # get prior and target patches
    prior_patches = torch.nn.functional.unfold(
        priors[:, None, ...], kernel_size=patch_size, stride=stride
    )
    target_patches = torch.nn.functional.unfold(
        targets[:, None, ...],
        kernel_size=patch_size,
        stride=stride,
    )
    samps_patches = torch.nn.functional.unfold(
        samps[:, None, ...],
        kernel_size=patch_size,
        stride=stride,
    )

    # valid patches chosen by mean value across image
    cutoff = valid_threshold * targets.view(targets.shape[0], -1).quantile(
        0.999, dim=-1
    )
    valid_target_patches = torch.mean(target_patches, dim=1) > cutoff

    valid_patches = valid_target_patches

    # get similarity between patches
    similarity = torch.nn.functional.cosine_similarity(
        prior_patches, target_patches, dim=1
    )

    prior_patches = rearrange(
        prior_patches, "B (h w) n_p -> B n_p h w", h=patch_size, w=patch_size
    )
    target_patches = rearrange(
        target_patches, "B (h w) n_p -> B n_p h w", h=patch_size, w=patch_size
    )
    samps_patches = rearrange(
        samps_patches, "B (h w) n_p -> B n_p h w", h=patch_size, w=patch_size
    )

    return (
        prior_patches,
        target_patches,
        samps_patches,
        valid_patches,
        similarity,
    )


def preprocess_images_for_metrics(img_input, flatten=True):

    # ensure data converted to numpy array
    imgs = convert_to_numpy(img_input)

    # ensure shape and datatype consistent
    if flatten:
        imgs = imgs.reshape(imgs.shape[0], -1)

    imgs = imgs.astype(np.float32)

    # ensure scaling consistent (scale [0,255] -> [0,1])
    if np.max(imgs) > 10:
        imgs = imgs / 255.0

    return imgs


def mask_image_for_metrics(
    samps: np.ndarray,
    targs: np.ndarray,
    priors: np.ndarray,
    method: str = "none",
):
    """
    Mask the samples, targets, and priors with the prior masks.
    Assumes shape of all inputs is (B, *im_shape)

    Parameters
    ----------
    samps : np.ndarray
    targs : np.ndarray
    priors : np.ndarray
    method : str, optional
        Method to use for masking. Currently supports:
            "ants" - mask using ANTsPy
            "thresh" - mask using a threshold of 0 in the priors
            "none" - no masking
        Defaults to "none".

    Returns
    -------
    samps_masked : np.ndarray
    targs_masked : np.ndarray
    prior_masked : np.ndarray
    """

    # default no masking
    if method is None or method.lower() == "none":
        return samps, targs, priors

    # A list to store the prior masks
    prior_masks = []

    # If priors are not three dimensional, then raise an error
    if len(priors.shape) != 3:
        raise ValueError("Masking only supported for 2D images (batched)")

    # Iterate through the priors and get the masks from ants
    for i in range(priors.shape[0]):
        # Get the prior mask from ants
        if method == "ants":
            prior_antz = ants.from_numpy(priors[i, ...])
            prior_mask = ants.get_mask(prior_antz).numpy()
        else:
            prior_mask = priors[i, ...] > 0
        prior_masks.append(prior_mask)

    # Concatenate the prior masked together
    prior_masks = np.stack(prior_masks)

    # Make sure that the prior masks are the same shape as samps and targs before multiplying them
    if (
        prior_masks.shape != samps.shape
        or prior_masks.shape != targs.shape
        or prior_masks.shape != priors.shape
    ):
        raise ValueError(
            "Prior masks must be the same shape as samps, targs, and priors"
        )

    # Concatenate the mask and multiply it with samps and targs
    prior_masked = priors * prior_masks
    samps_masked = samps * prior_masks
    targs_masked = targs * prior_masks

    # Return the samp, targets, and priors
    return samps_masked, targs_masked, prior_masked


def IdentityDistance(
    x_obs: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
    x_tr: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
) -> np.ndarray:
    """
    Compute the identity distance (ID) metric between samples, targets, and priors (if given).

    For all array inputs, expect input in one of the following forms:
    - (B,*im_shape) array (numpy or tensor)
    - list of (*im_shape) arrays (numpy or tensor)

    Parameters
    ----------
    x_obs : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        Observed data (samples)
    x_tr : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        True data (targets)

    Returns
    -------
    dists : np.ndarray
        ID metric for each item in batch
    """

    # compute metrics
    samps = preprocess_images_for_metrics(x_obs)
    targs = preprocess_images_for_metrics(x_tr)

    id_dists = np.sum((samps - targs) ** 2, axis=-1)

    return id_dists


def PSNR(
    x_obs: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
    x_tr: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
) -> np.ndarray:
    """
    Compute PSNR between samples and targets.

    For all array inputs, expect input in one of the following forms:
    - (B,*im_shape) array (numpy or tensor)
    - list of (*im_shape) arrays (numpy or tensor)

    Parameters
    ----------
    x_obs : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        Observed data (samples)
    x_tr : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        True data (targets)

    Returns
    -------
    psnr : np.ndarray
        PSNR for each item in batch
    """

    # compute metrics
    samps = torch.asarray(preprocess_images_for_metrics(x_obs, flatten=False))
    targs = torch.asarray(preprocess_images_for_metrics(x_tr, flatten=False))

    # Create an instance of the PSNR metric
    psnr_metric = PeakSignalNoiseRatio(reduction="none", data_range=1.0, dim=(-2, -1))

    # Compute PSNR
    psnr = psnr_metric(samps, targs).cpu().numpy()

    return psnr


def NRMSE(
    x_tr: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
    x_obs: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
) -> np.ndarray:
    """
    Compute NRMSE between samples and targets.

    For all array inputs, expect input in one of the following forms:
    - (B,*im_shape) array (numpy or tensor)
    - list of (*im_shape) arrays (numpy or tensor)

    Parameters
    ----------
    x_obs : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        Observed data (samples)
    x_tr : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        True data (targets)

    Returns
    -------
    nrmse : np.ndarray
        NRMSE for each item in batch
    """

    x_tr = preprocess_images_for_metrics(x_tr)
    x_obs = preprocess_images_for_metrics(x_obs)

    rmse = np.linalg.norm(x_tr - x_obs, axis=(-2, -1), ord=2)
    nrmse = rmse / np.linalg.norm(x_tr, axis=(-2, -1), ord=2)

    return nrmse


def SSIM(
    x_obs: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
    x_tr: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
) -> np.ndarray:
    """
    Compute SSIM between samples and targets.

    For all array inputs, expect input in one of the following forms:
    - (B,*im_shape) array (numpy or tensor)
    - list of (*im_shape) arrays (numpy or tensor)

    Parameters
    ----------
    x_obs : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        Observed data (samples)
    x_tr : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        True data (targets)

    Returns
    -------
    ssim : np.ndarray
        SSIM for each item in batch
    """

    # enforce data is numpy array scaled (0,1)
    samps = torch.asarray(preprocess_images_for_metrics(x_obs, flatten=False))[
        :, None, ...
    ]
    targs = torch.asarray(preprocess_images_for_metrics(x_tr, flatten=False))[
        :, None, ...
    ]

    # Create an instance of the SSIM metric
    ssim_metric = StructuralSimilarityIndexMeasure(reduction="none", data_range=1.0)

    # Compute SSIM
    ssim = ssim_metric(samps, targs).cpu().numpy()

    return ssim


def ANC(
    x_obs: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
    x_tr: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
) -> np.ndarray:
    """
    Compute average normalized cross correlation between samples and targets.

    For all array inputs, expect input in one of the following forms:
    - (B,*im_shape) array (numpy or tensor)
    - list of (*im_shape) arrays (numpy or tensor)

    Parameters
    ----------
    x_obs : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        Observed data (samples)
    x_tr : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        True data (targets)

    Returns
    -------
    anc : np.ndarray
        ANC for each item in batch
    """

    # enforce data is numpy array scaled (0,1)
    samps = torch.asarray(preprocess_images_for_metrics(x_obs, flatten=True))
    targs = torch.asarray(preprocess_images_for_metrics(x_tr, flatten=True))

    similarity = torch.nn.functional.cosine_similarity(samps, targs, dim=-1)

    return similarity.cpu().numpy()


def MSSIM(
    x_obs: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
    x_tr: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
) -> np.ndarray:
    """
    Compute multi-scale SSIM between samples and targets.

    For all array inputs, expect input in one of the following forms:
    - (B,*im_shape) array (numpy or tensor)
    - list of (*im_shape) arrays (numpy or tensor)

    Parameters
    ----------
    x_obs : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        Observed data (samples)
    x_tr : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        True data (targets)

    Returns
    -------
    ssim : np.ndarray
        SSIM for each item in batch
    """

    # enforce data is numpy array scaled (0,1)
    samps = torch.asarray(preprocess_images_for_metrics(x_obs, flatten=False))[
        :, None, ...
    ]
    targs = torch.asarray(preprocess_images_for_metrics(x_tr, flatten=False))[
        :, None, ...
    ]

    # Create an instance of the SSIM metric
    mssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
        reduction="none", data_range=1.0
    )

    # Compute SSIM
    mssim = mssim_metric(samps, targs).cpu().numpy()

    return mssim


def VIF(
    x_obs: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
    x_tr: Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]],
) -> np.ndarray:
    """
    Compute Visual Information Fidelity (VIF) between samples and targets.

    For all array inputs, expect input in one of the following forms:
    - (B,*im_shape) array (numpy or tensor)
    - list of (*im_shape) arrays (numpy or tensor)

    Parameters
    ----------
    x_obs : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        Observed data (samples)
    x_tr : Union[np.ndarray, torch.Tensor, list[torch.Tensor], list[np.ndarray]]
        True data (targets)

    Returns
    -------
    vif : np.ndarray
        vif for each item in batch
    """

    # enforce data is numpy array scaled (0,1)
    samps = torch.asarray(preprocess_images_for_metrics(x_obs, flatten=False))[
        :, None, ...
    ]
    targs = torch.asarray(preprocess_images_for_metrics(x_tr, flatten=False))[
        :, None, ...
    ]

    # Create an instance of the VIF metric
    vif_metric = VisualInformationFidelity()

    # Compute VIF
    vif = []
    for i in range(samps.shape[0]):
        vif.append(vif_metric(samps[i : i + 1], targs[i : i + 1]).cpu().numpy())
    vif = np.concatenate(vif, axis=0)

    return vif
