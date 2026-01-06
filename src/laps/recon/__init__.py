"""
Reconstruction algorithms and utilities for MRI reconstruction.

This module provides various reconstruction methods including:
- Classical methods (CG-SENSE, FISTA, LACS)
- Deep learning methods (Stable Diffusion, MoDL)
- Linear operators and proximal operators
"""

from .adadiff import AdaDiffParams, AdaDiffReconstructor
from .algs import conjugate_gradient, fista, lin_solve
from .classic_recons import CG_SENSE_recon, FISTA_recon, LACS_fista_recon
from .linops import CartesianSenseLinop, WaveletLinop, linop
from .nerp import NERPParams, NERPReconstructor
from .power import power_method_operator
from .prox import L1Wav, prox_vector, soft_thresh
from .reconstructor import (
    AHbReconstructor,
    CGParams,
    CGReconstructor,
    DiffusersReconstructor,
    LACSParams,
    LDMParams,
    LACSReconstructor,
    ModlParams,
    MoDLReconstructor,
    ReconParams,
    Reconstructor,
    ReconstructorOutput,
    StableDiffusionReconstructor,
)
from .utils import get_reconstructors

__all__ = [
    # Main classes
    "Reconstructor",
    "ReconstructorOutput",
    "ReconParams",
    "StableDiffusionReconstructor",
    "CGReconstructor",
    "LACSReconstructor",
    "MoDLReconstructor",
    "AHbReconstructor",
    "get_reconstructors",
    # NERP
    "NERPReconstructor",
    "NERPParams",
    # AdaDiff
    "AdaDiffReconstructor",
    "AdaDiffParams",
    # Classic reconstruction functions
    "CG_SENSE_recon",
    "FISTA_recon",
    "LACS_fista_recon",
    # Linear operators
    "linop",
    "CartesianSenseLinop",
    "WaveletLinop",
    # Algorithms
    "conjugate_gradient",
    "fista",
    "lin_solve",
    # Proximal operators
    "L1Wav",
    "soft_thresh",
    "prox_vector",
    # Utilities
    "power_method_operator",
]
