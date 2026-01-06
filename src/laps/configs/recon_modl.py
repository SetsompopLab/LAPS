import os
from pathlib import Path
from typing import Literal
from laps import PROJECT_ROOT
from laps.recon import ModlParams
from loguru import logger

modl_1d = ModlParams(
    n_layers=5,
    unroll_iters=10,
    n_filters=64,
    norm_type="instance-affine",
    scale_denoiser=True,
    path=os.path.join(
        PROJECT_ROOT,
        Path("models/modl/model_1d_joint_finetune.pth"),
    ),
)

modl_1d_r3 = ModlParams(
    n_layers=5,
    unroll_iters=10,
    n_filters=64,
    norm_type="instance-affine",
    scale_denoiser=True,
    path=os.path.join(
        PROJECT_ROOT,
        Path("models/modl/model_1d_3_finetune.pth"),
    ),
)

modl_1d_r5 = ModlParams(
    n_layers=5,
    unroll_iters=10,
    n_filters=64,
    norm_type="instance-affine",
    scale_denoiser=True,
    path=os.path.join(
        PROJECT_ROOT,
        Path("models/modl/model_1d_5_finetune.pth"),
    ),
)

modl_1d_r6 = ModlParams(
    n_layers=5,
    unroll_iters=10,
    n_filters=64,
    norm_type="instance-affine",
    scale_denoiser=True,
    path=os.path.join(
        PROJECT_ROOT,
        Path("models/modl/model_1d_6_finetune.pth"),
    ),
)

modl_1d_r7 = ModlParams(
    n_layers=5,
    unroll_iters=10,
    n_filters=64,
    norm_type="instance-affine",
    scale_denoiser=True,
    path=os.path.join(
        PROJECT_ROOT,
        Path("models/modl/model_1d_7_finetune.pth"),
    ),
)

modl_1d_r9 = ModlParams(
    n_layers=5,
    unroll_iters=10,
    n_filters=64,
    norm_type="instance-affine",
    scale_denoiser=True,
    path=os.path.join(
        PROJECT_ROOT,
        Path("models/modl/model_1d_9_finetune.pth"),
    ),
)

modl_2d = ModlParams(
    n_layers=5,
    unroll_iters=10,
    n_filters=64,
    norm_type="instance-affine",
    scale_denoiser=True,
    path=os.path.join(
        PROJECT_ROOT,
        Path("models/modl/model_2d_joint_finetune.pth"),
    ),
)

modl_2d_r5 = ModlParams(
    n_layers=5,
    unroll_iters=10,
    n_filters=64,
    norm_type="instance-affine",
    scale_denoiser=True,
    path=os.path.join(
        PROJECT_ROOT,
        Path("models/modl/model_2d_5_finetune.pth"),
    ),
)

modl_2d_r10 = ModlParams(
    n_layers=5,
    unroll_iters=10,
    n_filters=64,
    norm_type="instance-affine",
    scale_denoiser=True,
    path=os.path.join(
        PROJECT_ROOT,
        Path("models/modl/model_2d_10_finetune.pth"),
    ),
)

modl_2d_r15 = ModlParams(
    n_layers=5,
    unroll_iters=10,
    n_filters=64,
    norm_type="instance-affine",
    scale_denoiser=True,
    path=os.path.join(
        PROJECT_ROOT,
        Path("models/modl/model_2d_15_finetune.pth"),
    ),
)

modl_2d_r20 = ModlParams(
    n_layers=5,
    unroll_iters=10,
    n_filters=64,
    norm_type="instance-affine",
    scale_denoiser=True,
    path=os.path.join(
        PROJECT_ROOT,
        Path("models/modl/model_2d_20_finetune.pth"),
    ),
)

modl_2d_r30 = ModlParams(
    n_layers=5,
    unroll_iters=10,
    n_filters=64,
    norm_type="instance-affine",
    scale_denoiser=True,
    path=os.path.join(
        PROJECT_ROOT,
        Path("models/modl/model_2d_30_finetune.pth"),
    ),
)


def get_modl_config(accel_dim: Literal[1, 2], R: float) -> ModlParams:
    """
    Returns the appropriate ModlParams based on the acceleration dimension and rate.
    """
    R = int(R)

    if accel_dim == 1:
        if R <= 4:
            logger.info("Initializing MoDL 1D model for R3")
            return modl_1d_r3
        elif R == 5:
            logger.info("Initializing MoDL 1D model for R5")
            return modl_1d_r5
        elif R == 6:
            logger.info("Initializing MoDL 1D model for R6")
            return modl_1d_r6
        elif R == 7:
            logger.info("Initializing MoDL 1D model for R7")
            return modl_1d_r7
        else:
            logger.info("Initializing MoDL 1D model for R9")
            return modl_1d_r9
    else:
        if R <= 7: 
            logger.info("Initializing MoDL 2D model for R5")
            return modl_2d_r5
        elif 8 <= R <= 12:
            logger.info("Initializing MoDL 2D model for R10")
            return modl_2d_r10
        elif 13 <= R <= 17:
            logger.info("Initializing MoDL 2D model for R15")
            return modl_2d_r15
        elif 18 <= R <= 22:
            logger.info("Initializing MoDL 2D model for R20")
            return modl_2d_r20
        elif 28 <= R <= 32:
            logger.info("Initializing MoDL 2D model for R30")
            return modl_2d_r30
        else:
            # default is old model
            logger.info("Initializing MoDL 2D model trained only at R25")
            return modl_2d
