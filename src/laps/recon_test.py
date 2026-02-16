"""
Inference script for MRI reconstruction using various baselines against our proposed LAPS recon.
Requires SLAM dataset to be downloaded as described in the README.md file.
"""

import os
from dataclasses import dataclass, field
import gc
from math import ceil, sqrt
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import tyro
from loguru import logger

import wandb
from laps import PROJECT_ROOT, WANDB_DIR, TMP_DIR
from laps.configs import recon as recon_configs
from laps.configs.recon_modl import get_modl_config
from laps.dataloaders.labels import IMAGE_KEY, KSP_KEY, MASK_KEY, MPS_KEY, PRIOR_KEY
from laps.dataloaders.paired_dataloader import (
    AccelerationType,
    Dimension,
    PairedDataset,
    ScanPlane,
    ScanType,
)
from laps.dataset import LAPS_PAIRED_DATASETS
from laps.loggers import ImageLogger, PatchBasedMetricsLogger, TimeLogger
from laps.mri import (
    retrospective_undersample_mask,
    set_retrospective_undersample_parameters,
)
from laps.recon import (
    AdaDiffParams,
    CGParams,
    DiffusersReconstructor,
    LACSParams,
    LDMParams,
    ModlParams,
    NERPParams,
    ReconParams,
    Reconstructor,
    get_reconstructors,
)
from laps.recon.coils import acs_from_ksp, csm_from_espirit
from laps.recon.linops import CartesianSenseLinop, linop
from laps.utils import (
    clear_cache,
    compress_coils,
    create_exp_dir,
    get_torch_device,
    torch_resize,
)


@dataclass
class Config:
    # Selected GPU
    device_idx: int = 0

    # wandb project
    project_name: str = "laps-inference"
    exp_name: str = "mrm-release"
    exp_dir: Optional[str] = None  # if None, will be created based on exp_name
    base_dir: str = "inference"
    seed: int = 42

    # test type
    accel_dim: Literal[1, 2] = 1  # 1D or 2D accel.
    R: float = 5  # acceleration

    # dataset and filters
    dataset: str = "slam-test"
    scan_plane_filter: ScanPlane = ScanPlane.ALL  # filter by ax, sag, cor
    scan_type_filter: ScanType = ScanType.ALL  # filter by t1, t2, flair, etc
    middle_slices_only: bool = True  # filter to only use middle slices
    minimum_quality: Optional[int] = None
    shuffle: bool = False
    max_recons: Optional[int] = None
    log_interval: int = 1
    mod_jump: int = 1  # how many adjacent slices to jump
    n_slices: Optional[int] = 5  # number of slices to subsample per volume

    # For retrospective undersampling for real data
    retrospective_undersampling: bool = True
    vd_factor_1D: float = 0.8
    vd_factor_2D: float = 1.5
    use_gt_mps: bool = True # use ground truth sensitivity maps for faster inference for demo.
    seed_undersampling: bool = False

    recons: Dict[str, ReconParams] = field(
        default_factory=lambda: {
            "CG": CGParams(max_iter=15, lamda_l2=1e-4),
            "CS": LACSParams(
                lamda_1 = 5e-5,
                lamda_2 = 0,
                max_iter = 1,
                max_fista_iter = 100,
            ),
            "LACS": LACSParams(
                lamda_1=1e-5, lamda_2=1e-4, max_iter=5, max_fista_iter=25
            ),
            "MODL": ModlParams(),
            "NERP": NERPParams(),
            "AdaDiff": AdaDiffParams(),
            "CAPS": recon_configs.sd_caps_medvae_4,
            "LAPS": recon_configs.sd_laps_medvae_4,
        }
    )

    # metrics
    metric_methods: list = field(default_factory=lambda: ["psnr", "ssim"])

    # misc
    debug_slc: Optional[int] = None # debug mode: only run for a single slice
    wandb_enable: bool = False # toggle wandb on/off for debugging


@dataclass
class ReconLoggers:
    MetricsLogger: PatchBasedMetricsLogger
    ImLogger: ImageLogger
    TimeLogger: TimeLogger


def setup(args: Config) -> Tuple[Config, Sequence[torch.device], ReconLoggers]:
    """
    Initialize the script enviroment.
    """
    Tlogger = TimeLogger()
    Tlogger.setup_start()

    # setup tmpdir
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR, exist_ok=True)
    os.environ["TMPDIR"] = str(TMP_DIR)

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_torch_device(args.device_idx)

    args.mod_jump = args.mod_jump if not args.middle_slices_only else 1
    args.n_slices = args.n_slices if not args.middle_slices_only else None

    # default experiment name
    args.exp_name = f"{args.exp_name}-{args.dataset}"
    if args.scan_plane_filter != ScanPlane.ALL:
        args.exp_name += f"-{args.scan_plane_filter.value}"
    if args.scan_type_filter != ScanType.ALL:
        args.exp_name += f"-{args.scan_type_filter.value}"
    if args.middle_slices_only:
        args.exp_name += "-mid-slices"
    if args.minimum_quality is not None:
        args.exp_name += f"-qmin{args.minimum_quality}"
    if args.retrospective_undersampling:
        args.exp_name += f"-D{args.accel_dim}R{args.R}"
    if args.use_gt_mps:
        args.exp_name += "-gt-mps"
    if args.max_recons is not None:
        args.exp_name += f"-max_samples{args.max_recons}"
    args.exp_name += f"-seed{args.seed}"

    if args.exp_dir is None:
        base_dir = PROJECT_ROOT
        exp_dir = create_exp_dir(Path(base_dir) / args.base_dir, args.exp_name)
        args.exp_dir = str(exp_dir)
    args.exp_dir = str(args.exp_dir)
    logger.add(os.path.join(args.exp_dir, "log-{time}"))

    # select MODL model based on undersampling tested.
    for r in args.recons.keys():
        if isinstance(args.recons[r], ModlParams):
            args.recons[r] = get_modl_config(args.accel_dim, args.R)

        if isinstance(args.recons[r], LDMParams):
            if args.accel_dim == 1:
                if args.R >= 7:
                    output_dc_config = {
                        "n_iters": 5,
                        "threshold": 1e-5,
                        "lambda_l2": 5e-4,
                        "lambda_ldm": 0.03,
                        "lambda_l2_from_data": False,
                        "avg_before_dc": False,
                    }
                else:
                    output_dc_config = {
                        "n_iters": 10,
                        "threshold": 1e-5,
                        "lambda_l2": 5e-4,
                        "lambda_ldm": 0.025,
                        "lambda_l2_from_data": False,
                        "avg_before_dc": False,
                    }
            else:
                if args.R <= 20:
                    output_dc_config = {
                        "n_iters": 18,
                        "threshold": 1e-5,
                        "lambda_l2": 1e-3,
                        "lambda_ldm": 0.03,
                        "lambda_l2_from_data": False,
                        "avg_before_dc": False,
                    }
                else:
                    output_dc_config = {
                        "n_iters": 16,
                        "threshold": 1e-5,
                        "lambda_l2": 1e-3,
                        "lambda_ldm": 0.03,
                        "lambda_l2_from_data": False,
                        "avg_before_dc": False,
                    }
            args.recons[r].output_dc_config = output_dc_config

    # init wandb
    if args.wandb_enable:
        wandb.init(
            project=args.project_name,
            name=f"{args.exp_name}-{args.exp_dir.split('/')[-1]}",
            config=vars(args),
            dir = WANDB_DIR,
        )

    # Init loggers
    MetricsLogger = PatchBasedMetricsLogger(
        recon_types=list(args.recons.keys()),
        metric_types=args.metric_methods,
        sim_percentile=90.0,
        dissim_percentile=10.0,
        comp_types=["samp_targ", "samp_prior", "targ_prior"],
        wandb_enable=args.wandb_enable,
        export_dir=os.path.join(args.exp_dir, "metrics"),
    )
    ImLogger = ImageLogger(
        log_dir=args.exp_dir,
        wandb_enable=args.wandb_enable,
    )
    loggers = ReconLoggers(
        MetricsLogger=MetricsLogger,
        ImLogger=ImLogger,
        TimeLogger=Tlogger,
    )

    return args, device, loggers


def load_ds(args: Config) -> PairedDataset:
    """
    Load dataset with specific filters and parameters.
    """
    load_dim = Dimension.MIDDLE_SLICE if args.middle_slices_only else Dimension.DIM_2D

    # scan we accidentally included twice with different views in test dataset
    test_dataset = LAPS_PAIRED_DATASETS[args.dataset].get_dataset(
        acceleration_filter=(
            AccelerationType.R_1D_FS
            if args.accel_dim == 1
            else AccelerationType.R_2D_FS
        ),
        scan_plane_filter=args.scan_plane_filter,
        scan_type_filter=args.scan_type_filter,
        minimum_quality=args.minimum_quality,
        load_dimension=load_dim,
        return_metadata=True,
        verbose=True,
        shuffle=args.shuffle,
        max_samples=args.max_recons,
        mod_jump=args.mod_jump,
        n_slices=args.n_slices,
    )

    return test_dataset


def instantiate_reconstructors(args: Config, device: torch.device):
    im_size_tmp = (256, 256)  # temporary, will be overwritten later
    for _, params in args.recons.items():
        params.device = device
        params.im_size = im_size_tmp
        params.debug = False

    # initialize all the reconstructors / bring into cache but not GPU memory
    reconstructors = get_reconstructors(
        args.recons, linop(im_size_tmp, im_size_tmp), idle_sd_models_on_cpu=True
    )

    return reconstructors


def reconstruct_batch(
    args: Config,
    reconstructors: Dict[str, Reconstructor],
    test_dataset: torch.utils.data.Dataset,
    ind: int,
    device: torch.device,
) -> dict:
    """
    Perform reconstruction on a single dataset item.
    """

    batch = test_dataset[ind]
    targs = batch[IMAGE_KEY].to(device)
    priors = batch[PRIOR_KEY].to(device)

    targs = targs.to(torch.complex64)
    priors = priors.to(torch.complex64)

    # Data
    mps_gt = batch[MPS_KEY].to(device)
    mask = batch[MASK_KEY].to(device)

    # metadata
    meta = batch["metadata"]

    # retrospective undersampling optional
    if args.retrospective_undersampling:
        if args.seed_undersampling:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        if args.accel_dim == 1:
            R1D = args.R
            R2D = None
        else:
            R1D = None
            R2D = args.R

        if args.R > 1:
            retro_params = set_retrospective_undersample_parameters(
                R_1D=R1D,
                R_2D=R2D,
                vd_factor_1d=args.vd_factor_1D,
                vd_factor_2d=args.vd_factor_2D,
            )
            mask = retrospective_undersample_mask(
                torch.abs(mask),
                verbose=True,
                **retro_params,
            )
        else:
            mask = torch.abs(mask)

        mask = mask.to(device).to(torch.complex64)

    # Kspace
    ksp = batch[KSP_KEY].to(device)

    # coil compression to at most 16 coils for memory
    _, ksp, mps_gt = compress_coils(ksp, 16, ksp, mps_gt)
    if args.retrospective_undersampling:
        ksp *= mask.unsqueeze(0)
    ksp = ksp.to(torch.complex64)

    # re-estimate sensitivity maps.
    if args.retrospective_undersampling and (args.R > 1) and not args.use_gt_mps:
        accel_ndim = meta["AccelNumDim"]
        if accel_ndim in [0, 1]:
            Nacs = retro_params["kwargs_1D"]["N_fs_max"]
        else:
            Nacs = int(ceil(retro_params["kwargs_2D"]["N_fs_max"] * sqrt(2)))
        mps = csm_from_espirit(
            acs_from_ksp(ksp, Nacs=Nacs, ndim=2).to(device),
            im_size=ksp.shape[1:],
            thresh=0.02,
            kernel_width=6,
            crp=0.94,
            verbose=False,
            use_cupy_for_blocks=False,
        )[0]
    else:
        mps = mps_gt.clone()

    # Set forward model
    im_size = targs.shape[-2:]
    A = CartesianSenseLinop(mps=mps, mask=mask, ishape=im_size)
    A = A.to(device)
    for recon_typ, reconstructor in reconstructors.items():
        reconstructor.forward_model = A
        if isinstance(reconstructor, DiffusersReconstructor):
            reconstructor.dc_args["forward_model"] = A

    # normalize prior to [0, 1] mag if possible
    AHb = A.H(ksp).abs()
    ksp /= AHb.quantile(0.999)
    targs /= targs.abs().max()
    priors /= priors.abs().max()

    # add batch dim
    targs = targs.unsqueeze(0)
    priors = priors.unsqueeze(0)
    ksp = ksp.unsqueeze(0)
    x0 = targs.clone()  # for debugging

    sens_mask = torch_resize(torch.linalg.vector_norm(mps, dim=0) > 0.5, im_size).cpu()

    # Run recons
    collected_res = {}
    for recon_type, reconstructor in reconstructors.items():
        clear_cache()
        logger.info(f"Running {recon_type} on slice {ind}...")
        output = reconstructor.reconstruct(
            ksp,
            x0=x0,
            priors=priors,
        )
        output.recon = output.recon.cpu() * sens_mask.unsqueeze(0)
        if output.error is not None:
            output.error = output.error.cpu() * sens_mask.unsqueeze(0)
        collected_res[recon_type] = output  # (output.recon, output.error)
        if output.extra_outputs is not None:
            for k, v in output.extra_outputs.items():
                meta[f"{recon_type}_{k}"] = v

    return dict(
        collected_res=collected_res,
        targs=targs.cpu(),
        priors=priors.cpu(),
        mask=A.mask.real.cpu(),
        metadata=meta,
    )


def batch_call(
    args: Config,
    reconstructors: Dict[str, Reconstructor],
    device: Sequence[torch.device],
    test_dataset: torch.utils.data.Dataset,
    idx: int,
    loggers: ReconLoggers,
) -> ReconLoggers:

    loggers.TimeLogger.recon_start()
    output = reconstruct_batch(
        args,
        reconstructors,
        test_dataset,
        idx,
        device,
    )
    loggers.TimeLogger.recon_end()

    # Logging
    loggers.TimeLogger.reporting_start()
    collected_res = output["collected_res"]
    targs = output["targs"]
    priors = output["priors"]
    mask = output["mask"][0]
    metadata = output["metadata"]
    recons_for_metrics = {
        recon_type: res.recon for recon_type, res in collected_res.items()
    }
    loggers.MetricsLogger.update_batch_metrics(
        recons_dict=recons_for_metrics,
        targs=targs,
        priors=priors,
        metadata=[metadata],
    )
    if idx % args.log_interval == 0:
        logger.info(f"Step {idx}: Logging to WandB...")
        loggers.ImLogger.log_images(
            recon_dict=collected_res,
            targets=targs,
            priors=priors,
            mask=mask,
            wandb_prefix="test/",
            step=idx,
        )
        loggers.MetricsLogger.log_batch_metrics(
            step=idx,
        )
    loggers.TimeLogger.reporting_end()

    return loggers


def main(args: Config):

    args, device, loggers = setup(args)

    # Load dataset
    test_dataset = load_ds(args)

    # instantiate reconstructors
    reconstructors = instantiate_reconstructors(args, device)

    loggers.TimeLogger.setup_end()

    # iterate through dataset
    if args.max_recons is None:
        args.max_recons = len(test_dataset)
    Nrecon = min(args.max_recons, len(test_dataset))
    
    for idx in range(Nrecon):
        logger.info(f"Processing recon {idx + 1}/{Nrecon}.")

        # Debugging: only run on a single slice
        if args.debug_slc is not None:
            idx = args.debug_slc
            logger.info(f"Debugging mode: running on slice {idx}.")

        # run on each device, and then log results
        loggers = batch_call(
            args,
            reconstructors,
            device,
            test_dataset,
            idx,
            loggers,
        )

        if args.debug_slc is not None:
            # Debugging: only run on a single slice
            break

    # Cleanup
    if args.wandb_enable:
        wandb.finish()
    logger.info(f"Done! Saved to {args.exp_dir}.")
    loggers.TimeLogger.report()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    args = tyro.cli(Config)
    main(args)
