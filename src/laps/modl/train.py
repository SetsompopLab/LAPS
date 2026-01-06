import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from accelerate.utils import set_seed
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

import wandb
from laps import DATA_DIR
from laps.dataloaders.kspace_dataset import KspaceDataset
from laps.dataloaders.labels import IMAGE_KEY, KSP_KEY, MASK_KEY, MPS_KEY, PRIOR_KEY
from laps.loggers import ImageLogger
from laps.metrics import PSNR
from laps.modl.models import Modl
from laps.modl.utils import conjugate_gradient
from laps.mri import (
    retrospective_undersample_mask,
    set_retrospective_undersample_parameters,
)
from laps.recon.linops import CartesianSenseLinop
from laps.recon.reconstructor import ReconstructorOutput
from laps.utils import compress_coils, create_exp_dir, get_torch_device, torch_resize

# not needed for single batch size
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


class KspaceDatasets(Enum):
    SLAM = "slam"
    FASTMRI = "fastmri"
    MODL = "modl"


kspace_datasets = {
    KspaceDatasets.SLAM: KspaceDataset(
        csv_path=DATA_DIR / "slam_train_with_ksp_im_sizes.csv",
        data_dir=DATA_DIR / "slam_train_with_ksp",
        dataset_id=0,
        verbose=False,
    ),
    KspaceDatasets.FASTMRI: KspaceDataset(
        csv_path=DATA_DIR / "fastmri_with_ksp_im_sizes.csv",
        data_dir=DATA_DIR / "fastmri_with_ksp",
        dataset_id=1,
        verbose=False,
    ),
    KspaceDatasets.MODL: KspaceDataset(
        csv_path=DATA_DIR / "modl_paper_im_sizes.csv",
        data_dir=DATA_DIR / "modl_paper",
        dataset_id=2,
        verbose=False,
    ),
}


@dataclass
class Config:
    device: int = 2
    seed: int = 42

    project: str = "modl-slamming"
    exp_name: str = "modl"
    exp_dir: Optional[Path] = None

    # Dataset
    datasets: Sequence[KspaceDatasets] = field(
        default_factory=lambda: [KspaceDatasets.SLAM]
    )
    val_frac: float = 0.01

    # Model parameters
    n_layers: int = 5
    n_filters: int = 64
    unroll_iters: int = 1
    N_COILS: int = 10
    use_prior: bool = False

    # I/O
    num_workers: int = 8
    log_interval: int = 500
    eval_interval: int = 1000
    checkpoint_path: Optional[str] = None

    # Training details
    lr: float = 1e-3
    batch_size: int = 8
    n_epochs: int = 1000
    n_train_steps: int = 10000

    # possible training configs
    train_dc_lambda: bool = True
    scale_denoiser: bool = False  # false for stage 1, true for stage 2
    modl_weights_init_scale: Optional[float] = None
    normalize_recons_to_target: bool = False
    modl_norm_type: str = "instance-affine"

    # Simulation enabling
    sim: bool = True
    sim_noise_std: float = 1e-4
    im_size: tuple = field(default_factory=lambda: (256, 256))

    # Retrospective undersampling parameters
    retrospective_undersampling: bool = True
    vd_factor_1d: float = 0.8
    vd_factor_2d: float = 1.5
    R_1d: Sequence[int] = field(default_factory=lambda: [])
    R_2d: Sequence[int] = field(default_factory=lambda: [])


def psnr_batch(y_batch, y_pred_batch):
    maxes = y_batch.abs().amax(dim=(-2, -1), keepdim=True)
    return np.mean(PSNR(y_batch / maxes, y_pred_batch / maxes)).item()


def get_modl_datasets() -> Tuple[List[KspaceDataset], List[KspaceDataset]]:
    """
    Since modl already has train/test split up.
    """
    ds = kspace_datasets[KspaceDatasets.MODL].dataset.copy()
    data_dir = str(kspace_datasets[KspaceDatasets.MODL].data_dir)
    ds = ds[ds["prefix"] == "trn"]
    ds["recon_path"] = data_dir + "/" + ds["recon_path"]
    ds["data_path"] = data_dir + "/" + ds["data_path"]
    ds = ds.reset_index(drop=True)
    train_ds = KspaceDataset(
        csv_path=DATA_DIR / "modl_paper_im_sizes.csv",
        data_dir=DATA_DIR / "modl_paper",
        dataset_id=0,
        verbose=False,
    )
    train_ds.dataset = ds
    train_ds.print_stats()

    ds = kspace_datasets[KspaceDatasets.MODL].dataset.copy()
    ds = ds[ds["prefix"] == "tst"]
    ds["recon_path"] = data_dir + "/" + ds["recon_path"]
    ds["data_path"] = data_dir + "/" + ds["data_path"]
    ds = ds.reset_index(drop=True)
    val_ds = KspaceDataset(
        csv_path=DATA_DIR / "modl_paper_im_sizes.csv",
        data_dir=DATA_DIR / "modl_paper",
        dataset_id=0,
        verbose=False,
    )
    val_ds.dataset = ds
    val_ds.print_stats()

    train_datasets = [train_ds]
    val_datasets = [val_ds]

    return train_datasets, val_datasets


def get_datasets(
    cfg: Config,
) -> Tuple[List[KspaceDataset], List[KspaceDataset], int, int]:
    MIN_LOADER_SIZE = 6
    MIN_VAL_CHOICE_SIZE = 300

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    # MODL case
    if KspaceDatasets.MODL in cfg.datasets:
        train_datasets, val_datasets = get_modl_datasets()
        if len(cfg.datasets) == 1:
            return (
                train_datasets,
                val_datasets,
                len(train_datasets[0]),
                len(val_datasets[0]),
            )
    else:
        train_datasets, val_datasets = [], []

    # Make full dataset table with all datasets
    joint_dataset = None
    cols = [
        "recon_path",
        "data_path",
        "quality",
        "is_middle_slice",
        "Nc",  # number of coils
        "Nx",  # number of pixels in x direction
        "Ny",  # number of pixels in y direction
        "im_size",  # image size
    ]
    for dataset in cfg.datasets:
        if dataset == KspaceDatasets.MODL:
            continue

        df = kspace_datasets[dataset].dataset

        if dataset == KspaceDatasets.SLAM:
            logger.info("Trimming SLAM dataset based on R_1d and R_2d parameters.")
            if len(cfg.R_1d) == 0:
                df = df[~(df["AccelNumDim"] == 1)]
            if len(cfg.R_2d) == 0:
                df = df[~(df["AccelNumDim"] == 2)]
            df = df.reset_index(drop=True)

        data_dir = str(kspace_datasets[dataset].data_dir)
        df["recon_path"] = data_dir + "/" + df["recon_path"]
        df["data_path"] = data_dir + "/" + df["data_path"]
        ds = df[cols].copy()
        if joint_dataset is None:
            joint_dataset = ds
        else:
            joint_dataset = pd.concat([joint_dataset, ds], ignore_index=True)

    assert joint_dataset is not None, "No datasets found. Check your dataset paths."
    joint_dataset.reset_index(drop=True, inplace=True)

    if cfg.batch_size == 1:
        # Don't worry about im_size for batch_size 1
        val_inds = np.random.choice(
            joint_dataset.index,
            size=int(len(joint_dataset) * cfg.val_frac),
            replace=False,
        )
        train_inds = np.setdiff1d(joint_dataset.index, val_inds)
        train_dataset = joint_dataset.iloc[train_inds].reset_index(drop=True)
        val_dataset = joint_dataset.iloc[val_inds].reset_index(drop=True)
        train_datasets.append(
            KspaceDataset(
                csv_path=train_dataset,
                data_dir="",
                dataset_id=0,
                verbose=False,
            )
        )
        val_datasets.append(
            KspaceDataset(
                csv_path=val_dataset,
                data_dir="",
                dataset_id=0,
                verbose=False,
            )
        )
    else:
        # Split into train/val
        possible_val_inds = joint_dataset[
            joint_dataset["im_size"].map(
                joint_dataset["im_size"].value_counts() > MIN_VAL_CHOICE_SIZE
            )
        ].index
        val_inds = np.random.choice(
            possible_val_inds,
            size=int(len(possible_val_inds) * cfg.val_frac),
            replace=False,
        )
        train_inds = np.setdiff1d(joint_dataset.index, val_inds)
        train_dataset = joint_dataset.iloc[train_inds].reset_index(drop=True)
        val_dataset = joint_dataset.iloc[val_inds].reset_index(drop=True)

        # drop any subsets with too few samples
        train_dataset = train_dataset[
            train_dataset["im_size"].map(
                train_dataset["im_size"].value_counts() >= MIN_LOADER_SIZE
            )
        ]
        val_dataset = val_dataset[
            val_dataset["im_size"].map(
                val_dataset["im_size"].value_counts() >= MIN_LOADER_SIZE
            )
        ]

        # make a dataloader for each dataset
        train_sizes = train_dataset["im_size"].unique()
        train_sizes = sorted(
            train_sizes,
            key=lambda x: train_dataset[train_dataset["im_size"] == x].shape[0],
            reverse=True,
        )
        for i, size in enumerate(train_sizes):
            subset = train_dataset[train_dataset["im_size"] == size].reset_index(
                drop=True
            )
            kspace_dataset = KspaceDataset(
                csv_path=subset,
                data_dir="",
                dataset_id=i,
                verbose=False,
            )
            train_datasets.append(kspace_dataset)

        val_sizes = val_dataset["im_size"].unique()
        val_sizes = sorted(
            val_sizes,
            key=lambda x: val_dataset[val_dataset["im_size"] == x].shape[0],
            reverse=True,
        )
        for i, size in enumerate(val_sizes):
            subset = val_dataset[val_dataset["im_size"] == size].reset_index(drop=True)
            kspace_dataset = KspaceDataset(
                csv_path=subset,
                data_dir="",
                dataset_id=i,
                verbose=False,
            )
            val_datasets.append(kspace_dataset)

    # sort datasets by lengths
    train_datasets.sort(key=lambda x: len(x), reverse=True)
    val_datasets.sort(key=lambda x: len(x), reverse=True)

    n_train = 0
    n_val = 0
    for i in range(len(train_datasets)):
        n_train += len(train_datasets[i])
    for i in range(len(val_datasets)):
        n_val += len(val_datasets[i])

    return train_datasets, val_datasets, n_train, n_val


def collate_fn_orig(batch, fixed_coils=32):
    # Collate function to handle variable-sized tensors
    for b in batch:
        n_coils = min(b[MPS_KEY].shape[0], fixed_coils)
        _, ksp, mps = compress_coils(b[KSP_KEY], n_coils, b[KSP_KEY], b[MPS_KEY])
        if fixed_coils > n_coils:
            ksp = torch.nn.functional.pad(ksp, (0, 0, 0, 0, 0, fixed_coils - n_coils))
            mps = torch.nn.functional.pad(mps, (0, 0, 0, 0, 0, fixed_coils - n_coils))
        b[KSP_KEY] = ksp
        b[MPS_KEY] = mps

    keys = [k for k in batch[0].keys() if k != "group_id"]
    stack_keys = [IMAGE_KEY, PRIOR_KEY, KSP_KEY, MPS_KEY, MASK_KEY]
    collated_batch = dict()
    for key in keys:
        collated_batch[key] = []
        for item in batch:
            collated_batch[key].append(item[key])

        # no longer stacking ksp, maps, mask
        if key in stack_keys:
            if isinstance(collated_batch[key][0], torch.Tensor):
                collated_batch[key] = torch.stack(collated_batch[key])
            elif isinstance(collated_batch[key][0], np.ndarray):
                collated_batch[key] = torch.tensor(
                    np.stack(collated_batch[key]), dtype=torch.float32
                )
            else:
                collated_batch[key] = np.array(collated_batch[key])

    return collated_batch


def prepare_modl_batch(
    batch: Dict[str, torch.Tensor],
    cfg: Config,
    device,
    seed: Optional[int] = None,
):

    if seed is not None:
        set_seed(seed)

    targs = batch[IMAGE_KEY].to(device)
    targs = targs.to(torch.complex64)

    if cfg.use_prior:
        priors = batch[PRIOR_KEY].to(device)
        priors = priors.to(torch.complex64)
    else:
        priors = None

    # make each target normalized
    norms = targs.norm(dim=(-2, -1), keepdim=True)
    targs = targs / norms

    B = targs.shape[0]

    mps = batch[MPS_KEY].to(device).to(torch.complex64)
    mask = batch[MASK_KEY].to(device).to(torch.complex64)

    if cfg.retrospective_undersampling:
        for i in range(B):
            if seed is not None:
                seed = seed + i
            retro_params = sample_retrospective_parameters(cfg, seed=seed)

            # don't undersample if R_2D = 1
            if (retro_params["R_2D"] is not None) and (retro_params["R_2D"] == 1):
                if (retro_params["R_1D"] is not None) and (retro_params["R_1D"] == 1):
                    continue

            mask[i] = (
                retrospective_undersample_mask(
                    torch.abs(mask[i]),
                    verbose=False,
                    seed=seed,
                    **retro_params,
                )
                .to(device)
                .to(torch.complex64)
            )

    # supports varied ishapes, so we use a wrapper
    A = CartesianSenseLinop(
        mps=mps,
        mask=mask,
        batch_size=B,
        ishape=targs.shape[1:],
    )
    A = A.to(device)

    # prep kspace
    if cfg.sim:
        ksp = A(targs)  # get kspace from targets
        ksp += torch.randn_like(ksp) * cfg.sim_noise_std  # add noise
        ksp = (ksp * mask[:, None]).to(torch.complex64)  # apply mask
    else:
        ksp = batch[KSP_KEY].to(device).to(torch.complex64)
        if cfg.retrospective_undersampling:
            ksp *= mask[:, None].to(torch.complex64)

    return targs, priors, ksp, mask, A


def normalize_analytic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        scale = torch.linalg.vecdot(
            x.flatten(start_dim=1).abs(),
            y.flatten(start_dim=1).abs(),
            dim=1,
        ) / (torch.linalg.vector_norm(x.flatten(start_dim=1), dim=1) ** 2 + 1e-8)
    x = x * scale.unsqueeze(1).unsqueeze(1)
    return x


def get_cg_recon(
    A,
    ksp,
    lambda_l2=1e-3,
    max_iterations=25,
):
    """
    Get the conjugate gradient reconstruction for the given kspace and linear operator A.
    """
    ATA = lambda x: A.N(x) + lambda_l2 * x

    with torch.no_grad():
        recons_cg = conjugate_gradient(
            ATA,
            A.H(ksp),
            max_iterations=max_iterations,
            verbose=False,
        )
    return recons_cg


def log_modl_batch(
    targs,
    recons,
    recons_cg,
    priors,
    cfg: Config,
    masks,
    step,
    ImLogger: ImageLogger,
    val=False,
    n_images_to_log: int = 4,
):
    if priors is not None:
        priors = priors.detach().cpu()

    N_masks = min(len(masks), n_images_to_log)
    masks_plot_H = np.max([masks[i].shape[0] for i in range(N_masks)])
    masks_plot_W = np.max([masks[i].shape[1] for i in range(N_masks)])
    masks_plot = torch.zeros((N_masks, masks_plot_H, masks_plot_W), dtype=torch.float32)
    for i in range(N_masks):
        masks_plot[i] = torch_resize(masks[i].real, (masks_plot_H, masks_plot_W))

    recons = ReconstructorOutput(recons.detach().cpu())
    recons_cg = ReconstructorOutput(recons_cg.detach().cpu())

    prefix = "val/" if val else "train/"

    ImLogger.log_images(
        recon_dict={"CG": recons_cg, "MODL": recons},
        targets=targs.cpu(),
        priors=priors,
        mask=masks_plot,
        wandb_prefix=prefix,
        n_images_to_log=n_images_to_log,
        step=step,
    )

    return ImLogger


def sample_retrospective_parameters(cfg: Config, seed: Optional[int] = None) -> dict:
    """
    Samples the retrospective parameters for training example.

    For training multiple undersampling patterns jointly.
    """
    if seed is not None:
        np.random.seed(seed)

    R_1d = cfg.R_1d
    R_2d = cfg.R_2d
    vd_factor_1d = cfg.vd_factor_1d
    vd_factor_2d = cfg.vd_factor_2d

    if len(R_1d) > 0:
        R_1d = int(np.random.choice(R_1d).item())
    elif len(R_1d) == 0:
        R_1d = None

    if len(R_2d) > 0:
        R_2d = int(np.random.choice(R_2d).item())
    elif len(R_2d) == 0:
        R_2d = None

    has_2d = R_2d is not None

    if R_1d is not None:
        assert isinstance(
            R_1d, int
        ), f"R_1d must be an integer at this point, but got {R_1d} of type {type(R_1d)}"
    if R_2d is not None:
        assert isinstance(
            R_2d, int
        ), f"R_2d must be an integer at this point, but got {R_2d} of type {type(R_2d)}"

    params = set_retrospective_undersample_parameters(
        R_1D=R_1d,
        R_2D=R_2d,
        vd_factor_1d=vd_factor_1d,
        vd_factor_2d=vd_factor_2d,
    )

    # Speedup for 2D undersampling
    if has_2d:
        params["undersampling_type"] = "2D"

    return params


def train(cfg: Config):
    set_seed(cfg.seed)

    device = get_torch_device(cfg.device)

    dataset_str = "_".join([dataset.value for dataset in cfg.datasets])

    exp_name = f"{cfg.exp_name}_{cfg.unroll_iters}unroll_{dataset_str}"
    if cfg.sim:
        exp_name += f"_sim"

    if cfg.use_prior:
        raise NotImplementedError(
            "Prior usage is not implemented yet. Please set use_prior=False."
        )

    if cfg.exp_dir is None:
        cfg.exp_dir = create_exp_dir(Path("./logs"), exp_name)
    logger.add(cfg.exp_dir / "log-{time}")

    wandb.init(project=cfg.project, name=exp_name, config=vars(cfg))

    if isinstance(cfg.R_1d, int):
        cfg.R_1d = [cfg.R_1d]
    if isinstance(cfg.R_2d, int):
        cfg.R_2d = [cfg.R_2d]

    assert (
        len(cfg.R_1d) + len(cfg.R_2d)
    ) > 0, "At least one of R_1d or R_2d must be specified for training."

    # List of datasets where each dataset has the same kspace size
    train_datasets, val_datasets, n_train, n_val = get_datasets(cfg)
    collate_fn = lambda batch: collate_fn_orig(batch, fixed_coils=cfg.N_COILS)

    # pre-form dataloaders if only 1 train_dataset / val_dataset
    has_one_train_dataloader = False
    has_one_val_dataloader = False
    train_dataloader = None
    val_dataloader = None
    if len(train_datasets) == 1:
        has_one_train_dataloader = True
    if len(val_datasets) == 1:
        has_one_val_dataloader = True

    logger.info(f"HAS ONE TRAIN DATASET: {has_one_train_dataloader}")
    logger.info(f"HAS ONE VAL DATASET: {has_one_val_dataloader}")
    if has_one_train_dataloader:
        train_dataloader = torch.utils.data.DataLoader(
            train_datasets[0],
            shuffle=True,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
        )
    if has_one_val_dataloader:
        val_dataloader = torch.utils.data.DataLoader(
            val_datasets[0],
            shuffle=False,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
        )

    model = Modl(
        n_layers=cfg.n_layers,
        unroll_iters=cfg.unroll_iters,
        n_filters=cfg.n_filters,
        use_prior=cfg.use_prior,
        train_dc_lambda=cfg.train_dc_lambda,
        scale_denoiser=cfg.scale_denoiser,
        norm_type=cfg.modl_norm_type,
        weights_init_scale=cfg.modl_weights_init_scale,
    )

    if cfg.checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {cfg.checkpoint_path}")
        model.load_state_dict(
            torch.load(cfg.checkpoint_path, map_location="cpu"), strict=True
        )
    elif cfg.unroll_iters > 1:
        logger.warning(
            "Training a model with unroll_iters > 1 from scratch is not recommended. "
            "It is better to pre train for a few epochs with unroll_iters=1, and then train with unroll_iters > 1."
        )

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_f_real = nn.MSELoss()
    loss_f = lambda x, y: 0.5 * (
        loss_f_real(x.real, y.real) + loss_f_real(x.imag, y.imag)
    )

    if cfg.sim:
        logger.info(f"Running on simulated kspace with noise std={cfg.sim_noise_std}")

    best_val_loss = np.inf
    step = 0

    ImLogger = ImageLogger(log_dir=cfg.exp_dir)

    total_train_steps = min(n_train * cfg.n_epochs, cfg.n_train_steps)
    pbar = tqdm(total=total_train_steps, desc="Training")

    while True:
        model.train()
        for train_dataset in train_datasets:
            if not has_one_train_dataloader:
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    shuffle=True,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    collate_fn=collate_fn,
                )
            assert train_dataloader is not None
            for batch in train_dataloader:
                targs, priors, ksp, masks, A = prepare_modl_batch(
                    batch,
                    cfg,
                    device,
                )

                optimizer.zero_grad()

                recons = model(ksp, A, priors)

                if cfg.normalize_recons_to_target:
                    recons = normalize_analytic(recons, targs)

                loss = loss_f(recons, targs)

                if torch.isnan(loss):
                    logger.error(
                        f"NaN loss encountered at step {step * cfg.batch_size}. Skipping optimizer.step()."
                    )
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    1.0,
                )  # Global gradient clipping

                optimizer.step()

                if (step * cfg.batch_size) % cfg.log_interval == 0:
                    # CG Recon
                    ImLogger = log_modl_batch(
                        targs,
                        recons,
                        get_cg_recon(A, ksp),
                        priors,
                        cfg,
                        masks,
                        step * cfg.batch_size,
                        ImLogger,
                        val=False,
                    )

                log_dict = {}
                log_dict["train_loss"] = loss.item()
                log_dict["lamda"] = model.dc.lamda.item()
                wandb.log(log_dict, step=step * cfg.batch_size)
                pbar.set_description(
                    f"Step {step * cfg.batch_size}, Loss: {loss.item():.4e}, Lamda: {model.dc.lamda.item():.4e}"
                )
                pbar.update(1)

                if (step > 0) & ((step * cfg.batch_size) % cfg.eval_interval == 0):
                    running_psnr = 0
                    model.eval()
                    with torch.no_grad():
                        val_loss = 0
                        n_val_batches = 0
                        pbar_val = tqdm(
                            total=n_val,
                            desc="Validation Batches",
                            leave=False,
                        )
                        for vi, val_dataset in enumerate(val_datasets):
                            if not has_one_val_dataloader:
                                val_dataloader = torch.utils.data.DataLoader(
                                    val_dataset,
                                    shuffle=False,
                                    batch_size=cfg.batch_size,
                                    num_workers=cfg.num_workers,
                                    collate_fn=collate_fn,
                                )
                            assert val_dataloader is not None
                            for l_idx, batch in enumerate(val_dataloader):
                                targs, priors, ksp, masks, A = prepare_modl_batch(
                                    batch,
                                    cfg,
                                    device,
                                    seed=cfg.seed,
                                )
                                recons = model(ksp, A, priors)

                                if cfg.normalize_recons_to_target:
                                    recons = normalize_analytic(recons, targs)

                                val_loss += loss_f(recons, targs).item()
                                running_psnr += psnr_batch(targs.abs(), recons.abs())
                                n_val_batches += 1
                                pbar_val.update(cfg.batch_size)

                                if (vi == 0) and (l_idx == (len(val_dataloader) // 2)):
                                    ImLogger = log_modl_batch(
                                        targs,
                                        recons,
                                        get_cg_recon(A, ksp),
                                        priors,
                                        cfg,
                                        masks,
                                        step * cfg.batch_size,
                                        ImLogger,
                                        val=True,
                                    )
                    pbar_val.close()
                    val_loss /= n_val_batches
                    running_psnr /= n_val_batches
                    logger.info(f"Validation loss: {val_loss}, PSNR: {running_psnr}")
                    wandb.log(
                        {"val_loss": val_loss, "val_psnr": running_psnr},
                        step=step * cfg.batch_size,
                    )
                    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                    if val_loss < best_val_loss:
                        logger.info(f"Saving best model with val loss: {val_loss}")
                        best_val_loss = val_loss
                        torch.save(cpu_state_dict, cfg.exp_dir / "best_model.pth")

                    torch.save(cpu_state_dict, cfg.exp_dir / f"model_latest.pth")

                    model.train()

                step += 1
                if step >= cfg.n_train_steps:
                    break

            if step >= cfg.n_train_steps:
                break

    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state_dict, cfg.exp_dir / f"model_final.pth")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    train(cfg)
