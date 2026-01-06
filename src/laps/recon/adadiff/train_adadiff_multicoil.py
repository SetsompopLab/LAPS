import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from laps import PROJECT_ROOT
from laps.dataloaders import (
    COLLATE_IMAGE_KEY,
    COLLATE_TEXT_KEY,
    IMAGE_KEY,
    ConcatDataset,
    LoaderType,
)
from laps.dataset import LAPS_DATASETS
from laps.recon.adadiff.utils.EMA import EMA
from laps.recon.adadiff.utils.networks.discriminator import (
    Discriminator_large,
    Discriminator_small,
)
from laps.recon.adadiff.utils.networks.ncsnpp_generator_adagn import NCSNpp


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1.0 - torch.exp(2.0 * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1.0 - eps_small) + eps_small
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients:
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = torch.sqrt(1 - self.a_s_cum**2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    x_t = (
        extract(coeff.a_s_cum, t, x_start.shape) * x_start
        + extract(coeff.sigmas_cum, t, x_start.shape) * noise
    )
    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = (
        extract(coeff.a_s, t + 1, x_start.shape) * x_t
        + extract(coeff.sigmas, t + 1, x_start.shape) * noise
    )
    return x_t, x_t_plus_one


class Posterior_Coefficients:
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (
                torch.tensor([1.0], dtype=torch.float32, device=device),
                self.alphas_cumprod[:-1],
            ),
            0,
        )
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(
            coefficients.posterior_log_variance_clipped, t, x_t.shape
        )
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = 1 - (t == 0).type(torch.float32)
        return (
            mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise
        )

    sample_x_pos = p_sample(x_0, x_t, t)
    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)  # .to(x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
    return x


def setup_file_logging(logger, exp_path):
    """Setup additional file logging alongside accelerate logger"""
    # Create log file path with timestamp
    log_file = exp_path / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create rotating file handler (100 MB max size, keep 7 backup files)
    file_handler = RotatingFileHandler(
        log_file,
        mode="a",
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=7,  # Keep 7 backup files (similar to 7 days retention)
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    # Get the underlying logger and add handler
    underlying_logger = getattr(logger, "logger", logger)
    if hasattr(underlying_logger, "addHandler"):
        underlying_logger.addHandler(file_handler)
        underlying_logger.setLevel(logging.DEBUG)
    else:
        # Fallback: add to logger directly if it's already a logging.Logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

    print(f"File logging setup complete: {log_file}")
    print(f"Log rotation: 100MB max size, 7 backup files")
    print(f"Logger type: {type(logger)}, underlying: {type(underlying_logger)}")
    print(
        f"Logger handlers: {len(underlying_logger.handlers) if hasattr(underlying_logger, 'handlers') else 'N/A'}"
    )

    return log_file


def setup_console_logging(logger):
    """Setup console logging for accelerate logger"""
    # Get the underlying logger
    underlying_logger = getattr(logger, "logger", logger)

    # Check if console handler already exists
    handlers = getattr(underlying_logger, "handlers", [])
    has_console_handler = any(
        isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr
        for handler in handlers
    )

    if not has_console_handler:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)

        # Create formatter for console (simpler than file)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        # Add handler to the logger
        if hasattr(underlying_logger, "addHandler"):
            underlying_logger.addHandler(console_handler)
            underlying_logger.setLevel(logging.INFO)
        else:
            # Fallback
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)


def normalize_image_to_uint8(image):
    """Normalize image to uint8 following the VAE pattern"""
    draw_img = image.copy()
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 0.1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def log_images_wandb(images, wandb_prefix, step, args, accelerator, max_images=4):
    """Log complex MRI images to wandb showing magnitude and phase"""
    if args.use_wandb and args.log_images:
        with torch.no_grad():
            # Detach and limit number of images to avoid overwhelming wandb
            images = images.detach().cpu()
            if images.size(0) > max_images:
                images = images[:max_images]

            # Handle complex images by showing both magnitude and phase
            if (
                images.dim() == 4 and images.size(1) == 2
            ):  # [B, C, H, W] with complex representation
                # Convert to complex tensor
                complex_images = torch.complex(images[:, 0], images[:, 1])

                # Get magnitude and phase
                magnitude = torch.abs(complex_images).numpy()
                phase = torch.angle(complex_images).numpy()

                # Normalize phase to [0, 1] range for visualization
                phase_normalized = (phase + np.pi) / (2 * np.pi)

                wandb_images = []
                for i in range(magnitude.shape[0]):
                    # Normalize magnitude and phase for display
                    mag_norm = normalize_image_to_uint8(magnitude[i])
                    phase_norm = normalize_image_to_uint8(phase_normalized[i])

                    # Concatenate magnitude and phase horizontally for comparison
                    combined_image = np.concatenate([mag_norm, phase_norm], axis=-1)

                    wandb_images.append(
                        wandb.Image(combined_image, caption=f"{i}_mag_phase")
                    )

                accelerator.log({f"images/{wandb_prefix}": wandb_images}, step=step)

            else:
                # Fallback for non-complex images
                # Normalize to [0, 1]
                images_norm = (images - images.min()) / (
                    images.max() - images.min() + 1e-8
                )

                wandb_images = []
                for i in range(images_norm.size(0)):
                    img = images_norm[i].detach().cpu().numpy()
                    if img.shape[0] == 1:  # Single channel
                        img = img[0]
                    else:  # Multi-channel, take first channel
                        img = img[0]
                    wandb_images.append(wandb.Image(img, caption=f"{i}"))

                accelerator.log({f"images/{wandb_prefix}": wandb_images}, step=step)


def train(args):
    # Setup logging
    exp = args.exp
    parent_dir = (
        PROJECT_ROOT
        / "logs"
        / "adadiff"
        / args.dataset
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    exp_path = parent_dir / exp
    os.makedirs(exp_path, exist_ok=True)

    project_config = ProjectConfiguration(
        project_dir=exp_path,
        logging_dir=exp_path,
        automatic_checkpoint_naming=True,
        total_limit=2,
    )

    accelerator = Accelerator(
        project_config=project_config, log_with="wandb" if args.use_wandb else None
    )
    accelerator.init_trackers(
        "adadiff",
        config=args,
        init_kwargs={"wandb": {"name": args.exp, "dir": exp_path}},
    )
    device = accelerator.device

    args.exp_path = exp_path
    set_seed(args.seed)

    # Get accelerate logger and setup file logging
    logger = get_logger(__name__)

    # Setup console logging (needed for older accelerate versions)
    setup_console_logging(logger)

    if accelerator.is_main_process:
        log_file = setup_file_logging(logger, exp_path)
        logger.info(f"Experiment path: {args.exp_path}")
        logger.info(f"Log file: {log_file}")

    batch_size = args.batch_size
    nz = args.nz  # latent dimension
    args.attn_resolutions = tuple(args.attn_resolutions)

    logger.info(f"Batch size: {batch_size}, Image size: {args.image_size}")
    logger.info(f"Number of timesteps: {args.num_timesteps}")

    # dataset=CreateDatasetMultiCoil()

    dataset_kwargs = dict(
        image_size=(args.image_size, args.image_size),
        num_channels=args.num_channels,
        randcrop=False,
        random_flip=True,
        rot_degree=1,
        complex_dropout_frac=0.0,
        complex_global_phase_modulation=False,
        complex_output=True,
        logger=logger,
    )
    dataset_list = [
        LoaderType.SLAM,
        LoaderType.SLAM_DICOM,
        LoaderType.FASTMRI_T1,
        LoaderType.FASTMRI_T2,
        LoaderType.FASTMRI_FLAIR,
        LoaderType.FASTMRI_T1POST,
        LoaderType.FASTMRI_T1PRE,
    ]
    dataset = ConcatDataset(
        [LAPS_DATASETS[d].get_dataset("train", **dataset_kwargs) for d in dataset_list]
    )

    # Create a smaller dataset for debugging
    if args.debug_dataset_size is not None and args.debug_dataset_size > 0:
        subset_indices = list(range(min(args.debug_dataset_size, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    logger.info(f"Dataset loaded with {len(dataset)} samples")

    def collate_fn(examples):
        """
        Stack the pixel values and tokenize the captions and return as a dict
        for torch.dataloader collation.
        """

        pixel_values = torch.stack([example[IMAGE_KEY] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        return {COLLATE_IMAGE_KEY: pixel_values}

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    netG = NCSNpp(args)
    netD = Discriminator_large(
        nc=2 * args.num_channels,
        ngf=args.ngf,
        t_emb_dim=args.t_emb_dim,
        act=nn.LeakyReLU(0.2),
    )

    optimizerD = optim.Adam(
        netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2)
    )
    optimizerG = optim.Adam(
        netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2)
    )
    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerG, args.num_epoch, eta_min=1e-5
    )
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerD, args.num_epoch, eta_min=1e-5
    )

    (
        netG,
        optimizerG,
        data_loader,
        schedulerG,
        netD,
        optimizerD,
        schedulerD,
    ) = accelerator.prepare(
        netG, optimizerG, data_loader, schedulerG, netD, optimizerD, schedulerD
    )

    if accelerator.is_main_process:
        if not os.path.exists(exp_path):
            copy_source(__file__, exp_path)
            shutil.copytree(
                Path(__file__).parent / "utils/networks",
                os.path.join(exp_path, "utils/networks"),
            )

        logger.info(f"Experiment directory: {exp_path}")
        logger.info(
            f"Model architectures: Generator={type(netG).__name__}, Discriminator={type(netD).__name__}"
        )

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    global_step, epoch, init_epoch = 0, 0, 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume_from_ckpt}")
        accelerator.load_state(args.resume_from_ckpt)

    # Training loop
    for epoch in range(init_epoch, args.num_epoch + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        total_iter_time = 0.0

        logger.info(f"Starting epoch {epoch}/{args.num_epoch}")

        # for iteration, (x, y) in enumerate(data_loader):
        for iteration, batch in enumerate(data_loader):
            iter_start_time = time.time()
            x = batch[COLLATE_IMAGE_KEY].to(dtype=torch.float32)
            for p in netD.parameters():
                p.requires_grad = True
            netD.zero_grad()
            # sample from p(x_0)
            real_data = x
            x_t_1 = torch.randn_like(real_data)
            # sample t
            t = torch.randint(
                0, args.num_timesteps, (real_data.size(0),), device=device
            )
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True
            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            errD_real = F.softplus(-D_real)
            errD_real = errD_real.mean()
            accelerator.backward(errD_real, retain_graph=True)
            if args.lazy_reg is None:
                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True
                )[0]
                grad_penalty = (
                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                accelerator.backward(grad_penalty)
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                        outputs=D_real.sum(), inputs=x_t, create_graph=True
                    )[0]
                    grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    accelerator.backward(grad_penalty)

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errD_fake = F.softplus(output)
            errD_fake = errD_fake.mean()
            accelerator.backward(errD_fake)
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            # update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            t = torch.randint(
                0, args.num_timesteps, (real_data.size(0),), device=device
            )
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errG = F.softplus(-output)
            errG = errG.mean()
            accelerator.backward(errG)
            optimizerG.step()
            global_step += 1

            # Calculate iteration timing
            iter_time = time.time() - iter_start_time
            total_iter_time += iter_time
            avg_iter_time = total_iter_time / (iteration + 1)

            # Accumulate losses for epoch logging
            epoch_g_loss += errG.item()
            epoch_d_loss += errD.item()
            num_batches += 1

            # Log training progress
            if iteration % args.log_freq == 0:
                logger.info(
                    f"Epoch {epoch:3d} | Iter {iteration:4d}/{len(data_loader)} | G Loss: {errG.item():.4f} |"
                    f"D Loss: {errD.item():.4f} |"
                    f"Time: {iter_time:.2f}s (avg: {avg_iter_time:.2f}s) | Global Step: {global_step}"
                )

                accelerator.log(
                    {
                        "train/generator_loss": errG.item(),
                        "train/discriminator_loss": errD.item(),
                        "train/discriminator_real_loss": errD_real.item(),
                        "train/discriminator_fake_loss": errD_fake.item(),
                        "train/epoch": epoch,
                        "train/iteration": iteration,
                        "train/global_step": global_step,
                        "train/learning_rate_g": (
                            schedulerG.get_last_lr()[0]
                            if not args.no_lr_decay
                            else args.lr_g
                        ),
                        "train/learning_rate_d": (
                            schedulerD.get_last_lr()[0]
                            if not args.no_lr_decay
                            else args.lr_d
                        ),
                    },
                    step=global_step,
                )

                # Log sample images during training
                if args.log_images and iteration % args.image_log_freq == 0:
                    # Generate full clean samples for logging (not noisy posterior)
                    with torch.no_grad():
                        x_t_1 = torch.randn_like(real_data)
                        clean_samples = sample_from_model(
                            pos_coeff, netG, args.num_timesteps, x_t_1, T, args
                        )
                    log_images_wandb(
                        clean_samples,
                        f"train_samples",
                        global_step,
                        args,
                        accelerator,
                    )

        # End of epoch logging
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches

        logger.info(
            f"Epoch {epoch} completed | Avg G Loss: {avg_g_loss:.4f} | Avg D Loss: {avg_d_loss:.4f}"
        )

        accelerator.log(
            {
                "epoch/avg_generator_loss": avg_g_loss,
                "epoch/avg_discriminator_loss": avg_d_loss,
                "epoch/epoch": epoch,
            },
            step=global_step,
        )

        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()

        # Only main process logs images and saves checkpoints
        if accelerator.is_main_process:
            # Generate full samples for logging and saving
            x_t_1 = torch.randn_like(real_data)
            fake_sample = sample_from_model(
                pos_coeff, netG, args.num_timesteps, x_t_1, T, args
            )

            # Log generated images to wandb (no file saving)
            if args.log_images:
                log_images_wandb(
                    fake_sample,
                    f"epoch_samples",
                    global_step,
                    args,
                    accelerator,
                )

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.optimizer.swap_parameters_with_ema(
                        store_params_in_ema=True
                    )
                # Unwrap model to avoid shared tensor removal warnings
                model_to_save = accelerator.unwrap_model(netG)
                torch.save(model_to_save.state_dict(), exp_path / f"netG_{epoch}.pth")
                logger.info(f"Model checkpoint saved: netG_{epoch}.pth")
                if args.use_ema:
                    optimizerG.optimizer.swap_parameters_with_ema(
                        store_params_in_ema=True
                    )

        if args.save_content:
            if epoch % args.save_content_every == 0:
                logger.info("Saving checkpoint...")
                accelerator.save_state(
                    exp_path / "checkpoint", safe_serialization=False
                )
                logger.info(f"Checkpoint saved at epoch {epoch}")

    logger.info("Training completed!")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("adadiff parameters")
    parser.add_argument(
        "--seed", type=int, default=1024, help="seed used for initialization"
    )
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--resume_from_ckpt",
        type=str,
        default=None,
        help="path to checkpoint to resume from",
    )
    parser.add_argument("--image_size", type=int, default=256, help="size of image")
    parser.add_argument("--num_channels", type=int, default=2, help="channel of image")
    parser.add_argument(
        "--centered", action="store_false", default=True, help="-1,1 scale"
    )
    parser.add_argument("--use_geometric", action="store_true", default=False)
    parser.add_argument(
        "--beta_min", type=float, default=0.1, help="beta_min for diffusion"
    )
    parser.add_argument(
        "--beta_max", type=float, default=20.0, help="beta_max for diffusion"
    )
    parser.add_argument(
        "--num_channels_dae",
        type=int,
        default=64,
        help="number of initial channels in denosing model",
    )
    parser.add_argument(
        "--n_mlp", type=int, default=3, help="number of mlp layers for z"
    )
    parser.add_argument(
        "--ch_mult",
        nargs="+",
        type=int,
        default=[1, 1, 1, 2, 2],
        help="channel multiplier",
    )
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="number of resnet blocks per scale",
    )
    parser.add_argument(
        "--attn_resolutions",
        nargs="+",
        type=int,
        default=[18],
        help="resolution of applying attention",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument(
        "--resamp_with_conv",
        action="store_false",
        default=True,
        help="always up/down sampling with conv",
    )
    parser.add_argument(
        "--conditional", action="store_false", default=True, help="noise conditional"
    )
    parser.add_argument("--fir", action="store_false", default=True, help="FIR")
    parser.add_argument("--fir_kernel", default=[1, 3, 3, 1], help="FIR kernel")
    parser.add_argument(
        "--skip_rescale", action="store_false", default=True, help="skip rescale"
    )
    parser.add_argument(
        "--resblock_type",
        default="biggan",
        help="tyle of resnet block, choice in biggan and ddpm",
    )
    parser.add_argument(
        "--progressive",
        type=str,
        default="none",
        choices=["none", "output_skip", "residual"],
        help="progressive type for output",
    )
    parser.add_argument(
        "--progressive_input",
        type=str,
        default="residual",
        choices=["none", "input_skip", "residual"],
        help="progressive type for input",
    )
    parser.add_argument(
        "--progressive_combine",
        type=str,
        default="sum",
        choices=["sum", "cat"],
        help="progressive combine method.",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="positional",
        choices=["positional", "fourier"],
        help="type of time embedding",
    )
    parser.add_argument(
        "--fourier_scale", type=float, default=16.0, help="scale of fourier transform"
    )
    parser.add_argument("--not_use_tanh", action="store_true", default=False)
    # geenrator and training
    parser.add_argument("--exp", default="slam", help="name of experiment")
    parser.add_argument("--dataset", default="slam", help="name of dataset")
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--num_timesteps", type=int, default=8)
    parser.add_argument("--z_emb_dim", type=int, default=256)
    parser.add_argument("--t_emb_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=12, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--lr_g", type=float, default=1.6e-4, help="learning rate g")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="learning rate d")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)
    parser.add_argument(
        "--use_ema", action="store_true", default=True, help="use EMA or not"
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.999, help="decay rate for EMA"
    )
    parser.add_argument("--r1_gamma", type=float, default=1.0, help="coef for r1 reg")
    parser.add_argument("--lazy_reg", type=int, default=10, help="lazy regulariation.")
    parser.add_argument("--save_content", action="store_true", default=True)
    parser.add_argument(
        "--save_content_every",
        type=int,
        default=50,
        help="save content for resuming every x epochs",
    )
    parser.add_argument(
        "--save_ckpt_every", type=int, default=25, help="save ckpt every x epochs"
    )

    # Logging arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=True,
        help="use wandb for experiment tracking",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="AdaDiff", help="wandb project name"
    )
    parser.add_argument(
        "--wandb_run_suffix",
        type=str,
        default="debug",
        help="suffix to add to wandb run name",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=20,
        help="logging frequency for training metrics",
    )
    parser.add_argument(
        "--log_images", action="store_true", default=True, help="log images to wandb"
    )
    parser.add_argument(
        "--image_log_freq",
        type=int,
        default=200,
        help="frequency for logging images during training",
    )
    parser.add_argument(
        "--save_image_freq",
        type=int,
        default=1000,
        help="frequency for saving images to disk",
    )
    parser.add_argument(
        "--debug_dataset_size",
        type=int,
        default=None,
        help="use only N samples for debugging (default: use full dataset)",
    )

    args = parser.parse_args()
    train(args)
