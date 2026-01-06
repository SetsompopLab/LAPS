from typing import Optional

import matplotlib

# matplotlib.use("webagg")
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from loguru import logger
from tqdm import tqdm

from ...utils import logging

extra_dc_step_done = False

def cplx_encode(x, vae):
    if x.ndim == 2:
        x = x[None,]
    x = torch.stack(
        [x.real, x.imag], dim=1
    )
    return vae.encode(x).latent_dist.mode() * vae.config.scaling_factor

def cplx_decode(z, vae):
    z = vae.decode(
        z / vae.config.scaling_factor,
        return_dict = False,
    )[0]
    x = z[:, 0] + 1j * z[:, 1]
    if x.shape[0] == 1:
        x = x[0]
    return x


class SimplePlateauLrScheduler:

    def __init__(
        self,
        initial_lr_scale: float = 1,
        patience: int = 3,
        decay_factor: float = 0.2,
        min_lr_scale: float = 5e-4,
        verbose: bool = True,
    ):
        self.lr_scale = initial_lr_scale
        self.min_lr_scale = min_lr_scale
        self.minimum_achieved = False
        self.patience = patience
        self.decay_factor = decay_factor
        self.didnt_improve_count = 0
        self.prev_loss = torch.inf
        self.verbose = verbose

    def step(self, loss):

        if self.minimum_achieved:
            return

        if loss >= self.prev_loss * 0.95:
            self.didnt_improve_count += 1
        else:
            self.didnt_improve_count = 0
            self.prev_loss = loss

        if self.didnt_improve_count >= self.patience:
            if self.verbose:
                logger.info(
                    f"Plateau scheduler: reducing lr scale from {self.lr_scale} to {self.lr_scale * self.decay_factor}"
                )
            self.lr_scale *= self.decay_factor
            self.didnt_improve_count = 0

            if self.lr_scale < self.min_lr_scale:
                self.lr_scale = self.min_lr_scale
                self.minimum_achieved = True
                if self.verbose:
                    logger.info(
                        f"Minimum learning rate scale reached: {self.min_lr_scale}. Stopping further reductions."
                    )

    def get_lr_scale(self):
        return self.lr_scale


def predict_z0_hat(prediction_type, model_output, z_t, beta_prod_t, alpha_prod_t):
    """
    Predict the latent variable at time 0 based on the prediction type.
    From formula (12) of https://arxiv.org/pdf/2010.02502.pdf

    Args:
        prediction_type (str): The prediction type. Can be "epsilon", "sample", or "v_prediction".
        model_output (torch.Tensor): The output of the U-net model.
        z_t (torch.Tensor): The current latent variable.
        beta_prod_t (float): The product of the betas up to time t.
        alpha_prod_t (float): The product of the alphas up to time t.

    Returns:
        torch.Tensor: The predicted latent variable at time 0.

    """
    if prediction_type == "epsilon":
        z_0_hat = (z_t - model_output * (beta_prod_t ** (0.5))) / (
            alpha_prod_t ** (0.5)
        )
    elif prediction_type == "v_prediction":
        z_0_hat = (alpha_prod_t**0.5) * z_t - (beta_prod_t**0.5) * model_output
    elif prediction_type == "sample":
        z_0_hat = model_output
    return z_0_hat


def estimate_zt_from_z0_hat_for_ldps(
    prediction_type, model_output, z_0_opt, beta_prod_t, alpha_prod_t, z_t_orig=None
):
    """
    Undo the prediction of z_t based on the prediction type.
    For use after optimizing z0_hat using any data consistency method and getting back to z_t for ldps.

    Args:
        prediction_type (str): The prediction type. Can be "epsilon", "sample", or "v_prediction".
        model_output (torch.Tensor): The output of the U-net model.
        z_0_opt (torch.Tensor): The (DC-optimized) latent variable at time 0.
        beta_prod_t (float): The product of the betas up to time t.
        alpha_prod_t (float): The product of the alphas up to time t.
        z_t_orig (torch.Tensor): The original latent variable at time t used as the input to the Unet. Required for sample prediction.

    Returns:
        torch.Tensor: The estimated (and optimized) latent variable at time t.
    """

    if prediction_type == "epsilon":
        # undo predict_z0_hat()
        z_t_opt = (alpha_prod_t**0.5) * z_0_opt + (beta_prod_t**0.5) * model_output

    elif prediction_type == "v_prediction":
        # undo predict_z0_hat()
        z_t_opt = (z_0_opt + (beta_prod_t**0.5) * model_output) / (alpha_prod_t**0.5)

    elif prediction_type == "sample":
        # require z_t and do update based on optimized z_0_hat
        assert z_t_orig is not None, "z_t must be provided for sample prediction"

        z_t_opt = z_t_orig + (alpha_prod_t**0.5) * (z_0_opt - model_output)

        # same as doing the following:
        # # estimate noise from original model output estimate of z_0
        # z_0_model = model_output
        # pred_epsilon = (z_t_orig - alpha_prod_t ** (0.5) * z_0_model) / beta_prod_t ** (0.5)
        # # now just estimate z_t | z_0_hat, eps like epsilon prediction
        # z_t = alpha_prod_t ** 0.5 * z_0_hat + beta_prod_t ** 0.5 * pred_epsilon

    return z_t_opt


def normalize_scale(
    x, per=0.01, out_range=(-1, 1), top=None, bottom=None, return_top_bottom=False
):
    """
    Normalize the input tensor `x` based on percentiles and map it to the specified output range.
    This will clamp the input tensor based on the specified percentiles (same for top and bottom),
    then normalize it to the required range.

    Args:
        x (torch.Tensor): The input tensor to be normalized.
        per (float, optional): The percentile value used for clamping the tensor. Defaults to 0.01.
        out_range (tuple, optional): The output range to map the normalized tensor to. Defaults to (-1, 1).

    Returns:
        torch.Tensor: The normalized tensor.

    """
    top_per = 1 - per
    reshaped_x = rearrange(x, "b c h w -> b c (h w)")

    # clamp based on percentiles
    if per < 1e-4:
        bottom = torch.min(reshaped_x, dim=-1)[0]
        top = torch.max(reshaped_x, dim=-1)[0]
    else:
        if bottom is None:
            bottom = torch.kthvalue(reshaped_x, int(per * reshaped_x.size(-1)), dim=-1)[
                0
            ]
        if top is None:
            top = torch.kthvalue(
                reshaped_x, int(top_per * reshaped_x.size(-1)), dim=-1
            )[0]
    clamped_tensor = torch.clamp(
        x, min=bottom[..., None, None], max=top[..., None, None]
    )

    # normalize to [0, 1]
    normalized_tensor = (clamped_tensor - bottom[..., None, None]) / (top - bottom)[
        ..., None, None
    ]

    # svale to out_range
    normalized_tensor = normalized_tensor * (out_range[1] - out_range[0]) + out_range[0]

    if return_top_bottom:
        return normalized_tensor, top, bottom
    else:
        return normalized_tensor


def ksp_loss(y_hat, y):
    """
    Calculates the k-space loss between the predicted k-space data (y_hat) and the ground truth k-space data (y).
    Assumes both are complex numbers.

    Args:
        y_hat (torch.Tensor): Predicted k-space data.
        y (torch.Tensor): Ground truth k-space data.

    Returns:
        torch.Tensor: The k-space loss.

    """
    real_loss = torch.sum(
        (y_hat.real - y.real) ** 2, dim=(-1, -2, -3)
    )  # sum over all dimensions except batch
    imag_loss = torch.sum((y_hat.imag - y.imag) ** 2, dim=(-1, -2, -3))

    return 0.5 * (real_loss + imag_loss)


def batched_conjugate_gradient(
    AHA, AHb, x0, max_iterations: int = 10, tolerance: float = 1e-6, lambda_l2=1e-3
):
    """
    Solves a linear system of equations using the Complex Conjugate Gradient method.
    This function solves Ax = b, by minimizing ||Ax - b||_2^2 + lambda_l2 * ||x||_2^2.

    Args:
        AHA (callable): A function that computes the product of A and its Hermitian transpose (AHA). Assumes that the
        input size is batched [B, C, H, W]
        AHb (torch.Tensor): The product of A's Hermitian transpose and the target vector b.
        x0 (torch.Tensor): The initial guess for the solution.
        max_iterations (int, optional): The maximum number of iterations. Defaults to 10.
        tolerance (float, optional): The convergence tolerance. Defaults to 1e-6.
        lambda_l2 (float, optional): The L2 regularization parameter. Defaults to 1e-3.

    Returns:
        torch.Tensor: The solution to the linear system of equations.
    """

    if max_iterations < 1:
        return x0

    AHA_wrapper = lambda x: AHA(x) + lambda_l2 * x

    r = AHb - AHA_wrapper(x0)  # [B, H, W]
    p = r.clone()  # [B, H, W]
    x = x0.clone()  # [B, H, W]
    r_dot_r = torch.real(torch.conj(r) * r).sum(dim=(-1, -2))

    for _ in range(max_iterations):
        AHAp = AHA_wrapper(p)
        alpha = r_dot_r / torch.real(torch.conj(p) * AHAp).sum(dim=(-1, -2))
        x = x + alpha.unsqueeze(-1).unsqueeze(-1) * p
        r = r - alpha.unsqueeze(-1).unsqueeze(-1) * AHAp

        new_r_dot_r = torch.real(torch.conj(r) * r).sum(dim=(-1, -2))
        beta = new_r_dot_r / r_dot_r
        p = r + beta.unsqueeze(-1).unsqueeze(-1) * p
        r_dot_r = new_r_dot_r

        # Since we are batching, this will stop if all of the elements in the batch have passed the threshold. Not ideal
        # but it's the best we can do.
        if torch.sqrt(r_dot_r).max() < tolerance:
            break

    return x


def stochastic_resampling(z_0_hat, z_t, scheduler, t, prev_t, gamma=40, simple=False):
    """
    Perform stochastic resampling of latent variables based on the paper https://arxiv.org/pdf/2307.08123.pdf.
    Given some z_0, we estimate the corresponding z_t.

    Args:
        z_0_hat (torch.Tensor): A refined estimation of the latents at time 0.
        z_t (torch.Tensor): The current latents at time t.
        scheduler (Scheduler): The scheduler object.
        t (int): The current time step.
        prev_t (int): The previous time step.
        gamma (float, optional): The gamma parameter. Defaults to 40.
        simple (bool, optional): Whether to use the simple resampling method. Defaults to False.

    Returns:
        torch.Tensor: The resampled latent variable.
    """
    # some schedulers don't have final_alphas_cumprod
    try:
        final_alpha_cumprod = (
            scheduler.final_alpha_cumprod
            if scheduler.final_alpha_cumprod is not None
            else 1
        )
    except:
        final_alpha_cumprod = 1

    # shortcut since the weight on z_0_hat should be 0 at t=0
    if t == 0:
        return z_t

    alpha_prod_t = scheduler.alphas_cumprod[t] if t > 0 else final_alpha_cumprod
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_t] if prev_t > 0 else final_alpha_cumprod
    )

    # The simple version just samples from p(z_t|z_0, y)
    if simple:
        return torch.sqrt(alpha_prod_t) * z_0_hat + torch.sqrt(1 - alpha_prod_t) * (
            torch.randn_like(z_0_hat)
        )

    var_t = (
        gamma
        * ((1 - alpha_prod_t_prev) / alpha_prod_t)
        * (1 - (alpha_prod_t / alpha_prod_t_prev))
    )

    # equation (13) in https://arxiv.org/pdf/2307.08123.pdf
    mu = (var_t * torch.sqrt(alpha_prod_t) * z_0_hat + (1 - alpha_prod_t) * z_t) / (
        var_t + (1 - alpha_prod_t)
    )
    noise_std = torch.sqrt((var_t * (1 - alpha_prod_t)) / (var_t + (1 - alpha_prod_t)))
    noise = torch.randn_like(mu) * noise_std
    z_t_resapmled = mu + noise

    return z_t_resapmled

@torch.no_grad()
def ddim_inversion(
    scheduler,
    unet,
    z0: torch.Tensor,
    timesteps: torch.Tensor,
    cross_attention_kwargs = None,
    ) -> torch.Tensor:
    """
    Implement DDIM Inversion from DDIM paper
    """

    timesteps_reverse = timesteps.flip(dims=(0,)).clone()

    # add 0
    if timesteps_reverse[0] != 0:
        timesteps_reverse = torch.cat(
            [torch.tensor([0], device=timesteps.device, dtype=timesteps.dtype), timesteps_reverse]
        )

    latents = z0.clone()

    for i in tqdm(range(len(timesteps_reverse) - 1), desc="DDIM Inversion", leave=False):
                    
        t = timesteps_reverse[i]
        tp1 = timesteps_reverse[i + 1] 

        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_tp1 = scheduler.alphas_cumprod[tp1]
        beta_prod_tp1 = 1 - alpha_prod_tp1

        # predict the noise residual
        model_pred = unet(
            scheduler.scale_model_input(latents, tp1),
            tp1,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        if scheduler.config.prediction_type == "epsilon":
            pred_epsilon = model_pred
        elif scheduler.config.prediction_type == "sample":
            pred_epsilon = (latents - alpha_prod_tp1 ** (0.5) * model_pred) / beta_prod_tp1 ** (0.5)
        elif scheduler.config.prediction_type == "v_prediction":
            pred_epsilon = (alpha_prod_tp1**0.5) * model_pred + (beta_prod_tp1**0.5) * latents
        else:
            raise ValueError(
                f"prediction_type given as {config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # latents = (latents - (1-alpha_prod_t).sqrt()*pred_epsilon)*(alpha_prod_tp1.sqrt()/alpha_prod_t.sqrt()) + (1-alpha_prod_tp1).sqrt()*pred_epsilon
        noise_scale = (1-alpha_prod_tp1).sqrt() - ((1-alpha_prod_t).sqrt() * (alpha_prod_tp1.sqrt()/alpha_prod_t.sqrt()))
        latent_scale = (alpha_prod_tp1.sqrt()/alpha_prod_t.sqrt())
        latents = latent_scale * latents + noise_scale * pred_epsilon

    return latents

def dc_latent_optimization(
    forward_model,
    autoencoder,
    opt_params,
    y,
    z_0,
    forward_scales: Optional[torch.Tensor] = None,
    generator=None,
    verbose=True,
):
    """
    Performs data consistency optimization in latent space.

    Args:
        forward_model (torch.nn.Module): The forward model used for reconstruction.
        autoencoder (torch.nn.Module): The latent autoencoder.
        opt_params (dict): Optimization parameters including n_iters, lr, and threshold.
        y (torch.Tensor): Measurements.
        z_0 (torch.Tensor): The initial latent variable.
        scale_factor (float): The scale factor used for decoding.

    Returns:
        torch.Tensor: The optimized latent variable.
    """
    assert z_0.device == y.device, "z_0 and y must be on the same device"

    n_iters = opt_params["n_iters"]
    lr = opt_params["lr"]
    threshold = opt_params["threshold"]
    latent_consistency = opt_params.get("latent_consistency", 0)

    # jointly optimize scale s to that minimizes ||s * A(D(z_0_hat)) - y||
    if forward_scales is not None:
        s = forward_scales.detach().clone()
    else:
        s = (
            forward_model.adjoint(y)
            .abs()
            .max(-1, keepdim=True)
            .values.max(-2, keepdim=True)
            .values.clone()
        )
        s = s[:, None]  # coil dimension

    with torch.enable_grad():
        z_opt = z_0.clone().detach().requires_grad_(True)
        s = s.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z_opt, s], lr=lr)

        # calculate normalization parameters based on the initial image
        # with torch.no_grad():
        #     im_opt = torch.mean(autoencoder.decode(
        #         z_opt / autoencoder.config.scaling_factor,
        #         return_dict=False,
        #         generator=generator,
        #     )[0], dim=1, keepdim=True)  # [B, 1, H, W], average on the channel dimension
        #     _, top, bottom = normalize_scale(
        #         im_opt, per=norm_per, out_range=(0, 1), return_top_bottom=True
        #     )

        progress_bar = tqdm(total=n_iters, disable=not verbose, leave=False)

        for i in range(n_iters):
            optimizer.zero_grad()
            # Estimate image
            im_opt = autoencoder.decode(
                z_opt / autoencoder.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[
                0
            ]  # [B, C, H, W], average on the channel dimension
            loss = 0
            if latent_consistency > 0:
                im_enc_opt = (
                    autoencoder.encode(im_opt).latent_dist.sample()
                    * autoencoder.config.scaling_factor
                )
                consistency_loss = torch.mean((z_opt - im_enc_opt) ** 2)
                loss = loss + latent_consistency * consistency_loss

            # recover complex data
            im_cplx = im_opt[:, 0] + 1j * im_opt[:, 1]

            # Pass through the forward model
            y_hat = forward_model(im_cplx)

            # adjust scaling
            y_hat = y_hat * s  # (s + 1j * 0)

            batch_loss = ksp_loss(y_hat, y)
            dc_loss = torch.mean(batch_loss)
            loss = loss + dc_loss
            loss.backward()
            optimizer.step()

            if verbose:
                cl_string = (
                    ""
                    if latent_consistency == 0
                    else f" Consistency loss: {consistency_loss}"
                )
                progress_bar.set_description(
                    f"DC latent optimization {i+1}/{n_iters} - Loss: {loss}, DC loss: {dc_loss}"
                    + cl_string
                )  # Update the progress bar description with loss
                progress_bar.update(1)

            # ideally the stopping point for different images would be different but we want batching for efficiency
            if batch_loss.max() <= threshold:
                break

    s = s.requires_grad_(False).detach().clone()

    progress_bar.close()

    return z_opt, s


def dc_image_optimization(
    forward_model,
    autoencoder,
    opt_params,
    y,
    z_0,
    forward_scales: Optional[torch.Tensor] = None,
    generator=None,
    encode_output=True,
):
    """
    Perform data consistency optimization in image space.

    Args:
        forward_model (object): The forward model used for image reconstruction.
        autoencoder (object): The autoencoder model.
        opt_params (dict): Optimization parameters including n_iters, threshold, lambda_l2, etc.
        y (torch.Tensor): Measurements.
        z_0 (torch.Tensor): The initial latent space representation.
        scale_factor (float): The scale factor used for encoding and decoding.
        encode_output (bool, optional): Whether to encode the output image back to latent space. Defaults to True.

    Returns:
        torch.Tensor: The optimized latent space representation if encode_output is True, otherwise the optimized image.
    """
    assert z_0.device == y.device, "z_0 and y must be on the same device"

    n_iters = opt_params["n_iters"]
    threshold = opt_params["threshold"]
    lambda_l2 = opt_params["lambda_l2"]

    # jointly optimize scale s to that minimizes ||s * A(D(z_0_hat)) - y||
    if forward_scales is not None:
        s = forward_scales.detach().clone()
    else:
        s = (
            forward_model.adjoint(y)
            .abs()
            .max(-1, keepdim=True)
            .values.max(-2, keepdim=True)
            .values.clone()
        )

    # Decode
    x_0 = autoencoder.decode(
        z_0 / autoencoder.config.scaling_factor,
        return_dict=False,
        generator=generator,
    )[
        0
    ]  # [B, C, H, W]

    # Complex
    x_cplx = x_0[:, 0] + 1j * x_0[:, 1]  # [B, H, W]

    # Apply scaling
    x_cplx = x_cplx * s

    lamda_ldm = 0  # 1e-3 -> TODO: play with
    AHA = lambda x: forward_model.normal(x) + lamda_ldm * x
    AHb = forward_model.adjoint(y) + lamda_ldm * x_cplx

    # Optimize
    x_cplx_opt = batched_conjugate_gradient(
        AHA,
        AHb,
        x_cplx,
        max_iterations=n_iters,
        tolerance=threshold,
        lambda_l2=lambda_l2,
    )

    # undo scaling
    x_cplx_opt = x_cplx_opt / s

    # go back to channel space
    if autoencoder.config.in_channels == 2:
        x_0_opt = torch.stack([x_cplx_opt.real, x_cplx_opt.imag], dim=1)
    elif autoencoder.config.in_channels == 3:
        x_0_opt = torch.stack(
            [x_cplx_opt.real, x_cplx_opt.imag, x_cplx_opt.abs()], dim=1
        )
    else:
        raise ValueError("Only 2 or 3 channels are supported for complex data")

    # If we want to go back to latent space
    if encode_output:
        z_0_opt = (
            autoencoder.encode(x_0_opt).latent_dist.mode()
            * autoencoder.config.scaling_factor
        )

        return z_0_opt
    else:
        return x_0_opt


def dc_latent_manifold_optimization(
    forward_model,
    autoencoder,
    unet,
    opt_params,
    y,
    z_t,
    t,
    alpha_prod_t,
    beta_prod_t,
    prediction_type,
    lr_scale=1,
    forward_scales: Optional[torch.Tensor] = None,
    generator=None,
    verbose=True,
    unet_args={},
):
    """
    Performs data consistency optimization in latent space.

    Args:
        forward_model (torch.nn.Module): The forward model used for reconstruction.
        autoencoder (torch.nn.Module): The latent autoencoder.
        unet (torch.nn.Module): The U-Net model for diffusive denoising.
        opt_params (dict): Optimization parameters including n_iters, lr, and threshold.
        y (torch.Tensor): Measurements.
        z_t (torch.Tensor): latent variable at timestep t
        prompt_embeds (torch.Tensor): The prompt embeddings.
        timestep_cond (torch.Tensor): The timestep condition.
        added_cond_kwargs (dict): Additional condition arguments.
        t (int): The current timestep.
        alpha_prod_t (float): The product of the alphas up to time t.
        beta_prod_t (float): The product of the betas up to time t.
        prediction_type (str): The prediction type used for z_0_hat.
        cross_attention_kwargs (dict): The cross-attention arguments.
        lr_scale (float, optional): The learning rate scale. Defaults to 1.
        generator (torch.Generator, optional): The random number generator. Defaults to None.
        verbose (bool, optional): Whether to display the progress bar. Defaults to True.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): (z_t_opt, model_output, loss, scale)

            z_t_opt: The optimized latent variable
            model_output_opt: The optimized output of the U-net model for the optimized latent variable
            loss: The loss value

    """
    assert z_t.device == y.device, "z_0 and y must be on the same device"

    n_iters = opt_params["n_iters"]

    # global extra_dc_step_done
    # if t < 5 and not extra_dc_step_done:
    #     n_iters = 20
    #     extra_dc_step_done = True
    # elif extra_dc_step_done:
    #     n_iters = 0

    if n_iters == 0:
        loss = torch.tensor(0.0).to(z_t.device)

    lr = lr_scale * opt_params["lr"]
    threshold = opt_params["threshold"]
    latent_consistency = opt_params.get("latent_consistency", 0)

    init_z_t = z_t.clone().detach()

    with torch.enable_grad():
        z_t_opt = z_t.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z_t_opt], lr=lr)
        progress_bar = tqdm(total=n_iters, disable=((not verbose) or (n_iters == 1)), leave=False)

        for i in range(n_iters):
            optimizer.zero_grad()
            # Estimate z_0_hat
            model_output = unet(
                z_t_opt,
                t,
                return_dict=False,
                **unet_args,
            )[0]

            z_0_hat = predict_z0_hat(
                prediction_type,
                model_output,
                z_t_opt,
                beta_prod_t,
                alpha_prod_t,
            )

            # Estimate image
            im_opt = autoencoder.decode(
                z_0_hat / autoencoder.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[
                0
            ]  # [B, 3, H, W], average on the channel dimension
            loss = 0
            if latent_consistency > 0:
                im_enc_opt = (
                    autoencoder.encode(im_opt).latent_dist.mode()
                    * autoencoder.config.scaling_factor
                )
                consistency_loss = torch.mean((z_0_hat - im_enc_opt) ** 2)
                loss = loss + latent_consistency * consistency_loss

            # complex output
            im_opt_cplx = im_opt[:, 0] + 1j * im_opt[:, 1]

            y_hat = forward_model(im_opt_cplx)

            # adjust scaling
            with torch.no_grad():
                s = torch.linalg.vecdot(
                    y_hat.flatten(1), y.flatten(1)
                ) / torch.linalg.vecdot(y_hat.flatten(1), y_hat.flatten(1))
                s = s[:, None, None, None]
            y_hat = y_hat * s  # (s + 1j * 0)

            batch_loss = ksp_loss(y_hat, y)
            dc_loss = torch.mean(batch_loss)

            if opt_params.get("z_t_lam", 0) > 0:
                z_t_loss = torch.mean((z_t_opt - init_z_t) ** 2)
                loss = loss + opt_params.get("z_t_lam", 0) * z_t_loss

            loss = loss + dc_loss
            loss.backward()
            optimizer.step()

            if verbose:
                cl_string = (
                    ""
                    if latent_consistency == 0
                    else f" Consistency loss: {consistency_loss.item()}"
                )
                z_t_string = (
                    ""
                    if opt_params.get("z_t_lam", 0) == 0
                    else f" Z_t loss: {z_t_loss.item()}"
                )
                progress_bar.set_description(
                    f"DC latent optimization {i+1}/{n_iters} - Loss: {loss}, DC loss: {dc_loss}"
                    + cl_string
                    + z_t_string
                )  # Update the progress bar description with loss
                progress_bar.update(1)

            # ideally the stopping point for different images would be different but we want batching for efficiency
            if batch_loss.max() <= threshold:
                break

    progress_bar.close()

    # fix scaling factor
    s = s.requires_grad_(False).detach().clone()

    # do one final unet call to get the most updated model output
    model_output_opt = unet(
        z_t_opt,
        t,
        return_dict=False,
        **unet_args,
    )[0]

    return z_t_opt, model_output_opt, loss, s


def dc_plds(
    forward_model,
    autoencoder,
    unet,
    opt_params,
    y,
    z_t,
    t,
    alpha_prod_t,
    beta_prod_t,
    prediction_type,
    lr_scale=1,
    generator=None,
    verbose=True,
    unet_args={},
):
    """
    Performs data consistency optimization in latent space.

    Args:
        forward_model (torch.nn.Module): The forward model used for reconstruction.
        autoencoder (torch.nn.Module): The latent autoencoder.
        unet (torch.nn.Module): The U-Net model for diffusive denoising.
        opt_params (dict): Optimization parameters including n_iters, lr, and threshold.
        y (torch.Tensor): Measurements.
        z_t (torch.Tensor): latent variable at timestep t
        prompt_embeds (torch.Tensor): The prompt embeddings.
        timestep_cond (torch.Tensor): The timestep condition.
        added_cond_kwargs (dict): Additional condition arguments.
        t (int): The current timestep.
        alpha_prod_t (float): The product of the alphas up to time t.
        beta_prod_t (float): The product of the betas up to time t.
        prediction_type (str): The prediction type used for z_0_hat.
        cross_attention_kwargs (dict): The cross-attention arguments.
        lr_scale (float, optional): The learning rate scale. Defaults to 1.
        generator (torch.Generator, optional): The random number generator. Defaults to None.
        verbose (bool, optional): Whether to display the progress bar. Defaults to True.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): (z_t_opt, model_output, loss, scale)

            z_t_opt: The optimized latent variable
            model_output_opt: The optimized output of the U-net model for the optimized latent variable
            loss: The loss value

    """
    assert z_t.device == y.device, "z_0 and y must be on the same device"

    n_iters = opt_params["n_iters"]

    if n_iters == 0:
        loss = torch.tensor(0.0).to(z_t.device)
        return z_t, loss

    lr = lr_scale * opt_params["lr"]
    threshold = opt_params.get("threshold", 1)
    latent_consistency = opt_params.get("latent_consistency", 0)
    AHb = forward_model.adjoint(y)

    with torch.enable_grad():
        z_t_opt = z_t.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z_t_opt], lr=lr)

        if n_iters > 1:
            progress_bar = tqdm(total=n_iters, disable=not verbose, leave=False)

        for i in range(n_iters):
            optimizer.zero_grad()

            if t == 0:
                # at t=0 we just use the z_t as z_0_hat, so can do this for speedup
                z_0_hat = z_t_opt
            else:
                # Estimate z_0_hat
                model_output = unet(
                    z_t_opt,
                    t,
                    return_dict=False,
                    **unet_args,
                )[0]

                z_0_hat = predict_z0_hat(
                    prediction_type,
                    model_output,
                    z_t_opt,
                    beta_prod_t,
                    alpha_prod_t,
                )

            # Estimate image
            im_opt = autoencoder.decode(
                z_0_hat / autoencoder.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]  # [B, 3, H, W], average on the channel dimension

            loss = 0

            # complex output
            im_opt_cplx = im_opt[:, 0] + 1j * im_opt[:, 1]

            y_hat = forward_model(im_opt_cplx)

            # adjust scaling
            with torch.no_grad():
                s = torch.linalg.vecdot(
                    y_hat.flatten(1), y.flatten(1)
                ) / torch.linalg.vecdot(y_hat.flatten(1), y_hat.flatten(1))
                s = s[:, None, None, None]
            y_hat = y_hat * s  # (s + 1j * 0)

            if latent_consistency > 0:
                Dz0_proj = AHb / s[..., 0] + (im_opt_cplx - forward_model.normal(im_opt_cplx))
                Dz0_proj = torch.stack([Dz0_proj.real, Dz0_proj.imag], dim=1)
                enc_Dz0 = (
                    autoencoder.encode(Dz0_proj).latent_dist.sample()
                    * autoencoder.config.scaling_factor
                )
                consistency_loss = torch.mean((z_0_hat - enc_Dz0) ** 2)
                loss = loss + latent_consistency * consistency_loss

            batch_loss = ksp_loss(y_hat, y)
            dc_loss = torch.mean(batch_loss)

            loss = loss + dc_loss
            loss.backward()
            optimizer.step()

            if verbose:
                cl_string = (
                    ""
                    if latent_consistency == 0
                    else f" Consistency loss: {consistency_loss.item()}"
                )
                if n_iters > 1:
                    progress_bar.set_description(
                        f"DC latent optimization {i+1}/{n_iters} - Loss: {loss}, DC loss: {dc_loss}"
                        + cl_string
                    )  # Update the progress bar description with loss
                    progress_bar.update(1)

            # ideally the stopping point for different images would be different but we want batching for efficiency
            if batch_loss.max() <= threshold:
                break

    # final loss
    with torch.no_grad():
        if t == 0:
            z_0_hat = z_t_opt
        else:
            model_output = unet(
                z_t_opt,
                t,
                return_dict=False,
                **unet_args,
            )[0]
            z_0_hat = predict_z0_hat(
                prediction_type,
                model_output,
                z_t_opt,
                beta_prod_t,
                alpha_prod_t,
            )
        im_opt = autoencoder.decode(
            z_0_hat / autoencoder.config.scaling_factor,
            return_dict=False,
            generator=generator,
        )[0]  # [B, 3, H, W], average on the channel dimension
        im_opt_cplx = im_opt[:, 0] + 1j * im_opt[:, 1]
        y_hat = forward_model(im_opt_cplx)
        s = torch.linalg.vecdot(
            y_hat.flatten(1), y.flatten(1)
        ) / torch.linalg.vecdot(y_hat.flatten(1), y_hat.flatten(1))
        y_hat = y_hat * s[:, None, None, None]  # (s + 1j * 0)
        final_loss = ksp_loss(y_hat, y)
        final_loss = torch.mean(final_loss.squeeze())

    if n_iters > 1:
        progress_bar.close()

    return z_t_opt, final_loss



# -------------------- Debug functions -------------------- #
def debug_plot(x, i=0):
    plt.imshow(x[i, 0, :, :].cpu().detach().numpy(), "gray")
    # plt.colorbar()
    plt.show()
    plt.close()


def debug_encoder_plot(x, autpencoder, i=0):
    t = autpencoder(x)[0]
    plt.imshow(t[i, 0, :, :].cpu().detach().numpy(), "gray")
    # plt.colorbar()
    plt.show()
    plt.close()


def debug_latent_plot(z, autoencoder, i=0):
    t = torch.mean(
        autoencoder.decode(z / autoencoder.config.scaling_factor)[0],
        dim=1,
        keepdim=True,
    )
    plt.imshow(t[i, 0, :, :].cpu().detach().numpy(), "gray")
    # plt.colorbar()
    plt.show()
    plt.close()


def debug_latent_list_plot(zs, names, autoencoder, i=0, timestep=None):
    """
    zs are list
    """
    tstr = f"at timestep {timestep}" if timestep is not None else ""

    fig, axs = plt.subplots(1, len(zs), figsize=(20, 10))
    for k, z in enumerate(zs):
        t = torch.mean(
            autoencoder.decode(z / autoencoder.config.scaling_factor)[0],
            dim=1,
            keepdim=True,
        )
        axs[k].imshow(t[i, 0, :, :].cpu().detach().numpy(), "gray")
        axs[k].set_title(f"{names[k]} sample {i} {tstr}")

    # plt.colorbar()
    plt.show()
    plt.close()
