from copy import deepcopy

from laps.recon.reconstructor import AutoTpConfig, MedVaeLDMParams

N_step = 100
Nint = 1

# ------------------------------------------- caps-fast recon for AutoInit ------------------------------------------- #
sd_caps_medvae_4_fast = MedVaeLDMParams(
    model_name_or_path="yurman/uncond-sd2-base-complex-4",
    prediction_type="v_prediction",
    num_inference_steps=100,
    dc_type="ldps",
    opt_params={
        "latent": {
            "n_iters": 5,
            "lr": 5e-2,
            "threshold": 1,
            "latent_consistency": 0.0,
            "z_t_lam": 0.0,
        },
        "image": {
            "n_iters": 5,
            "threshold": 1e-5,
            "lambda_l2": 1e-2,
        },
    },
    scheduler_ty="DDIM",
    n_avgs=1,
    dc_latent_steps=[
        (0, 1, 1),
    ],
    dc_image_steps=None,
    start_with_prior=False,
    start_with_cg=True,
    prior_start_timestep=180,
    auto_tp_config=None,
    output_dc=True,
    latent_dc_cg_init=False,
    output_dc_config={
        "n_iters": 6,
        "threshold": 1e-5,
        "lambda_l2": 1e-4,
        "lambda_ldm": 0.02,
        "lambda_l2_from_data": False,
    },
)


# ------------------------------------------- laps model ------------------------------------------- #
sd_laps_medvae_4 = MedVaeLDMParams(
    model_name_or_path="yurman/uncond-sd2-base-complex-4",
    prediction_type="v_prediction",
    num_inference_steps=N_step,
    dc_type="plds",
    opt_params={
        "latent": {
            "n_iters": 10,
            "lr": 5e-2,
            "threshold": 1,
            "latent_consistency": 0.0,
            "z_t_lam": 0.0,  # add z_t_lam * ||z - z_t||^2 to loss
        },
        "image": {
            "n_iters": 5,
            "threshold": 1e-5,
            "lambda_l2": 1e-2,
        },
    },
    scheduler_ty="DDIM",
    n_avgs=4,
    dc_latent_steps=[
        (0, 1, Nint),
    ],
    dc_image_steps=None,
    # do resampling in between DC steps with the latest z_0_hat
    start_with_prior=True,
    prior_start_timestep="auto",

    auto_tp_config = AutoTpConfig(
        prior_recon_config = sd_caps_medvae_4_fast,
        reg_scale = 1.55,
        reg_shift = -350.0,
        clip_min = 150,
        clip_max = 550,
    ),

    # additional DC params
    output_dc=True,
    latent_dc_cg_init=False,
    output_dc_config={
        "n_iters": 6,
        "threshold": 1e-5,
        "lambda_l2": 1e-4,
        "lambda_ldm": 0.02,
        "lambda_l2_from_data": False,
    },
)

# ------------------------------------------- caps model ------------------------------------------- #
sd_caps_medvae_4 = deepcopy(sd_laps_medvae_4)
sd_caps_medvae_4.start_with_prior = False
sd_caps_medvae_4.start_with_cg = True
sd_caps_medvae_4.prior_start_timestep = 200
